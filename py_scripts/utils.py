import numpy as np
import torch
import torch.distributed as dist
import os
import random
import time
from tqdm import tqdm

def mean_std_calc(dataset, batch_size=16, num_workers=4):

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    first_batch = next(iter(loader))
    num_channels = first_batch.shape 

    channels_sum = torch.zeros(num_channels, dtype=torch.float64, device=device)
    channels_sq_sum = torch.zeros(num_channels, dtype=torch.float64, device=device)
    total_pixels = 0.0

    for batch in tqdm(loader):
        images = batch.to(torch.float64).to(device)
        
        # [B, C, H, W] -> [C, B * H * W]
        images = images.transpose(0, 1).flatten(1)
        
        channels_sum += images.sum(dim=1)
        channels_sq_sum += (images ** 2).sum(dim=1)
        total_pixels += images.shape

    mean = channels_sum / total_pixels
    var = (channels_sq_sum / total_pixels) - (mean ** 2)
    var = torch.clamp(var, min=0.0) 
    std = torch.sqrt(var)

    mean_list = [round(x, 4) for x in mean.tolist()]
    std_list = [round(x, 4) for x in std.tolist()]

    print(f"Mean: {mean_list}")
    print(f"Std:  {std_list}")
    
    return mean_list, std_list


def setup_reproducibility(seed=42):
    
    rank = int(os.environ.get("RANK", 0))
    worker_seed = seed + rank
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ddp_init():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return (local_rank, world_size, device)


def load_model(model, model_load_path, model_best_path, model_last_path, local_rank):
    
    if os.path.exists(model_load_path):
        checkpoint = torch.load(model_load_path, map_location=f"cuda:{local_rank}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if local_rank == 0:
            print(f"--> [SUCCESS] Weights are loaded.")
    else:
        if local_rank == 0:
            print(f"--> [WARNING] (Random Init Weights).")


def init_dataloaders(ds, RANDOM_SEED, BATCH_SIZE, local_rank, world_size):
    
    # Dataset split
    train_size = 314
    val_size = 120
    test_size = 12
    
    rn_gen = torch.Generator().manual_seed(RANDOM_SEED)
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [train_size, val_size, test_size], generator=rn_gen)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world_size, rank=local_rank, shuffle=True, seed=RANDOM_SEED
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=False
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds, num_replicas=world_size, rank=local_rank, shuffle=False, seed=RANDOM_SEED
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2, pin_memory=True, drop_last=False
    )

    return (train_dataloader, val_dataloader, train_sampler, val_sampler)


def train_loop(model, train_loader, device, optimizer, criterion_ce,
               criterion_fastsupcon, scaler, LAMBDA, train_iou_metric,
               is_main, start_epoch_time, epoch):
    
    running_loss = 0.0
    running_loss_ce = 0.0          
    running_loss_fastsupcon = 0.0  
    total_batches = 0
    
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
            
        optimizer.zero_grad()
        
        is_batch_valid = (masks == 0).sum() >= 100 and (masks == 1).sum() >= 100
        
        if is_batch_valid:
            # Branch A valid variant
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                contrast_features, segmentation_features = model(images)
                loss_ce = criterion_ce(segmentation_features, masks)
                loss_fastsupcon = criterion_fastsupcon(contrast_features, masks)
                loss = loss_ce + LAMBDA * loss_fastsupcon

            loss_ce_value = loss_ce.item()
            loss_fastsupcon_value = loss_fastsupcon.item()
            loss_value = loss.item()
                
            scaler.scale(loss).backward()
            
            # 1. Unscale grad to clip
            scaler.unscale_(optimizer)
            # 2. clipping gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        
            running_loss += loss_value
            running_loss_ce += loss_ce_value
            running_loss_fastsupcon += loss_fastsupcon_value
            total_batches += 1     
                
            with torch.inference_mode():
                pred_argmax = torch.argmax(segmentation_features, dim=1)
                train_iou_metric.update(pred_argmax, masks)
                
        else:
            # Branch B: Fake step for proper DDP work
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                contrast_features, segmentation_features = model(images)
                loss = criterion_ce(segmentation_features, masks) * 0.0
                
            scaler.scale(loss).backward()
            # unscaling
            scaler.unscale_(optimizer)
            # cliping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_batches += 1

# ---------------------------------------------------------------------------------------------------
#                                            TIME & PRINT
        if is_main:
            torch.cuda.synchronize()
            current_time = time.time()
            elapsed_time_min = (current_time - start_epoch_time) / 60
            avg_iter_time = elapsed_time_min / total_batches
            batches_left = len(train_loader) - total_batches
            eta_min = batches_left * avg_iter_time

            if total_batches % 5 == 0 or total_batches == len(train_loader):
                print(f"Epoch {epoch} | Train Batch {total_batches}/{len(train_loader)} | Loss Value {loss.item()} | ETA: {eta_min:.1f}m")
# ---------------------------------------------------------------------------------------------------

    # TODO: All_reduce
    
    final_avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    final_avg_ce = running_loss_ce / total_batches if total_batches > 0 else 0.0
    final_avg_fastsupcon = running_loss_fastsupcon / total_batches if total_batches > 0 else 0.0

    return final_avg_loss, final_avg_ce, final_avg_fastsupcon




def validation_loop(model, val_dataloader, device, val_iou_metric,
                    criterion_ce, criterion_fastsupcon, LAMBDA, is_main, epoch):

    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_fastsupcon = 0.0
    total_batches = 0

    model.eval()
    with torch.inference_mode():
        for images, masks in val_dataloader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
            if (masks == 0).sum() == 0 or (masks == 1).sum() == 0:
                continue
                
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                contrast_features, segmentation_features = model(images)
                loss_ce = criterion_ce(segmentation_features, masks)
                loss_fastsupcon = criterion_fastsupcon(contrast_features, masks)
                loss = loss_ce + LAMBDA * loss_fastsupcon

            loss_ce_value = loss_ce.item()
            loss_fastsupcon_value = loss_fastsupcon.item()
            loss_value = loss.item()

            running_loss += loss_value
            running_loss_ce += loss_ce_value
            running_loss_fastsupcon += loss_fastsupcon_value
            total_batches += 1  

            pred_argmax = torch.argmax(segmentation_features, dim=1)
            val_iou_metric.update(pred_argmax, masks)

    losses = torch.tensor(
        [running_loss, running_loss_ce, running_loss_fastsupcon, float(total_batches)],
        device=device
    )

    # dist.all_reduce(losses, op=dist.ReduceOp.SUM)

    loss = losses[0].item() / losses[3].item()
    loss_ce = losses[1].item() / losses[3].item()
    loss_fastsupcon = losses[2].item() / losses[3].item()
    
    val_ious = val_iou_metric.compute()
    val_land_iou = val_ious[0].item()
    val_water_iou = val_ious[1].item()
    val_miou = val_ious.mean().item()

    if is_main:
        print(f"\n[VAL EPOCH {epoch}] Finished!")
        print(f"--> Global Loss: {loss:.4f} (CE: {loss_ce:.4f}, Contrast: {loss_fastsupcon:.4f})")
        print(f"--> IoU Land (Суша): {val_land_iou:.4f} | IoU Water (Вода): {val_water_iou:.4f} | mIoU: {val_miou:.4f}\n")

    return loss, val_miou, val_land_iou, val_water_iou

