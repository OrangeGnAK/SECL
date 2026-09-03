import os
import random
import torch
import numpy as np
import time
import torch.distributed as dist
from torchmetrics.classification import JaccardIndex 
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


from model import contrastive_mit_b0
from losses import FastSupCon
from datasets import Sen1Floods11_DS
from transforms import Sen1Floods11_transform
from utils import setup_reproducibility, ddp_init, load_model, init_dataloaders, train_loop, validation_loop
from factories import model_factory, dataset_factory

def main():

    # TODO: Add YAML configuration
    # ------------------------------------------------------------------------

    with open('/kaggle/working/train_config.yaml', 'r') as file:
        config_file = file.read()
        
    config = yaml.load(config_file, Loader=Loader)

    
    RANDOM_SEED = config['hyperparams']['seed']
    BATCH_SIZE = config['hyperparams']['batch_size']
    LEARNING_RATE = config['hyperparams']['learning_rate']
    LAMBDA = config['hyperparams']['LAMBDA']
    START_EPOCH = 1
    TOTAL_EPOCHS = config['hyperparams']['epochs']
    
    NUM_CLASSES = config['model']['num_classes']
    model_load_path = config['model']['load_path']
    model_best_path = config['model']['best_path']
    model_last_path = config['model']['last_path']
    log_file_path = '/kaggle/working/fine_tune_fewshot_random_log.txt'
    
    # ------------------------------------------------------------------------
    
    setup_reproducibility(RANDOM_SEED)

    local_rank, world_size, device = ddp_init()
    is_main = (local_rank == 0)
    
# -----------------------------------------------------------------------------------
    #                 __________MODEL INIT__________
    
    model = model_factory(config['model']).to(device)
    
    #TODO: Change model loading logic?
    load_model(model, model_load_path, model_best_path, model_last_path, local_rank)


    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
    # Init criterions and optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=TOTAL_EPOCHS, 
    eta_min=1e-6  # Финальная микро-скорость на самом острие воронки
)
    scaler = torch.amp.GradScaler()

    ignore_index = config['criterion']['ignore_index']
    
    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
    criterion_fastsupcon = FastSupCon(ignore_index=ignore_index, num_classes=NUM_CLASSES).to(device)

# -----------------------------------------------------------------------------------
#                         DATASET SPLIT
# -----------------------------------------------------------------------------------

    ds = dataset_factory(config['dataset'])

    train_dl, val_dl, train_sampler, val_sampler = init_dataloaders(ds, RANDOM_SEED, BATCH_SIZE, local_rank, world_size)
# -----------------------------------------------------------------------------------

    # Metrics
    train_iou_metric = JaccardIndex(task="multiclass", num_classes=2, ignore_index=-1, average="none", sync_on_compute=True).to(device)
    val_iou_metric = JaccardIndex(task="multiclass", num_classes=2, ignore_index=-1, average="none", sync_on_compute=True).to(device)

    best_val_miou = 0.0
    
    # Log table head
    if is_main:
        with open(log_file_path, 'w') as f:
            f.write("Epoch | Train_Loss | Train_mIoU | Val_Land_IoU | Val_Water_IoU | Val_mIoU\n")

    # TRAIN&VALIDATION LOOP
    for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
        
        train_sampler.set_epoch(epoch - 1)
        
        train_iou_metric.reset()
        val_iou_metric.reset()

        # time measurement
        torch.cuda.synchronize()
        start_epoch_time = time.time()
        start_iter_time = time.time()

# ---------------------------------------------------------------------------------------------------
        
        # --- STEP 1: TRAIN ---
        train_loss, train_loss_ce, train_loss_fastsupcon = train_loop(
            model, train_dl, device, optimizer, criterion_ce,
            criterion_fastsupcon, scaler, LAMBDA, train_iou_metric,
            is_main, start_epoch_time, epoch)
        
        train_ious = train_iou_metric.compute()
        train_miou = train_ious.mean().item()
# ---------------------------------------------------------------------------------------------------
        
        # --- STEP 2: VALIDATION ---
        val_loss, val_miou, val_land_iou, val_water_iou = validation_loop(
            model, val_dl, device, val_iou_metric,
            criterion_ce, criterion_fastsupcon, LAMBDA, is_main, epoch)

        scheduler.step()
        
        # --- STEP 3: LOGS AND MODEL SAVING ---
        if is_main:
            print(f"\n=== RESULTS OF EPOCH {epoch} ===")
            print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Land IoU: {val_land_iou:.4f} | Val Water IoU: {val_water_iou:.4f} | Val MeanIoU: {val_miou:.4f}")
            print(f"Epoch time: {(time.time() - start_epoch_time)/60:.1f} min\n")

            # Writing to the text log
            with open(log_file_path, 'a') as f:
                f.write(f"{epoch} | {train_loss:.4f} | {train_miou:.4f} | {val_land_iou:.4f} | {val_water_iou:.4f} | {val_miou:.4f}\n")

            checkpoint = {
                'model_state_dict': model.module.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'loss': train_loss
            }

            torch.save(checkpoint, model_last_path)

            # Saving the best model yet
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save(checkpoint, model_best_path)
                print(f"--> [SAVED] Best chekpoint on the epoch {epoch} with Val mIoU: {val_miou:.4f}!")
            print("=========================\n")

        dist.barrier() 
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
