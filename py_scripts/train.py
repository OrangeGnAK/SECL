import os
import argparse 
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
from factories import model_factory, dataset_factory, weights_factory

def main():

    # ------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, help="path to the yaml config file")
    
    args = parser.parse_args()
    
    with open(args.yaml_config, 'r') as file:
        config_file = file.read()
        
    config = yaml.load(config_file, Loader=Loader)
    # print(f"/n/n/nconfig file is {config_file}/n/n/n")

    
    RANDOM_SEED = config['hyperparams']['seed']
    BATCH_SIZE = config['hyperparams']['batch_size']
    LEARNING_RATE = config['hyperparams']['learning_rate']
    LAMBDA = config['hyperparams']['LAMBDA']
    START_EPOCH = 1
    TOTAL_EPOCHS = config['hyperparams']['epochs']
    
    NUM_CLASSES = config['model']['num_classes']
    print(f"/n/n/nnum classes is {NUM_CLASSES}/n/n/n")
    model_load_path = config['model']['load_path']
    model_best_path = config['model']['best_path']
    model_last_path = config['model']['last_path']
    # LOGGING
    log_path = config['log_path']
    
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

    weights = weights_factory(config['dataset']['name'], device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=TOTAL_EPOCHS, 
    eta_min=1e-6  # Финальная микро-скорость на самом острие воронки
)
    scaler = torch.amp.GradScaler()

    ignore_index = config['criterion']['ignore_index']
    
    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights).to(device)
    criterion_fastsupcon = FastSupCon(
        ignore_index=ignore_index, num_classes=NUM_CLASSES, weight=weights).to(device)

# -----------------------------------------------------------------------------------
#                         DATASET SPLIT
# -----------------------------------------------------------------------------------

    train_ds, val_ds, _ = dataset_factory(config['dataset'], RANDOM_SEED)

    train_dl, train_sampler = init_dataloaders(train_ds, RANDOM_SEED, BATCH_SIZE, local_rank, world_size)
    val_dl, val_sampler = init_dataloaders(val_ds, RANDOM_SEED, BATCH_SIZE, local_rank, world_size)
# -----------------------------------------------------------------------------------

    # Metrics
    train_iou_metric = JaccardIndex(
        task="multiclass", num_classes=NUM_CLASSES, ignore_index=ignore_index,
        average="none", sync_on_compute=True, dist_sync_on_step=False).to(device)
    val_iou_metric = JaccardIndex(
        task="multiclass", num_classes=NUM_CLASSES, ignore_index=ignore_index,
        average="none", sync_on_compute=True, dist_sync_on_step=False).to(device)

    best_val_miou = 0.0
    
    # Setting up dictionary for logs
    logs_dict = {
        'train_loss' : [],
        'train_ce_loss' : [],
        'train_contrastive_loss' : [],
        'train_iou' : [],
        'train_miou' : [],
        'val_loss' : [],
        'val_iou' : [],
        'val_miou' : []
    }

    # TRAIN&VALIDATION LOOP
    for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
        
        train_sampler.set_epoch(epoch - 1)
        val_sampler.set_epoch(epoch - 1)
        
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
        val_loss = validation_loop(
            model, val_dl, device, val_iou_metric,
            criterion_ce, criterion_fastsupcon, LAMBDA, is_main, epoch)

        val_ious = val_iou_metric.compute()
        val_miou = val_ious.mean().item()

        scheduler.step()
        
        # --- STEP 3: LOGS AND MODEL SAVING ---
        if is_main:
            print(f"\n=== RESULTS OF EPOCH {epoch} ===")
            print(f"Train CE Loss: {train_loss_ce:.4f} | Train SECL: {train_loss_fastsupcon:.4f}")
            print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
            print(f"Epoch time: {(time.time() - start_epoch_time)/60:.1f} min\n")

            # Writing to the log
            logs_dict['train_loss'].append(train_loss)
            logs_dict['train_ce_loss'].append(train_loss_ce)
            logs_dict['train_contrastive_loss'].append(train_loss_fastsupcon)
            logs_dict['train_iou'].append(train_ious.cpu().numpy())
            logs_dict['train_miou'].append(train_miou)
            logs_dict['val_loss'].append(val_loss)
            logs_dict['val_iou'].append(val_ious.cpu().numpy())
            logs_dict['val_miou'].append(val_miou)
    

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
        
    if is_main:
        os.makedirs(log_path, exist_ok=True)
        np.save(os.path.join(log_path, 'train_loss.npy'),np.array(logs_dict['train_loss']))
        np.save(os.path.join(log_path, 'train_ce_loss.npy'),np.array(logs_dict['train_ce_loss']))
        np.save(os.path.join(log_path, 'train_contrastive_loss.npy'),np.array(logs_dict['train_contrastive_loss']))
        np.save(os.path.join(log_path, 'train_iou.npy'),np.array(logs_dict['train_iou']))
        np.save(os.path.join(log_path, 'train_miou.npy'),np.array(logs_dict['train_miou']))
        np.save(os.path.join(log_path, 'val_loss.npy'),np.array(logs_dict['val_loss']))
        np.save(os.path.join(log_path, 'val_iou.npy'),np.array(logs_dict['val_iou']))
        np.save(os.path.join(log_path, 'val_miou.npy'),np.array(logs_dict['val_miou']))
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
