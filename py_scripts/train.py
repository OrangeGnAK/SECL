import os
import random
import torch
import numpy as np
import time
import torch.distributed as dist
from torchmetrics.classification import JaccardIndex 

from model import contrastive_mit_b0
from losses import FastSupCon
from datasets import Sen1Floods11_DS
from transforms import Sen1Floods11_transform
from utils import setup_reproducibility, ddp_init, load_model, init_dataloaders, train_loop, validation_loop

def main():

    # TODO: Add YAML configuration
    # ------------------------------------------------------------------------
    RANDOM_SEED = 42
    BATCH_SIZE = 6
    LEARNING_RATE = 5e-5
    LAMBDA = 0.5
    START_EPOCH = 1
    TOTAL_EPOCHS = 60
    

    model_load_path = '/kaggle/input/models/orangeggnt/bilk-epochs/pytorch/default/3/bilk_2GPU_model_epoch_2th'
    model_best_path = '/kaggle/working/bilk_fewshot_random_best.pth'
    model_last_path = '/kaggle/working/bilk_fewshot_random_last.pth'
    log_file_path = '/kaggle/working/fine_tune_fewshot_random_log.txt'

    # TODO: calculate mean & std for each dataset. Those are BEN-14K mean & std
    mean = np.array([627.1802, 676.7937, 428.5101, 1091.0465,  -17.2513,  -11.1336])
    std =  np.array([419.1856, 272.5272, 229.5443, 385.9456,   3.7207,   3.5236])

    # ------------------------------------------------------------------------
    
    setup_reproducibility(RANDOM_SEED)

    local_rank, world_size, device = ddp_init()
    is_main = (local_rank == 0)

    
# -----------------------------------------------------------------------------------
    #                 __________MODEL INIT__________
    
    model = contrastive_mit_b0(in_channels=6, projection_dim=128).to(device)

    
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

    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
    criterion_fastsupcon = FastSupCon(ignore_index=-1, num_classes=2).to(device)


# -----------------------------------------------------------------------------------
#                         DATASET SPLIT
# -----------------------------------------------------------------------------------
    transform = Sen1Floods11_transform(mean, std)
    
    ds = Sen1Floods11_DS(
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S1Hand',
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S2Hand',
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/LabelHand',
        transform=transform
    )

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
