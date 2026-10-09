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


# -------------------------------------------------------------------------------------
#                                           CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, help="path to the yaml config file")
    
    args = parser.parse_args()
    
    with open(args.yaml_config, 'r') as file:
        config_file = file.read()
        
    config = yaml.load(config_file, Loader=Loader)

    RANDOM_SEED = config['hyperparams']['seed']
    BATCH_SIZE = config['hyperparams']['batch_size']
    LAMBDA = config['hyperparams']['LAMBDA']
    NUM_CLASSES = config['model']['num_classes']
    model_load_path = config['model']['load_path']
    log_path = config['log_path']

    local_rank, world_size, device = ddp_init()
    is_main = (local_rank == 0)

    setup_reproducibility(RANDOM_SEED)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
#                                           MODEL INIT
    model = model_factory(config['model']).to(device)

    load_model(model, model_load_path, local_rank)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
#                                           LOSSES INIT

    weights = weights_factory(config['dataset']['name'], device)
    
    ignore_index = config['criterion']['ignore_index']
    
    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)#, weight=weights).to(device)
    criterion_fastsupcon = FastSupCon(ignore_index=ignore_index, num_classes=NUM_CLASSES).to(device)#, weight=weights).to(device)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
#                                           DATASET AND LOADER INIT

    _, _, test_ds = dataset_factory(config['dataset'], RANDOM_SEED, train_mode=False)

    test_dl, test_sampler = init_dataloaders(test_ds, RANDOM_SEED, BATCH_SIZE, local_rank, world_size)

# -------------------------------------------------------------------------------------


    test_iou_metric = JaccardIndex(
            task="multiclass", num_classes=NUM_CLASSES, ignore_index=ignore_index,
            average="none", sync_on_compute=True, dist_sync_on_step=False).to(device)

    logs_dict = {
            'test_loss' : [],
            'test_iou' : [],
            'test_miou' : []
        }

# -------------------------------------------------------------------------------------
#                                           TEST LOOP

    test_sampler.set_epoch(0)
    test_iou_metric.reset()

    test_loss = validation_loop(
            model, test_dl, device, test_iou_metric,
            criterion_ce, criterion_fastsupcon, LAMBDA, is_main, 0)

    test_ious = test_iou_metric.compute()
    test_miou = test_ious.mean().item()


    if is_main:
        print(f"\n=== RESULTS ===")
        print(f"Test Loss: {test_loss:.4f} | Test mIoU: {test_miou:.4f}")

        # Writing to the log
        logs_dict['test_loss'].append(test_loss)
        logs_dict['test_iou'].append(test_ious.cpu().numpy())
        logs_dict['test_miou'].append(test_miou)

    dist.barrier()

# -------------------------------------------------------------------------------------

    if is_main:
        os.makedirs(log_path, exist_ok=True)
        np.save(os.path.join(log_path, 'test_loss.npy'),np.array(logs_dict['test_loss']))
        np.save(os.path.join(log_path, 'test_iou.npy'),np.array(logs_dict['test_iou']))
        np.save(os.path.join(log_path, 'test_miou.npy'),np.array(logs_dict['test_miou']))
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()