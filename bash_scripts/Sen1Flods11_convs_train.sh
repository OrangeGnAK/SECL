#!/usr/bin/env bash

set -e

echo "=== Train launch with unet_resnet34 & Sen1Floods11 ==="
torchrun --nproc_per_node=2 train.py --yaml_config yaml_configs/unet_resnet34_Sen1Floods11.yaml

echo "=== Train launch with unet_efficientnet_b7 & Sen1Floods11 ==="
torchrun --nproc_per_node=2 train.py --yaml_config yaml_configs/unet_efficientnet_b7_Sen1Floods11.yaml
