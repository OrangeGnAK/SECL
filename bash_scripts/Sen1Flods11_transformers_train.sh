#!/usr/bin/env bash

set -e

echo "=== Train launch with MiT-B0 & Sen1Floods11 ==="
torchrun --nproc_per_node=2 train.py --yaml_config yaml_configs/mit_b0_Sen1Floods11.yaml

echo "=== Train launch with MiT-B5 & Sen1Floods11 ==="
torchrun --nproc_per_node=2 train.py --yaml_config yaml_configs/mit_b5_Sen1Floods11.yaml
