#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port 46090 \
    train_grpo.py \
    configs/grpo_config.yaml \
    "$@"
