#!/bin/bash
set -e

MASTER_ADDR=$(hostname -s)

export PATH=$CUDA_HOME/bin:$PATH
# export WANDB_API_KEY=your_wandb_api_key_here
TASK="vae_train"

srun -e logs/$TASK.err -o logs/$TASK.out sh -c "python -m torch.distributed.run \
    --node_rank \$((SLURM_PROCID)) \
    --nnodes 1 \
    --nproc_per_node 8 \
    --master_addr \$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    --master_port 46079 \
    ../vae/train_vae.py \
    --run_name vae_train \
    --model_name_or_path \"meta-llama/Llama-3.1-8B\" \
    --lora_r 512 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --output_dir \"../checkpoints\" \
    --input_type \"full_format\" \
    --test_size 10 \
    --max_steps 30000 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --lr_scheduler_type \"cosine\" \
    --lr_scheduler_kwargs '{\"num_cycles\": 1}' \
    --warmup_steps 1000 \
    --optim \"adamw_torch\" \
    --weight_decay 0.03 \
    --eval_strategy \"no\" \
    --eval_interval 1 \
    --ddp_backend \"nccl\" \
    --fsdp \"hybrid_shard auto_wrap\" \
    --fsdp_config '{\"backward_prefetch\": \"backward_pre\", \"forward_prefetch\": true, \"cpu_ram_efficient_loading\": true, \"sync_module_states\": true, \"transformer_layer_cls_to_wrap\": [\"LlamaDecoderLayer\"], \"use_orig_params\": true, \"activation_checkpointing\": false}' \
    --notes \"VAE training experiment\""
