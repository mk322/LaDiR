#!/usr/bin/env bash
# Train the reasoning-block VAE on one node.
#   scripts/train_vae.sh data.path=/path/to/data.json [more dot.key=value overrides]
# Env: NUM_GPUS (default 8), CONFIG (default vae/configs/vae_4b_8b_64.yaml), DEEPSPEED=0 to disable ZeRO-3.
# Secrets (WANDB_API_KEY, HF_TOKEN) are read from the environment only; never put them in this file.
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-vae/configs/vae_4b_8b_64.yaml}
DS_ARGS=()
if [[ "${DEEPSPEED:-1}" == "1" ]]; then
  DS_ARGS=(--use_deepspeed --deepspeed_config_file vae/ds_zero3.json --zero3_init_flag true)
fi

exec accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  "${DS_ARGS[@]}" -m vae.train --config "$CONFIG" "$@"
