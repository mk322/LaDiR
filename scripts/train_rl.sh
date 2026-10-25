#!/usr/bin/env bash
# LaDi-RL on one node. Extra arguments are Hydra overrides of rl/config/ladi_rl_trainer.yaml.
#   LADIR_INIT    checkpoint to start from (default mk322/LaDiR-Code-8B)
#   LADI_RL_DATA  dir with train.parquet / validation.parquet (python -m rl.data.fetch_hf --out_dir <dir>)
#   LADI_RL_OUT   output root;  NGPU  GPUs (default 8);  PYTHON  interpreter of the RL env
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
MODEL_DIR=${MODEL_DIR:-$REPO/outputs/rl_model}
NGPU=${NGPU:-8}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export LADI_RL_DATA=${LADI_RL_DATA:-$REPO/data/ladi_rl}
LOGGER='["console"]'
if [ -n "${WANDB_API_KEY:-}" ]; then LOGGER='["console","wandb"]'; fi
cd "$REPO"
if [ ! -f "$MODEL_DIR/model_index.json" ]; then
    "$PY" -m rl.ladi_rl.model_dir "${LADIR_INIT:-mk322/LaDiR-Code-8B}" "$MODEL_DIR"
fi
exec "$PY" -m rl.ladi_rl.main \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    trainer.n_gpus_per_node="$NGPU" \
    actor_rollout_ref.rollout.agent.num_workers="$NGPU" \
    trainer.logger="$LOGGER" \
    "$@"
