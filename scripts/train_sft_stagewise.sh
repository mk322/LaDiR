#!/usr/bin/env bash
# Stage-wise LaDiR SFT on one 8-GPU node:
#   [pretokenize] -> precompute VAE latents (once) -> stage 1 (projectors) -> stage 2 (full, strict init from stage 1)
#
# Required env : LADIR_OUTPUT (run outputs), LADIR_DATA (pretokenized data + latents)
# Optional env : LADIR_RAW_JSON  raw json, only needed if $LADIR_DATA/ling_coder_tok does not exist yet
#                LADIR_VAE       VAE source: a vae.train export (.../weights/step_N.safetensors) or a LaDiR
#                                checkpoint with vae.* tensors (default mk322/LaDiR-Code-8B)
#                STAGES          stages to run (default "1 2"; e.g. STAGES=2 to (re)run stage 2 only)
#                TORCHRUN_ARGS   default "--standalone --nproc_per_node=8"
#                WANDB_PROJECT / WANDB_ENTITY / WANDB_API_KEY (logging is on iff WANDB_PROJECT is set), HF_TOKEN
# Secrets are read from the environment only. Extra args are OmegaConf overrides applied to every stage, e.g.
#   STAGES=2 scripts/train_sft_stagewise.sh model.init_from=mk322/LaDiR-Code-8B train.lr=2e-5
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LADIR_OUTPUT:?set LADIR_OUTPUT}"
: "${LADIR_DATA:?set LADIR_DATA}"
export LADIR_VAE="${LADIR_VAE:-mk322/LaDiR-Code-8B}"
STAGES="${STAGES:-1 2}"
read -r -a TR <<< "${TORCHRUN_ARGS:---standalone --nproc_per_node=8}"
TOK="$LADIR_DATA/ling_coder_tok"
LAT="$LADIR_DATA/ling_coder_latents"
mkdir -p "$LADIR_OUTPUT/logs"

if [ ! -f "$TOK/meta.json" ]; then
    : "${LADIR_RAW_JSON:?$TOK missing: set LADIR_RAW_JSON to the raw json to pretokenize}"
    python -m sft.data --input "$LADIR_RAW_JSON" --out_dir "$TOK" --tokenizer Qwen/Qwen3-8B-Base \
        --max_prompt_length 512 --max_reasoning_length 1024 --max_answer_length 1024
fi
if [ ! -f "$LAT/meta.json" ]; then
    torchrun "${TR[@]}" -m sft.precompute_latents --data_dir "$TOK" --out_dir "$LAT" --vae "$LADIR_VAE" \
        2>&1 | tee -a "$LADIR_OUTPUT/logs/precompute_latents.log"
fi
for stage in $STAGES; do
    torchrun "${TR[@]}" -m sft.train_stagewise --config "sft/configs/stage${stage}.yaml" "$@" \
        2>&1 | tee -a "$LADIR_OUTPUT/logs/stage${stage}.log"
done
