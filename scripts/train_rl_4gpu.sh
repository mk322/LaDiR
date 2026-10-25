#!/usr/bin/env bash
# LaDi-RL on 4 x 80GB GPUs: 32 prompts x 16 trajectories per step, mini-batches of 4 prompts.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NGPU=4 exec "$REPO/scripts/train_rl.sh" \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ladi.train_io_timeout=3 \
    "$@"
