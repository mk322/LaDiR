"""Stage-wise LaDiR SFT on precomputed VAE latents (the VAE is never loaded).

  stage 1: projectors / output head / special embeddings only, backbone frozen   (sft/configs/stage1.yaml)
  stage 2: everything, initialised strictly from stage 1's final checkpoint      (sft/configs/stage2.yaml)

Each stage is its own process; stage 2 points `model.init_from` at stage 1's `final/` directory, or at a released
stage-2 checkpoint (e.g. mk322/LaDiR-Code-8B) to continue training it.

    torchrun --nproc_per_node=8 -m sft.train_stagewise --config sft/configs/stage1.yaml [key=value ...]
"""

import argparse

import torch.distributed as dist

from .config import load_config
from .trainer import run

STAGES = {"projectors": 1, "all": 2}


def main(argv=None):
    p = argparse.ArgumentParser(description="stage-wise LaDiR SFT (one stage per invocation)")
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.lr=1e-5")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    if cfg.model.latent_source != "vae":
        raise SystemExit("train_stagewise needs model.latent_source=vae")
    if STAGES[cfg.model.trainable] == 2 and cfg.model.init_from is None:
        raise SystemExit("stage 2 needs model.init_from (stage 1 final/ dir or a released stage-2 checkpoint)")
    path = run(cfg)
    if dist.get_rank() == 0:
        print(f"final checkpoint: {path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
