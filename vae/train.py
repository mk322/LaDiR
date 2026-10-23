"""VAE training (accelerate; DeepSpeed ZeRO-3 optional via `accelerate launch --use_deepspeed`).

  accelerate launch [--use_deepspeed --deepspeed_config_file vae/ds_zero3.json] -m vae.train \
      --config vae/configs/vae_4b_8b_64.yaml [dot.key=value overrides ...]

* One optimizer step = `grad_accum` micro-batches via `accelerator.accumulate` (also correct under DeepSpeed,
  where the engine owns the step; `optimizer.step()` is then a no-op wrapper).
* `global_step` counts optimizer steps; logging/eval/save fire only on `sync_gradients`, never at step 0.
* `accelerator.save_state` checkpoints model, optimizer, LR scheduler, RNG, accelerate's micro-step counter and our
  TrainState (epoch, micro-batches consumed in the epoch); resume restores all of it and skips consumed batches
  of the (seeded, per-epoch) shuffle.
* W&B is used only when `wandb.enabled` and WANDB_API_KEY (or WANDB_MODE=offline) is set in the environment.
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import time
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .data import VAECollator, build_datasets
from .model import ReasoningVAE, VAEConfig, build_vae, config_to_dict, load_state_file

DEFAULTS = {
    "model": {},                            # VAEConfig fields
    "data": {"path": None, "cache_path": None, "max_tokens": 512, "language": "Python", "val_size": 100,
             "split_seed": None, "overlength": "drop", "tokenizer": None},
    "train": {"output_dir": "outputs/vae", "seed": 42, "micro_batch_size": 8, "grad_accum": 1, "epochs": 5,
              "max_steps": None, "lr": 2e-5, "weight_decay": 0.01, "betas": [0.9, 0.999],
              "lr_scheduler": "cosine", "warmup_steps": 500, "min_lr_ratio": 0.0, "max_grad_norm": 1.0,
              "kl_weight": 1e-6, "mixed_precision": "bf16", "dtype": "bfloat16", "eval_every": 500,
              "eval_batches": 100, "save_every": 5000, "keep_last": 2, "export_weights": True,
              "resume_from": "latest", "init_from": None, "log_every": 10, "num_workers": 2, "cpu": False},
    "wandb": {"enabled": True, "project": "vae-reasoning-compressor", "run_name": None},
}


def load_config(path: Optional[str] = None, overrides=()):
    cfg = OmegaConf.create(DEFAULTS)
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def vae_config(cfg) -> VAEConfig:
    return VAEConfig(**OmegaConf.to_container(cfg.model, resolve=True))


class TrainState:
    """Registered with accelerator.register_for_checkpointing so resume restores the data position."""

    def __init__(self):
        self.global_step, self.epoch, self.batches_in_epoch = 0, 0, 0

    def state_dict(self):
        return {"global_step": self.global_step, "epoch": self.epoch, "batches_in_epoch": self.batches_in_epoch}

    def load_state_dict(self, d):
        self.global_step, self.epoch, self.batches_in_epoch = d["global_step"], d["epoch"], d["batches_in_epoch"]


def lr_lambda(kind: str, warmup: int, total: int, min_ratio: float):
    def f(step):
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        if kind == "linear":
            return min_ratio + (1 - min_ratio) * (1 - prog)
        if kind == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        raise ValueError(kind)
    return f


def checkpoint_dirs(output_dir: str):
    dirs = glob.glob(os.path.join(output_dir, "checkpoints", "step_*"))
    return sorted(dirs, key=lambda d: int(re.findall(r"step_(\d+)$", d)[0]))


def _ds_zero3(accelerator) -> bool:
    p = getattr(accelerator.state, "deepspeed_plugin", None)
    return p is not None and p.zero_stage == 3


def load_init_weights(model: ReasoningVAE, path: str, zero3: bool):
    state = load_state_file(path)
    if zero3:   # params may already be partitioned by zero.Init
        import deepspeed
        with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                model.load_vae_state(state)
    else:
        model.load_vae_state(state)


@torch.no_grad()
def evaluate(accelerator, model, loader, kl_weight: float, max_batches: int):
    """Token-weighted CE / teacher-forced token accuracy over up to max_batches per process, reduced over ranks."""
    model.eval()
    sums = torch.zeros(5, dtype=torch.float64, device=accelerator.device)   # ce*n, n, correct, kl, batches
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = model(**batch, kl_weight=kl_weight)
        sums += torch.tensor([out["ce_loss"].item() * out["n_tokens"], out["n_tokens"], out["n_correct"],
                              out["kl_loss"].item(), 1], dtype=torch.float64, device=accelerator.device)
    sums = accelerator.reduce(sums, "sum")
    model.train()
    n = max(sums[1].item(), 1)
    ce, kl = sums[0].item() / n, sums[3].item() / max(sums[4].item(), 1)
    return {"val/ce_loss": ce, "val/token_acc": sums[2].item() / n, "val/kl_loss": kl,
            "val/loss": ce + kl_weight * kl, "val/tokens": sums[1].item()}


def save_checkpoint(accelerator, model, cfg, state: TrainState):
    out = cfg.train.output_dir
    ckpt = os.path.join(out, "checkpoints", f"step_{state.global_step}")
    accelerator.save_state(ckpt)
    if cfg.train.export_weights:
        sd = accelerator.get_state_dict(model)   # gathers ZeRO-3 shards (stage3_gather_16bit_weights_on_model_save)
        if accelerator.is_main_process:
            from safetensors.torch import save_file
            os.makedirs(os.path.join(out, "weights"), exist_ok=True)
            seen, clean = set(), {}
            for k, v in sd.items():          # drop aliased (tied) tensors; the loader tolerates them missing
                key = (v.data_ptr(), v.shape) if v.numel() else (k,)
                if key in seen:
                    continue
                seen.add(key)
                clean[k] = v.detach().contiguous().cpu()
            save_file(clean, os.path.join(out, "weights", f"step_{state.global_step}.safetensors"),
                      metadata={"vae_config": json.dumps(config_to_dict(accelerator.unwrap_model(model).config))})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and cfg.train.keep_last:
        for d in checkpoint_dirs(out)[:-cfg.train.keep_last]:
            shutil.rmtree(d, ignore_errors=True)
    accelerator.wait_for_everyone()
    return ckpt


def make_accelerator(cfg):
    from accelerate import Accelerator
    from accelerate.utils import DataLoaderConfiguration
    use_wandb = bool(cfg.wandb.enabled) and bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE"))
    mp = cfg.train.mixed_precision
    mp = "no" if mp in (False, None) else str(mp)     # YAML reads a bare `no` as False
    return Accelerator(gradient_accumulation_steps=cfg.train.grad_accum, mixed_precision=mp,
                       log_with="wandb" if use_wandb else None, project_dir=cfg.train.output_dir, cpu=cfg.train.cpu,
                       dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True)), use_wandb


def train(cfg, model: Optional[ReasoningVAE] = None, train_ds=None, val_ds=None, pad_token_id: Optional[int] = None,
          accelerator=None, stop_after_steps: Optional[int] = None):
    """Run training. model/datasets/pad id may be passed in; otherwise built from cfg.
    stop_after_steps: return after this many optimizer steps in this call."""
    from accelerate.utils import set_seed
    tc = cfg.train
    os.makedirs(tc.output_dir, exist_ok=True)
    use_wandb = False
    if accelerator is None:
        accelerator, use_wandb = make_accelerator(cfg)
    set_seed(tc.seed)

    if train_ds is None:
        from transformers import AutoTokenizer
        tok_name = cfg.data.tokenizer or cfg.model.get("encoder_name", VAEConfig.encoder_name)
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        with accelerator.main_process_first():
            train_ds, val_ds = build_datasets(cfg.data.path, tokenizer, cfg.data.max_tokens, cfg.data.language,
                                              cfg.data.val_size, cfg.data.split_seed, cfg.data.cache_path,
                                              cfg.data.overlength)
        pad_token_id = tokenizer.pad_token_id
    if model is None:
        vcfg = vae_config(cfg)
        model = build_vae(vcfg, pretrained=tc.init_from is None, dtype=getattr(torch, tc.dtype))
        if tc.init_from:
            load_init_weights(model, tc.init_from, _ds_zero3(accelerator))
    accelerator.print(f"train={len(train_ds)} val={len(val_ds)} trainable={model.trainable_summary()}")

    collate = VAECollator(pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=tc.micro_batch_size, shuffle=True, collate_fn=collate,
                              num_workers=tc.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tc.micro_batch_size, shuffle=False, collate_fn=collate,
                            num_workers=tc.num_workers)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tc.lr, betas=tuple(tc.betas), weight_decay=tc.weight_decay)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / tc.grad_accum))   # last partial window syncs too
    total_steps = tc.max_steps or tc.epochs * steps_per_epoch
    # built on the prepared optimizer and stepped once per optimizer step (not passed through prepare, which
    # would step it num_processes times per call)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(tc.lr_scheduler, tc.warmup_steps, total_steps, tc.min_lr_ratio))
    state = TrainState()
    accelerator.register_for_checkpointing(state)
    accelerator.register_for_checkpointing(scheduler)

    resume = tc.resume_from
    if resume == "latest":
        dirs = checkpoint_dirs(tc.output_dir)
        resume = dirs[-1] if dirs else None
    if resume:
        accelerator.load_state(resume)
        accelerator.print(f"resumed from {resume}: {state.state_dict()}")
    if use_wandb:
        accelerator.init_trackers(os.environ.get("WANDB_PROJECT", cfg.wandb.project),
                                  config=OmegaConf.to_container(cfg, resolve=True),
                                  init_kwargs={"wandb": {"name": cfg.wandb.run_name,
                                                         "entity": os.environ.get("WANDB_ENTITY")}})

    model.train()
    start_step, t0 = state.global_step, time.time()
    run = {"ce": 0.0, "kl": 0.0, "n": 0, "correct": 0, "micro": 0}
    history = []
    done = state.global_step >= total_steps
    while not done and state.epoch < tc.epochs:
        train_loader.set_epoch(state.epoch)
        loader = train_loader
        if state.batches_in_epoch:
            loader = accelerator.skip_first_batches(train_loader, state.batches_in_epoch)
            loader.set_epoch(state.epoch)   # the new loader starts at iteration 0, which would reset the shuffle
        for batch in loader:
            with accelerator.accumulate(model):
                out = model(**batch, kl_weight=tc.kl_weight)
                accelerator.backward(out["loss"])
                grad_norm = None
                if accelerator.sync_gradients and tc.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(params, tc.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            state.batches_in_epoch += 1
            run["ce"] += out["ce_loss"].item()
            run["kl"] += out["kl_loss"].item()
            run["n"] += out["n_tokens"]
            run["correct"] += out["n_correct"]
            run["micro"] += 1
            if not accelerator.sync_gradients:
                continue
            state.global_step += 1
            step = state.global_step
            if step % tc.log_every == 0 or step == total_steps:
                m = max(run["micro"], 1)
                logs = {"train/ce_loss": run["ce"] / m, "train/kl_loss": run["kl"] / m,
                        "train/token_acc": run["correct"] / max(run["n"], 1), "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": state.epoch, "train/steps_per_s": (step - start_step) / (time.time() - t0)}
                if grad_norm is not None:
                    logs["train/grad_norm"] = float(grad_norm)
                accelerator.log(logs, step=step)
                accelerator.print(f"step {step}/{total_steps} " + " ".join(f"{k[6:]}={v:.4g}" for k, v in logs.items()))
                history.append({"step": step, **logs})
                run = {"ce": 0.0, "kl": 0.0, "n": 0, "correct": 0, "micro": 0}
            if tc.eval_every and step % tc.eval_every == 0:
                ev = evaluate(accelerator, model, val_loader, tc.kl_weight, tc.eval_batches)
                accelerator.log(ev, step=step)
                accelerator.print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in ev.items()))
                history.append({"step": step, **ev})
            if tc.save_every and step % tc.save_every == 0:
                save_checkpoint(accelerator, model, cfg, state)
            if step >= total_steps or (stop_after_steps and step - start_step >= stop_after_steps):
                done = True
                break
        else:
            state.epoch += 1
            state.batches_in_epoch = 0
    finished = state.global_step >= total_steps or state.epoch >= tc.epochs
    if finished and not (tc.save_every and state.global_step % tc.save_every == 0):
        save_checkpoint(accelerator, model, cfg, state)
    accelerator.end_training()
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler, "state": state,
            "accelerator": accelerator, "history": history}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "vae_4b_8b_64.yaml"))
    ap.add_argument("overrides", nargs="*", help="omegaconf dotlist, e.g. train.lr=1e-5 model.num_latents=32")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
