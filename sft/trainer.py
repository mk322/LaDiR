"""SFT training loop.

Plain PyTorch + FSDP2 (see sft/distributed.py).

Output layout (cfg.output_dir):
    sft_config.yaml                 resolved config of the run
    train_log.jsonl                 one json line per logging step (rank 0)
    checkpoint-<step>/              model.safetensors + ladi_config.json   (loadable with ladir.checkpoint.load_ladir)
                                    trainer_state.json, optim/ (DCP, sharded), rng_rank<r>.pt   (resume)
    final/                          model.safetensors + ladi_config.json + trainer_state.json
Resume picks the checkpoint with the highest step number (numeric, not lexical).
"""

import json
import os
import random
import re
import shutil
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
                                                     set_model_state_dict, set_optimizer_state_dict)
from torch.utils.data import DataLoader, Sampler

from ladir.tokenizer import load_tokenizer

from .config import SFTConfig, to_dict, to_yaml
from .data import SFTCollator, SFTDataset
from .distributed import DTYPES, DistInfo, clean_fqn, make_mesh, setup_distributed, shard_model
from .modeling import build_model, check_state, initial_state, reinit_rotary_buffers, set_trainable, write_checkpoint
from .vae_weights import load_vae_tensors

CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
STATE_FILE = "trainer_state.json"


# ----------------------------------------------------------------------------------------------------------------
# data order
# ----------------------------------------------------------------------------------------------------------------
class ResumableSampler(Sampler):
    """Deterministic per-epoch shuffle, split across ranks, that can start mid-epoch (after `skip` samples)."""

    def __init__(self, num_samples: int, rank: int, world_size: int, seed: int):
        self.n, self.rank, self.world, self.seed = num_samples, rank, world_size, seed
        self.per_rank = num_samples // world_size
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0):
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g)[:self.per_rank * self.world]
        return iter(perm[self.rank::self.world][self.skip:].tolist())

    def __len__(self):
        return self.per_rank - self.skip


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    steps = [(int(m.group(1)), d) for d in os.listdir(output_dir) if (m := CKPT_RE.match(d))
             and os.path.exists(os.path.join(output_dir, d, STATE_FILE))]
    return os.path.join(output_dir, max(steps)[1]) if steps else None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_initialized():
        torch.cuda.set_rng_state(state["cuda"])


def make_optimizer(model, cfg: SFTConfig):
    t = cfg.train
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)   # biases / norm weights: no weight decay
    groups = [{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    optim = torch.optim.AdamW(groups, lr=t.lr, betas=(t.adam_beta1, t.adam_beta2), eps=t.adam_eps)
    warmup = t.warmup_steps
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, s / warmup) if warmup > 0 else 1.0)
    return optim, sched


class Logger:
    def __init__(self, cfg: SFTConfig, info: DistInfo):
        self.info, self.path, self.wandb = info, os.path.join(cfg.output_dir, "train_log.jsonl"), None
        if info.is_main and os.environ.get("WANDB_PROJECT"):
            import wandb
            id_file = os.path.join(cfg.output_dir, "wandb_run_id")
            run_id = open(id_file).read().strip() if os.path.exists(id_file) else wandb.util.generate_id()
            with open(id_file, "w") as f:
                f.write(run_id)
            self.wandb = wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ.get("WANDB_ENTITY"),
                                    name=cfg.run_name, id=run_id, resume="allow", config=to_dict(cfg))

    def log(self, step: int, metrics: Dict[str, float]):
        if not self.info.is_main:
            return
        rec = {"step": step, **metrics}
        print(json.dumps(rec), flush=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()


# ----------------------------------------------------------------------------------------------------------------
# trainer
# ----------------------------------------------------------------------------------------------------------------
class SFTTrainer:
    def __init__(self, cfg: SFTConfig, device=None):
        self.cfg = cfg
        self.info = info = setup_distributed(device)
        self.out = cfg.output_dir
        if info.is_main:
            os.makedirs(self.out, exist_ok=True)
            with open(os.path.join(self.out, "sft_config.yaml"), "w") as f:
                f.write(to_yaml(cfg))
        seed_everything(cfg.seed)
        self.tokenizer = load_tokenizer(cfg.model.backbone)
        d = cfg.data
        self.dataset = SFTDataset(
            d.train_dir, d.latents_dir, d.num_samples, cfg.seed,
            expected_lengths={"max_prompt_length": d.max_prompt_length, "max_reasoning_length": d.max_reasoning_length,
                              "max_answer_length": d.max_answer_length},
            expected_vocab_size=len(self.tokenizer))
        if self.dataset.latents is not None:
            lm = self.dataset.latents.meta
            if (lm["num_latents"], lm["latent_dim"]) != (cfg.model.num_latents, cfg.model.latent_dim):
                raise ValueError(f"latents in {d.latents_dir} are {lm['num_latents']}x{lm['latent_dim']}, "
                                 f"model expects {cfg.model.num_latents}x{cfg.model.latent_dim}")
            if info.is_main and cfg.model.vae is not None and lm["vae_source"] != cfg.model.vae.source:
                print(f"[sft] warning: latents were encoded with {lm['vae_source']}, checkpoints will carry the VAE "
                      f"from {cfg.model.vae.source}", flush=True)
        t = cfg.train
        self.steps_per_epoch = len(self.dataset) // info.world_size // t.per_device_batch_size // t.grad_accum_steps
        if self.steps_per_epoch == 0:
            raise ValueError(f"{len(self.dataset)} samples is less than one global batch")
        self.total_steps = cfg.train.num_epochs * self.steps_per_epoch
        if t.max_steps is not None:
            self.total_steps = min(self.total_steps, t.max_steps)
        self.resume_dir = latest_checkpoint(self.out) if t.resume else None
        if cfg.model.vae is not None and info.is_main:   # fail early, before any training, on a bad VAE source
            self._vae_tensors()
        self._build_model()
        self.optim, self.sched = make_optimizer(self.model, cfg)
        self.step = 0
        if self.resume_dir:
            self._load_resume_state()

    def _vae_tensors(self):
        m = self.cfg.model
        return load_vae_tensors(m.vae.source, m.vae.encoder, m.num_latents, m.latent_dim)

    # -- model ------------------------------------------------------------------------------------------------------
    def _build_model(self):
        cfg, info = self.cfg, self.info
        model = build_model(cfg.model, self.tokenizer, "meta")
        self.expected = {k: v.shape for k, v in model.state_dict().items()}
        set_trainable(model, cfg.model.trainable)
        layer_cls = type(model.backbone_llm.model.layers[0])          # Qwen3DecoderLayer
        self.mesh = make_mesh(info, cfg.fsdp.sharding)
        shard_model(model, layer_cls, self.mesh, cfg.fsdp.param_dtype, cfg.fsdp.reduce_dtype,
                    cfg.fsdp.activation_checkpointing)
        model.to_empty(device=info.device)
        reinit_rotary_buffers(model, info.device)
        source = self.resume_dir or cfg.model.init_from
        state = {}
        if info.is_main:
            state = initial_state(cfg.model, self.tokenizer, source)
            check_state(state, self.expected, source or f"pretrained {cfg.model.backbone} + fresh LaDiR modules")
        set_model_state_dict(model, state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True,
                                                                    strict=True))
        del state
        model.train()
        self.model = model
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        if info.is_main:
            n_train = sum(p.numel() for p in self.trainable)
            n_all = sum(p.numel() for p in model.parameters())
            print(f"[sft] init from {source or 'pretrained backbone'}; trainable {n_train / 1e6:.2f}M / "
                  f"{n_all / 1e6:.2f}M params; {len(self.dataset)} samples, {self.steps_per_epoch} steps/epoch, "
                  f"{self.total_steps} total steps", flush=True)

    def full_state_dict(self) -> Dict[str, torch.Tensor]:
        """Gathered fp32 state dict with checkpoint key names (rank 0; empty elsewhere)."""
        state = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if not self.info.is_main:
            return {}
        state = {clean_fqn(k): v for k, v in state.items()}
        check_state(state, self.expected, "gathered state dict")
        return state

    # -- checkpointing ----------------------------------------------------------------------------------------------
    def _trainer_state(self) -> dict:
        return {"step": self.step, "total_steps": self.total_steps, "steps_per_epoch": self.steps_per_epoch,
                "world_size": self.info.world_size, "scheduler": self.sched.state_dict(), "run_name": self.cfg.run_name}

    def save(self, final: bool = False) -> str:
        info = self.info
        name = "final" if final else f"checkpoint-{self.step}"
        path, tmp = os.path.join(self.out, name), os.path.join(self.out, name + ".tmp")
        if info.is_main:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
        dist.barrier()
        state = self.full_state_dict()
        if info.is_main:
            vae = self._vae_tensors() if self.cfg.model.vae is not None else None
            write_checkpoint(tmp, self.model, state, {"sft_run_name": self.cfg.run_name, "sft_step": self.step}, vae)
            with open(os.path.join(tmp, STATE_FILE), "w") as f:
                json.dump(self._trainer_state(), f, indent=2)
        del state
        if not final:
            dcp.save({"optim": get_optimizer_state_dict(self.model, self.optim)},
                     checkpoint_id=os.path.join(tmp, "optim"))
            torch.save(rng_state(), os.path.join(tmp, f"rng_rank{info.rank}.pt"))
        dist.barrier()
        if info.is_main:
            shutil.rmtree(path, ignore_errors=True)
            os.replace(tmp, path)
            if not final:
                self._rotate()
        dist.barrier()
        return path

    def _rotate(self):
        limit = self.cfg.train.save_total_limit
        if limit is None:
            return
        ckpts = sorted((int(m.group(1)), d) for d in os.listdir(self.out) if (m := CKPT_RE.match(d)))
        for _, d in ckpts[:-limit] if limit > 0 else ckpts:
            shutil.rmtree(os.path.join(self.out, d), ignore_errors=True)

    def _load_resume_state(self):
        with open(os.path.join(self.resume_dir, STATE_FILE)) as f:
            state = json.load(f)
        if state["world_size"] != self.info.world_size or state["steps_per_epoch"] != self.steps_per_epoch:
            raise RuntimeError(f"cannot resume {self.resume_dir}: saved with world_size={state['world_size']}, "
                               f"steps_per_epoch={state['steps_per_epoch']}; now {self.info.world_size}, "
                               f"{self.steps_per_epoch} (the data order would change)")
        osd = {"optim": get_optimizer_state_dict(self.model, self.optim)}
        dcp.load(osd, checkpoint_id=os.path.join(self.resume_dir, "optim"))
        set_optimizer_state_dict(self.model, self.optim, osd["optim"])
        self.sched.load_state_dict(state["scheduler"])
        self.step = state["step"]
        if self.info.is_main:
            print(f"[sft] resumed from {self.resume_dir} at step {self.step}", flush=True)

    # -- training ---------------------------------------------------------------------------------------------------
    def _loader(self, epoch: int, skip_batches: int):
        t, info = self.cfg.train, self.info
        sampler = ResumableSampler(len(self.dataset), info.rank, info.world_size, self.cfg.seed)
        sampler.set_epoch(epoch, skip_batches * t.per_device_batch_size)
        workers = self.cfg.data.num_workers
        # own generator: creating a DataLoader iterator must not consume the global RNG (exact resume)
        return DataLoader(self.dataset, batch_size=t.per_device_batch_size, sampler=sampler,
                          generator=torch.Generator().manual_seed(self.cfg.seed + epoch),
                          collate_fn=SFTCollator(self.tokenizer.pad_token_id), num_workers=workers, drop_last=True,
                          pin_memory=info.device.type == "cuda", persistent_workers=False,
                          prefetch_factor=4 if workers > 0 else None)

    def _to_device(self, batch):
        compute = DTYPES[self.cfg.fsdp.param_dtype]
        out = {k: v.to(self.info.device, non_blocking=True) for k, v in batch.items()}
        if "gt_latents" in out:
            out["gt_latents"] = out["gt_latents"].to(compute)
        return out

    def train(self) -> str:
        cfg, info, t = self.cfg, self.info, self.cfg.train
        final_dir = os.path.join(self.out, "final")
        if t.resume and os.path.exists(os.path.join(final_dir, STATE_FILE)):
            if info.is_main:
                print(f"[sft] {final_dir} exists; nothing to do", flush=True)
            return final_dir
        torch.manual_seed(cfg.seed * 1000 + info.rank + 1)   # per-rank noise / dropout streams
        if self.resume_dir:   # restore the exact streams saved with the checkpoint
            set_rng_state(torch.load(os.path.join(self.resume_dir, f"rng_rank{info.rank}.pt"), weights_only=False))
        logger = Logger(cfg, info)
        acc = torch.zeros(6, dtype=torch.float64, device=info.device)   # loss, flow, ce, micro, samples, tokens
        t0 = time.time()
        while self.step < self.total_steps:
            epoch, within = divmod(self.step, self.steps_per_epoch)
            batches = iter(self._loader(epoch, within * t.grad_accum_steps))
            for _ in range(within, self.steps_per_epoch):
                if self.step >= self.total_steps:
                    break
                for _ in range(t.grad_accum_steps):
                    batch = self._to_device(next(batches))
                    out = self.model(**batch)
                    (out["loss"] / t.grad_accum_steps).backward()
                    tokens = sum(batch[f"{k}_attention_mask"].sum() for k in ("prompt", "reasoning", "answer"))
                    acc += torch.stack([out["loss"].detach().double(), out["flow_loss"].double(),
                                        out["ce_loss"].double(), acc.new_tensor(1.0),
                                        acc.new_tensor(float(batch["prompt_input_ids"].shape[0])), tokens.double()])
                for p in self.trainable:   # params without a grad this step (e.g. null_embed) get a zero grad,
                    if p.grad is None:     # so every param always has optimizer state
                        p.grad = torch.zeros_like(p)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, t.max_grad_norm)
                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor()
                lr = self.sched.get_last_lr()[0]
                self.optim.step()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)
                self.step += 1
                if self.step % t.logging_steps == 0 or self.step == self.total_steps:
                    dist.all_reduce(acc)
                    if not torch.isfinite(acc[:3]).all():   # identical on every rank after the all-reduce
                        raise FloatingPointError(f"non-finite loss at step {self.step}")
                    dt = time.time() - t0
                    n = acc[3].item()
                    logger.log(self.step, {"loss": acc[0].item() / n, "flow_loss": acc[1].item() / n,
                                           "ce_loss": acc[2].item() / n, "grad_norm": float(grad_norm), "lr": lr,
                                           "epoch": self.step / self.steps_per_epoch,
                                           "samples_per_sec": acc[4].item() / dt, "tokens_per_sec": acc[5].item() / dt})
                    acc.zero_()
                    t0 = time.time()
                if self.step % t.save_steps == 0 and self.step < self.total_steps:
                    self.save()
        path = self.save(final=True)
        logger.close()
        return path


def run(cfg: SFTConfig, device=None) -> str:
    trainer = SFTTrainer(cfg, device)
    return trainer.train()
