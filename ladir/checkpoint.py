"""Load and save LaDiR checkpoints (HF repo id or local dir) with strict key checking.

Released checkpoint (model.safetensors): mk322/LaDiR-Code-8B, LaDiR model + vae.* (frozen Qwen3-4B VAE encoder).
"""

import json
import os
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from .modeling.model import LaDiRConfig, LaDiRModel
from .modeling.vae_encoder import VAEEncoder
from .tokenizer import load_tokenizer

HEADS_FILE = "ladi_heads.safetensors"      # split layout: <dir>/backbone-hf/ (plain HF Qwen3) + <dir>/ladi_heads.safetensors


def is_split(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, HEADS_FILE)) and os.path.isdir(os.path.join(ckpt_dir, "backbone-hf"))


def _load_split_state(ckpt_dir: str) -> dict:
    """Full state dict from a split checkpoint (backbone-hf/ + heads file)."""
    import glob
    state = {f"backbone_llm.{k}": v for f in sorted(glob.glob(os.path.join(ckpt_dir, "backbone-hf", "*.safetensors")))
             for k, v in load_file(f, device="cpu").items()}
    state.update(load_file(os.path.join(ckpt_dir, HEADS_FILE), device="cpu"))
    return state


def resolve(path_or_repo: str) -> str:
    if os.path.isdir(path_or_repo):
        return path_or_repo
    return snapshot_download(path_or_repo, allow_patterns=["model.safetensors", "*.json"])


def checkpoint_keys(ckpt_dir: str):
    if is_split(ckpt_dir):
        with safe_open(os.path.join(ckpt_dir, HEADS_FILE), "pt") as f:
            return set(f.keys())
    with safe_open(os.path.join(ckpt_dir, "model.safetensors"), "pt") as f:
        return set(f.keys())


def detect_latent_source(keys) -> str:
    return "vae" if any(k.startswith("vae.") for k in keys) else "none"


def load_ladir(path_or_repo: str, device="cuda", dtype=torch.bfloat16, load_vae: bool = False,
               config: Optional[LaDiRConfig] = None) -> LaDiRModel:
    """Build a LaDiRModel and load weights strictly.

    load_vae: also build and load the VAE encoder (needed only to encode new training
              targets; generation never uses it). Otherwise `vae.*` keys are ignored.
    """
    ckpt_dir = resolve(path_or_repo)
    keys = checkpoint_keys(ckpt_dir)
    source = detect_latent_source(keys)
    cfg = config or LaDiRConfig()
    cfg.latent_source = source if (source != "vae" or load_vae) else "none"

    tokenizer = load_tokenizer(cfg.backbone)
    # build directly on the target device (random init is cheap there), then copy weights in
    with torch.device(device):
        backbone = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg.backbone), torch_dtype=dtype,
                                                    attn_implementation="sdpa")
        backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        vae = None
        if cfg.latent_source == "vae":
            vae = VAEEncoder(cfg.vae_encoder_name, cfg.num_latents, cfg.latent_dim, cfg.vae_mask_mode)
        model = LaDiRModel(cfg, backbone, tokenizer, vae=vae).to(dtype)

    state = _load_split_state(ckpt_dir) if is_split(ckpt_dir) else \
        load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu")
    if cfg.latent_source != "vae":
        state = {k: v for k, v in state.items() if not k.startswith("vae.")}
    expected = set(model.state_dict().keys())
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/key mismatch: missing={sorted(missing)[:10]} unexpected={sorted(unexpected)[:10]}")
    model.load_state_dict(state, strict=True)   # copies and casts into the bf16 parameters
    del state
    return model.eval()


def save_ladir(model: LaDiRModel, out_dir: str, extra_config: dict = None, state_dict: dict = None):
    """Save in the single-file format of the released checkpoints (plus ladi_config.json).
    state_dict: pass a gathered full state dict when the live model is sharded (FSDP)."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in (state_dict or model.state_dict()).items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "ladi_config.json"), "w") as f:
        json.dump({**model.config.__dict__, **(extra_config or {})}, f, indent=2)
