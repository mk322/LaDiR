"""Model construction, strict initialisation and checkpoint writing for SFT.

Every rank builds the LaDiRModel on the meta device; only rank 0 materialises the full initial state dict (from a
LaDiR checkpoint, or pretrained backbone + fresh LaDiR modules), which is checked strictly against the model's key set
and shapes and then broadcast into the sharded parameters.
"""

import os
from typing import Dict, Optional

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from ladir.checkpoint import resolve, save_ladir
from ladir.modeling.model import LaDiRConfig, LaDiRModel

from .config import ModelCfg

MODEL_FILE = "model.safetensors"


def ladir_config(m: ModelCfg, hidden_size: int) -> LaDiRConfig:
    extra = {"vae_encoder_name": m.vae.encoder} if m.vae is not None else {}
    return LaDiRConfig(**extra,
        backbone=m.backbone, hidden_size=hidden_size, num_latents=m.num_latents, latent_dim=m.latent_dim,
        num_train_timesteps=m.num_train_timesteps, latent_source=m.latent_source,
        flow_mask=m.flow_mask, text_drop_prob=m.text_drop_prob, diffusion_batch_mul=m.diffusion_batch_mul,
        diffusion_loss_weight=m.diffusion_loss_weight, ce_loss_weight=m.ce_loss_weight,
        diffusion_forcing=m.diffusion_forcing, diffusion_forcing_prob=m.diffusion_forcing_prob,
        diffusion_forcing_noise_scale=m.diffusion_forcing_noise_scale)


def backbone_config(m: ModelCfg, tokenizer):
    cfg = AutoConfig.from_pretrained(m.backbone)
    if cfg.vocab_size > len(tokenizer):
        raise ValueError(f"backbone vocab {cfg.vocab_size} > tokenizer size {len(tokenizer)}")
    cfg.vocab_size = len(tokenizer)
    return cfg


def build_model(m: ModelCfg, tokenizer, device="meta") -> LaDiRModel:
    """LaDiRModel with fp32 parameters, randomly initialised (or uninitialised on `meta`)."""
    bcfg = backbone_config(m, tokenizer)
    with torch.device(device):
        backbone = AutoModelForCausalLM.from_config(bcfg, attn_implementation="sdpa", dtype=torch.float32)
        model = LaDiRModel(ladir_config(m, bcfg.hidden_size), backbone, tokenizer)
    return model


def set_trainable(model: LaDiRModel, trainable: str):
    """"projectors": everything outside the LLM backbone (projectors, output head, special/time embeddings) -- stage 1.
    "all": every parameter.
    `d_embed` is never trained: it is unused by the model and only kept for checkpoint compatibility."""
    if trainable not in ("projectors", "all"):
        raise ValueError(trainable)
    model.requires_grad_(True)
    if trainable == "projectors":
        model.backbone_llm.requires_grad_(False)
    model.d_embed.requires_grad_(False)


def read_ladir_state(source: str) -> Dict[str, torch.Tensor]:
    """Full LaDiR state dict from a checkpoint dir / HF repo, without `vae.*` (the VAE is never trained here)."""
    state = load_file(os.path.join(resolve(source), MODEL_FILE), device="cpu")
    return {k: v.float() for k, v in state.items() if not k.startswith("vae.")}


def initial_state(m: ModelCfg, tokenizer, source: Optional[str]) -> Dict[str, torch.Tensor]:
    """Rank-0 only: the full fp32 initial state dict."""
    if source is not None:
        return read_ladir_state(source)
    backbone = AutoModelForCausalLM.from_pretrained(m.backbone, dtype=torch.float32, attn_implementation="sdpa")
    backbone.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model = LaDiRModel(ladir_config(m, backbone.config.hidden_size), backbone, tokenizer)
    return {k: v.detach() for k, v in model.state_dict().items()}


def check_state(state: Dict[str, torch.Tensor], expected: Dict[str, torch.Size], what: str):
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise RuntimeError(f"{what}: key mismatch, missing={sorted(missing)[:10]} unexpected={sorted(unexpected)[:10]}"
                           f" ({len(missing)} missing, {len(unexpected)} unexpected)")
    bad = [(k, tuple(state[k].shape), tuple(s)) for k, s in expected.items() if state[k].shape != s]
    if bad:
        raise RuntimeError(f"{what}: shape mismatch {bad[:10]}")


def reinit_rotary_buffers(model: torch.nn.Module, device):
    """Non-persistent buffers are not in state dicts, so after meta init + to_empty they hold garbage. In Qwen3 the
    only ones are the rotary inverse frequencies; rebuild those modules on the target device."""
    for name, mod in list(model.named_modules()):
        if not mod._non_persistent_buffers_set:
            continue
        if not name.endswith("rotary_emb"):
            raise RuntimeError(f"unexpected non-persistent buffers in {name}: {mod._non_persistent_buffers_set}")
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, type(mod)(config=mod.config, device=device))


def write_checkpoint(out_dir: str, model: LaDiRModel, state: Dict[str, torch.Tensor], extra: Optional[dict] = None,
                     vae_tensors: Optional[Dict[str, torch.Tensor]] = None):
    """Single-file checkpoint via `ladir.checkpoint.save_ladir` from a gathered state dict, plus `vae.*` tensors."""
    if vae_tensors:
        state = {**state, **{f"vae.{k}": v for k, v in vae_tensors.items()}}
    save_ladir(model, out_dir, extra_config=extra, state_dict=state)
