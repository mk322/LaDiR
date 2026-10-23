"""Frozen-VAE encoder weights for stage-wise SFT, from either source format, validated strictly.

Sources
  * a LaDiR checkpoint (local dir or HF repo id): its `vae.*` tensors
  * a VAE trained with `vae.train`: its exported weights (weights/step_N.safetensors) or a .pt state dict; decoder /
    `ae_speical_token` entries are dropped, a tied encoder lm_head that the export omits is restored from the
    embeddings, and the rest must be exactly the encoder keys
Either way the result is validated strictly (names and shapes) against `ladir.modeling.vae_encoder.VAEEncoder`.
File reading is shared with the VAE package (`vae.model.read_stage2_vae_state` / `load_state_file`); encoding is
`VAEEncoder.encode`, which is batch-invariant (pad-skipping positions).
"""

from typing import Dict

import json
import os

import torch

from ladir.modeling.vae_encoder import VAEEncoder
from vae.model import load_state_file, read_stage2_vae_state

PREFIX = "vae."


def expected_vae_shapes(encoder_name: str, num_latents: int, latent_dim: int) -> Dict[str, torch.Size]:
    with torch.device("meta"):
        vae = VAEEncoder(encoder_name, num_latents, latent_dim)
    return {k: v.shape for k, v in vae.state_dict().items()}


def is_vae_training_file(source: str) -> bool:
    """True for a `vae.train` export (.safetensors without `vae.*` keys) or a .pt state dict."""
    if source.endswith(".pt"):
        return True
    if not (source.endswith(".safetensors") and os.path.isfile(source)):
        return False
    from safetensors import safe_open
    with safe_open(source, "pt") as f:
        return not any(k.startswith(PREFIX) for k in f.keys())


def vae_mask_mode(source: str, default: str = "causal") -> str:
    """Encoder mask a VAE was trained with (read from the `vae.train` export metadata); `default` otherwise."""
    if source.endswith(".safetensors") and os.path.isfile(source):
        from safetensors import safe_open
        with safe_open(source, "pt") as f:
            cfg = (f.metadata() or {}).get("vae_config")
        if cfg:
            return json.loads(cfg).get("encoder_mask_mode", default)
    return default


def _read_source(source: str) -> Dict[str, torch.Tensor]:
    """LaDiR checkpoint (dir / model.safetensors / HF repo) -> its vae.* tensors; VAE training file -> encoder part."""
    if not is_vae_training_file(source):
        return read_stage2_vae_state(source)
    state = load_state_file(source)
    state = {k[len(PREFIX):] if k.startswith(PREFIX) else k: v for k, v in state.items()}
    state = {k: v for k, v in state.items() if "decoder" not in k and "ae_speical_token" not in k}
    head, emb = "encoder_model.lm_head.weight", "encoder_model.model.embed_tokens.weight"
    if head not in state and emb in state:      # tied weights are exported once
        state[head] = state[emb]
    return state


def load_vae_tensors(source: str, encoder_name: str, num_latents: int, latent_dim: int) -> Dict[str, torch.Tensor]:
    """VAE-encoder tensors keyed like VAEEncoder.state_dict() (no `vae.` prefix), strictly validated."""
    state = _read_source(source)
    expected = expected_vae_shapes(encoder_name, num_latents, latent_dim)
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise RuntimeError(f"VAE source {source}: missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}")
    bad = {k: (tuple(state[k].shape), tuple(s)) for k, s in expected.items() if state[k].shape != s}
    if bad:
        raise RuntimeError(f"VAE source {source}: shape mismatch {dict(list(bad.items())[:8])}")
    return state


def build_vae(source: str, encoder_name: str, num_latents: int, latent_dim: int, mask_mode: str = "causal",
              device="cpu", dtype=torch.bfloat16) -> VAEEncoder:
    state = load_vae_tensors(source, encoder_name, num_latents, latent_dim)
    with torch.device(device):
        vae = VAEEncoder(encoder_name, num_latents, latent_dim, mask_mode)
    vae.load_state_dict(state, strict=True)
    return vae.to(dtype).eval()
