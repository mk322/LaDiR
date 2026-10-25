"""LaDiRModel as a verl-omni NonDiffusersModelBase.

All computation goes through forward(mode=...) so FSDP2 gathers the root-level parameters.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from ladir.checkpoint import resolve
from ladir.modeling.model import LaDiRConfig, LaDiRModel
from ladir.tokenizer import load_tokenizer

from verl_omni.pipelines.non_diffusers_model_base import NonDiffusersModelBase

from .common import cfg_velocity

UNUSED_IN_RL = ("d_embed.", "vae.")
FLOW_HEAD = ("flow_projector.", "output_projection.", "time_embed.", "t_embedder.")
TEXT_HEAD = ("ce_projector.",)

TRAINABLE_PRESETS = {
    "all": None,
    "flow_head": FLOW_HEAD,
    "heads": FLOW_HEAD + TEXT_HEAD + ("bot_embed", "eot_embed", "null_embed"),
}


def is_trainable(name: str, preset: str = "all") -> bool:
    if name.startswith(UNUSED_IN_RL):
        return False
    prefixes = TRAINABLE_PRESETS[preset]
    return prefixes is None or name.startswith(prefixes)


def is_backbone_param(name: str) -> bool:
    return name.startswith("backbone_llm.")


class SavableLaDiConfig(LaDiRConfig):
    """LaDiRConfig with the save_pretrained hook verl calls on module.config."""

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "ladi_config.json"), "w") as f:
            json.dump(dict(self.__dict__), f, indent=2)

    @classmethod
    def from_config(cls, cfg: LaDiRConfig) -> "SavableLaDiConfig":
        return cls(**dict(cfg.__dict__))


class LaDiForTraining(NonDiffusersModelBase, LaDiRModel):
    _no_split_modules = ["Qwen3DecoderLayer"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: LaDiRConfig, backbone_llm, tokenizer, vae=None):
        LaDiRModel.__init__(self, SavableLaDiConfig.from_config(config), backbone_llm, tokenizer, vae=vae)

    def enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
        self.gradient_checkpointing = True
        self.backbone_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        from ladir.checkpoint import save_ladir
        save_ladir(self, save_directory)

    @classmethod
    def build(cls, ladi_config: LaDiRConfig, device="cpu", dtype=torch.float32, backbone_config=None, tokenizer=None):
        tokenizer = tokenizer or load_tokenizer(ladi_config.backbone)
        with torch.device(device):
            bb_cfg = backbone_config or AutoConfig.from_pretrained(ladi_config.backbone)
            backbone = AutoModelForCausalLM.from_config(bb_cfg, dtype=dtype, attn_implementation="sdpa")
            if backbone.get_input_embeddings().weight.shape[0] != len(tokenizer):
                backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            model = cls(ladi_config, backbone, tokenizer).to(dtype)
        return model

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype: torch.dtype = torch.float32, device="cpu",
                        config: Optional[LaDiRConfig] = None, **kwargs) -> "LaDiForTraining":
        """Strict load of a LaDiR checkpoint or RL model dir, without the VAE."""
        from safetensors.torch import load_file

        from .model_dir import read_model_dir

        ckpt_dir = resolve(model_path)
        cfg, backbone_config, tok_path = read_model_dir(ckpt_dir, config)
        cfg.latent_source = "none"
        model = cls.build(cfg, device=device, dtype=torch_dtype, backbone_config=backbone_config,
                          tokenizer=load_tokenizer(tok_path))
        from ladir.checkpoint import is_split, _load_split_state
        state = _load_split_state(ckpt_dir) if is_split(ckpt_dir) else \
            load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("vae.")}
        expected = set(model.state_dict().keys())
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            raise RuntimeError(f"checkpoint/key mismatch: missing={sorted(missing)[:10]} "
                               f"unexpected={sorted(unexpected)[:10]}")
        model.load_state_dict(state, strict=True)
        del state
        return model

    def forward(self, mode: str = "flow", **kw):
        if mode == "flow":
            return self._flow(**kw)
        if mode == "text":
            return self._text(**kw)
        if mode == "sft":
            return LaDiRModel.forward(self, **kw)
        raise ValueError(mode)

    def _flow(self, prompt_ids, prompt_mask, latents, labels, cfg_scale: float = 1.0):
        """Velocity with CFG, as a 1-tuple."""
        return (cfg_velocity(self, prompt_ids, prompt_mask, latents, labels, cfg_scale),)

    def _text(self, prompt_ids, prompt_mask, latents, answer_ids, answer_mask, temperature: float = 1.0):
        """Per-token answer log-probs [B, L] given the prompt and fixed latents."""
        from rl.ladi_rl.text import answer_log_probs
        return answer_log_probs(self, prompt_ids, prompt_mask, latents.to(self.dtype), answer_ids, answer_mask,
                                temperature=temperature)
