"""Strict OmegaConf schema for SFT runs.

Every field below is consumed by the trainer; unknown keys (in YAML or CLI overrides) raise an error.
Load with `load_config(path, overrides)`; overrides are OmegaConf dotlist strings such as `train.lr=1e-5`.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional

from omegaconf import MISSING, OmegaConf

LATENT_SOURCES = ("vae",)
TRAINABLE = ("projectors", "all")
SHARDING = ("full_shard", "hybrid_shard", "no_shard")
DTYPES = ("bf16", "fp32")
FLOW_MASKS = ("train_matched", "hard")


@dataclass
class VaeCfg:
    """Frozen VAE. It is never instantiated during training (latents are precomputed); its
    tensors are copied as `vae.*` into every saved checkpoint."""
    source: str = MISSING                          # LaDiR checkpoint with vae.* (dir / HF repo) or VAE training .pt
    encoder: str = "Qwen/Qwen3-4B-Base"            # architecture config of the VAE encoder LLM


@dataclass
class ModelCfg:
    backbone: str = "Qwen/Qwen3-8B-Base"          # HF id or local dir; also provides the tokenizer
    # LaDiR checkpoint (local dir or HF repo with model.safetensors) loaded strictly; `vae.*` keys are ignored.
    # None = pretrained `backbone` weights + freshly initialised LaDiR modules.
    init_from: Optional[str] = None
    trainable: str = "all"                         # "projectors" (stage 1: backbone frozen) | "all"
    latent_source: str = "vae"                     # precomputed VAE latents
    num_latents: int = 64
    latent_dim: int = 2560
    num_train_timesteps: int = 1000
    diffusion_batch_mul: int = 4
    text_drop_prob: float = 0.1
    diffusion_loss_weight: float = 5.0
    ce_loss_weight: float = 1.0
    diffusion_forcing: bool = True
    diffusion_forcing_prob: float = 0.5
    diffusion_forcing_noise_scale: float = 0.3
    flow_mask: str = "train_matched"               # "train_matched" | "hard" (see ladir.modeling.flow)
    vae: Optional[VaeCfg] = None                   # see VaeCfg; null = save without vae.*


@dataclass
class DataCfg:
    train_dir: str = MISSING                       # output of `python -m sft.data`
    latents_dir: str = MISSING                     # output of sft/precompute_latents.py
    num_samples: Optional[int] = None              # seeded random subset; error if larger than the dataset
    # must equal the values the data was pretokenized with (checked against the dataset metadata)
    max_prompt_length: int = 512
    max_reasoning_length: int = 1024
    max_answer_length: int = 1024
    num_workers: int = 4


@dataclass
class TrainCfg:
    num_epochs: int = 1
    max_steps: Optional[int] = None                # optional cap on optimizer steps
    per_device_batch_size: int = MISSING
    grad_accum_steps: int = 1
    lr: float = MISSING
    warmup_steps: int = 0                          # linear warmup, then constant
    weight_decay: float = 0.01                     # not applied to biases / norm weights
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    logging_steps: int = 30
    save_steps: int = MISSING
    save_total_limit: Optional[int] = 3
    resume: bool = True                            # resume from the newest output_dir/checkpoint-<step>


@dataclass
class FsdpCfg:
    sharding: str = "hybrid_shard"                 # hybrid: shard within a node, replicate across nodes
    param_dtype: str = "bf16"                      # compute dtype; master weights + optimizer state stay fp32
    reduce_dtype: str = "fp32"
    activation_checkpointing: bool = True


@dataclass
class SFTConfig:
    run_name: str = MISSING
    output_dir: str = MISSING
    seed: int = 42
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    fsdp: FsdpCfg = field(default_factory=FsdpCfg)


def _check(cond, msg):
    if not cond:
        raise ValueError(f"invalid SFT config: {msg}")


def validate(cfg: SFTConfig) -> SFTConfig:
    m, d, t = cfg.model, cfg.data, cfg.train
    _check(m.latent_source in LATENT_SOURCES, f"model.latent_source must be one of {LATENT_SOURCES}")
    _check(m.trainable in TRAINABLE, f"model.trainable must be one of {TRAINABLE}")
    _check(m.flow_mask in FLOW_MASKS, f"model.flow_mask must be one of {FLOW_MASKS}")
    _check(cfg.fsdp.sharding in SHARDING, f"fsdp.sharding must be one of {SHARDING}")
    _check(cfg.fsdp.param_dtype in DTYPES and cfg.fsdp.reduce_dtype in DTYPES, f"fsdp dtypes must be in {DTYPES}")
    _check(t.per_device_batch_size > 0 and t.grad_accum_steps > 0, "batch size / grad accumulation must be > 0")
    _check(t.save_steps > 0 and t.logging_steps > 0, "save_steps / logging_steps must be > 0")
    _check(t.warmup_steps >= 0, "warmup_steps must be >= 0")
    _check(d.num_samples is None or d.num_samples > 0, "data.num_samples must be positive or null")
    return cfg


def load_config(path: str, overrides: Optional[List[str]] = None) -> SFTConfig:
    schema = OmegaConf.structured(SFTConfig)
    merged = OmegaConf.merge(schema, OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides or [])))
    cfg = OmegaConf.to_object(merged)     # resolves interpolations; raises on MISSING values
    return validate(cfg)


def to_yaml(cfg: SFTConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))


def to_dict(cfg: SFTConfig) -> dict:
    return asdict(cfg)
