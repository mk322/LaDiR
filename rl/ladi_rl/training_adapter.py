"""Training adapter: LaDiR in verl-omni's DiffusionModelBase registry, sharing the rollout's CFG and scheduler."""

from __future__ import annotations

import logging

import torch
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase

from .common import ALGORITHMS, ARCHITECTURE, FlowRLConfig, build_scheduler, labels_from_timesteps, set_scheduler_timesteps
from .ladi_model import LaDiForTraining, is_trainable

logger = logging.getLogger(__name__)


def flow_config_from_model_config(model_config) -> FlowRLConfig:
    """Latent-rollout settings from verl-omni's DiffusionModelConfig."""
    p, a = model_config.pipeline, model_config.algo
    steps = int(p.num_inference_steps)
    rng = list(a.sde_window_range) if a.sde_window_range is not None else [0, steps]
    return FlowRLConfig(
        num_steps=steps,
        cfg_scale=float(p.guidance_scale) if p.guidance_scale is not None else 1.0,
        noise_level=float(a.noise_level),
        sde_type=str(a.sde_type),
        window_size=int(a.sde_window_size or 2),
        window_lo=int(rng[0]),
        window_hi=int(rng[1]),
        window_schedule="sliding" if getattr(a, "sample_strategy", "random") == "progressive" else "random",
        window_slide_every=int(getattr(a, "iters_per_group", 1) or 1),
    )


def flow_config_from_sampling(sp: dict) -> FlowRLConfig:
    """Latent-rollout settings from the rollout sampling-params dict."""
    steps = int(sp.get("num_inference_steps") or 10)
    rng = sp.get("sde_window_range") or [0, steps]
    if isinstance(rng, str):
        import ast
        rng = ast.literal_eval(rng)
    gs = sp.get("guidance_scale")
    return FlowRLConfig(
        num_steps=steps,
        cfg_scale=float(gs) if gs is not None else 1.0,
        noise_level=float(sp.get("noise_level", 0.7)),
        sde_type=str(sp.get("sde_type", "sde")),
        window_size=int(sp.get("sde_window_size") or 2),
        window_lo=int(rng[0]),
        window_hi=int(rng[1]),
        window_schedule="sliding" if sp.get("sample_strategy", "random") == "progressive" else "random",
        window_slide_every=int(sp.get("iters_per_group") or 1),
    )


def _unwrap(module):
    return getattr(module, "_fsdp_wrapped_module", module)


class LaDiDiffusionBase(DiffusionModelBase):
    """Registered once per algorithm."""

    @classmethod
    def build_module(cls, model_config, torch_dtype: torch.dtype):
        logger.info("Loading LaDiForTraining from %s (dtype=%s)", model_config.local_path, torch_dtype)
        return LaDiForTraining.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def configure_trainable_params(cls, module, model_config):
        """Freeze parameters outside the ladi.trainable preset; keep trainable parameters in fp32."""
        preset = getattr(cls, "_ladi_trainable", "all") or "all"
        for name, param in module.named_parameters():
            param.requires_grad = is_trainable(name, preset)
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()

    @classmethod
    def build_scheduler(cls, model_config):
        return build_scheduler(int(model_config.pipeline.num_inference_steps), get_device_name())

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device: str):
        set_scheduler_timesteps(scheduler, int(model_config.pipeline.num_inference_steps), device)

    @classmethod
    def prepare_model_inputs(cls, module, model_config, latents, timesteps, prompt_embeds, prompt_embeds_mask,
                             negative_prompt_embeds, negative_prompt_embeds_mask, micro_batch, step: int):
        # CFG is applied inside the module forward (no separate negative branch)
        t = timesteps[:, step]
        model_inputs = {
            "mode": "flow",
            "prompt_ids": micro_batch["prompt_ids"],
            "prompt_mask": micro_batch["prompt_mask"],
            "latents": latents[:, step],
            "labels": labels_from_timesteps(t, _unwrap(module).config.num_train_timesteps),
            "cfg_scale": flow_config_from_model_config(model_config).cfg_scale,
        }
        return model_inputs, None

    @classmethod
    def forward_and_sample_previous_step(cls, module, scheduler, model_config, model_inputs, negative_model_inputs,
                                         scheduler_inputs, step: int):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]
        v = module(**model_inputs)[0]
        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=v.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
            include_logprob_normalizer=False,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

    @classmethod
    def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
        return module(**model_inputs)[0]


_REGISTERED = {}
for _alg in ALGORITHMS:
    _cls = type(f"LaDiDiffusion_{_alg}", (LaDiDiffusionBase,), {})
    DiffusionModelBase.register(ARCHITECTURE, algorithm=_alg)(_cls)
    _REGISTERED[_alg] = _cls

LaDiDiffusion = _REGISTERED["flow_grpo"]
