"""Schedule, velocity and SDE-window helpers shared by the rollout and the training adapter.

x_sigma = (1 - sigma) data + sigma noise, v = noise - data; scheduler timesteps are sigma * 1000 and LaDiR labels are
round((1 - sigma) T) clamped to T - 1.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields
from typing import Optional

import torch

ARCHITECTURE = "LaDiForLatentReasoning"
ALGORITHMS = ("flow_grpo", "grpo_guard", "dance_grpo")
SCHED_TRAIN_TIMESTEPS = 1000


@dataclass
class FlowRLConfig:
    """Latent-rollout settings (identical in rollout and training)."""
    num_steps: int = 10
    cfg_scale: float = 1.0
    noise_level: float = 0.7
    sde_type: str = "sde"                      # sde | cps | dance_sde (verl-omni scheduler)
    window_size: int = 2
    window_lo: int = 0                         # window lies inside steps [window_lo, window_hi)
    window_hi: int = 5
    window_schedule: str = "random"            # random | fixed | sliding
    window_start: int = 0                      # for fixed
    window_slide_every: int = 1                # for sliding
    include_logprob_normalizer: bool = False

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "FlowRLConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self):
        assert self.sde_type in ("sde", "cps", "dance_sde"), self.sde_type
        assert self.window_schedule in ("random", "fixed", "sliding"), self.window_schedule
        assert 1 <= self.window_size, self.window_size
        assert 0 <= self.window_lo and self.window_lo + self.window_size <= self.window_hi <= self.num_steps, \
            (self.window_lo, self.window_size, self.window_hi, self.num_steps)
        if self.sde_type == "sde":
            assert self.num_steps >= 2
        return self


def linear_sigmas(num_steps: int) -> list:
    """[1, ..., 1/num_steps] (terminal 0 is appended by the scheduler)."""
    return torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)[:-1].tolist()


def build_scheduler(num_steps: int, device=None):
    """verl-omni FlowMatchSDEDiscreteScheduler with linear sigmas."""
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    sched = FlowMatchSDEDiscreteScheduler(num_train_timesteps=SCHED_TRAIN_TIMESTEPS, shift=1.0)
    set_scheduler_timesteps(sched, num_steps, device)
    return sched


def set_scheduler_timesteps(sched, num_steps: int, device=None):
    sched.set_shift(1.0)
    sched.set_timesteps(sigmas=linear_sigmas(num_steps), device=device)
    sched.set_begin_index(0)
    return sched


def labels_from_timesteps(t: torch.Tensor, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Scheduler timestep (sigma * 1000) -> LaDi label round((1 - sigma) * T), clamped to T - 1."""
    sigma = t.double() / SCHED_TRAIN_TIMESTEPS
    return torch.round((1.0 - sigma) * num_train_timesteps).clamp(0, num_train_timesteps - 1).long()


def flow_velocity(model, prompt_ids, prompt_mask, latents, timesteps, prompt_embeds=None):
    """LaDiRModel.flow_velocity through the base Qwen3 model (no vocab logits); identical output."""
    from ladir.modeling.flow import flow_sequence_mask, positions_from_mask
    tok = model.tokenizer
    B, T = prompt_ids.shape
    N = latents.shape[1]
    if prompt_embeds is None:
        prompt_embeds = model.embed(prompt_ids)
    dtype = prompt_embeds.dtype
    inputs_embeds = torch.cat([
        prompt_embeds,
        model._special(model.bot_embed, B, dtype),
        model.time_embed(timesteps, dtype).unsqueeze(1),
        model.flow_projector(latents).to(dtype),
    ], dim=1)
    tail = torch.tensor([tok.bot_token_id, tok.time_token_id] + [tok.latent_token_id] * N, device=prompt_ids.device)
    full_ids = torch.cat([prompt_ids, tail.unsqueeze(0).expand(B, -1)], dim=1)
    valid = torch.cat([prompt_mask, prompt_mask.new_ones(B, N + 2)], dim=1)
    mask = flow_sequence_mask(full_ids, valid, tok.latent_token_id, model.config.flow_mask, dtype)
    hidden = model.backbone_llm.model(inputs_embeds=inputs_embeds, attention_mask=mask,
                                      position_ids=positions_from_mask(valid), use_cache=False,
                                      return_dict=True).last_hidden_state[:, -N:]
    return model.output_projection(hidden, model.t_embedder(timesteps, hidden.dtype))


def cfg_velocity(model, prompt_ids, prompt_mask, x, labels, cfg_scale: float):
    """Velocity with classifier-free guidance (unconditional branch = learned null prompt), one batched forward."""
    xin = x.to(model.dtype)
    if cfg_scale is None or cfg_scale <= 1.0:
        return flow_velocity(model, prompt_ids, prompt_mask, xin, labels)
    embeds = model.embed(prompt_ids)
    embeds = torch.cat([embeds, model.null_prompt_embeds(embeds)])
    v = flow_velocity(model, prompt_ids.repeat(2, 1), prompt_mask.repeat(2, 1), xin.repeat(2, 1, 1), labels.repeat(2),
                      prompt_embeds=embeds)
    cond, uncond = v.chunk(2)
    return uncond + cfg_scale * (cond - uncond)


def pick_window_start(cfg: FlowRLConfig, global_step: int = 0, rng: Optional[random.Random] = None) -> int:
    """First SDE step of the window; the window is [start, start + window_size)."""
    lo, W = cfg.window_lo, cfg.window_size
    last = cfg.window_hi - W                    # inclusive
    if cfg.window_schedule == "fixed":
        start = cfg.window_start
    elif cfg.window_schedule == "random":
        start = (rng or random).randint(lo, last)
    else:  # sliding
        n_pos = last - lo + 1
        start = lo + (max(int(global_step) - 1, 0) // max(cfg.window_slide_every, 1)) % n_pos
    if not lo <= start <= last:
        raise ValueError(f"window start {start} outside [{lo}, {last}]")
    return int(start)
