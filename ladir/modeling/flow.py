"""Flow-matching conventions, schedules and attention masks.

Convention:
    timestep label t in [0, T) with T = 1000;  s = t / T
    x_t   = (1 - s) * noise + s * data          # t = 0 is pure noise, t -> T is clean data
    target = noise - data                        # = dx/dsigma with sigma = 1 - s
So the model predicts v = dx/dsigma, and an ODE step from sigma_i to sigma_{i+1} is x += (sigma_{i+1} - sigma_i) * v.
"""

from typing import List, Tuple

import torch


def add_noise(data: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, num_train_timesteps: int):
    s = timesteps.float() / num_train_timesteps
    while s.dim() < data.dim():
        s = s.unsqueeze(-1)
    return ((1 - s) * noise + s * data).to(data.dtype)


def flow_target(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - data


def make_schedule(num_steps: int, num_train_timesteps: int = 1000) -> List[Tuple[int, float]]:
    """Return [(timestep_label, dt)] where each Euler step is x <- x + dt * v."""
    T = num_train_timesteps
    sig = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64).tolist()
    return [(min(round((1.0 - sig[i]) * T), T - 1), sig[i + 1] - sig[i]) for i in range(num_steps)]


# ----------------------------------------------------------------------------------------------------------------
# Attention masks
# ----------------------------------------------------------------------------------------------------------------

def flow_sequence_mask(full_ids: torch.Tensor, valid: torch.Tensor, latent_id: int, mode: str, dtype=torch.bfloat16):
    """4D additive mask for the full-sequence flow forward over [prompt, <bot>, <timestep>, latents].

    Pattern: causal text, latents bidirectional among themselves. valid: [B, L] (0 = padding).
      mode="train_matched": the pattern is added as a 1.0/0.0 float bias to the attention logits (the attention
          used to train the released checkpoints).
      mode="hard": the pattern as a 0/-inf mask.
    Padding is always excluded with -inf.
    """
    B, L = full_ids.shape
    m = torch.tril(torch.ones(L, L, dtype=torch.bool, device=full_ids.device)).unsqueeze(0).expand(B, L, L).clone()
    lat = full_ids == latent_id
    m |= lat.unsqueeze(2) & lat.unsqueeze(1)
    pad_cols = ~valid.bool().unsqueeze(1).expand(B, L, L)
    neg = torch.finfo(dtype).min
    if mode == "train_matched":
        out = m.to(dtype)
    elif mode == "hard":
        out = torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, neg)
    else:
        raise ValueError(f"unknown mask mode {mode}")
    out = out.masked_fill(pad_cols, neg)
    # pad rows: let them see themselves so no row is fully masked (their outputs are discarded)
    eye = torch.eye(L, dtype=torch.bool, device=m.device).unsqueeze(0) & ~valid.bool().unsqueeze(2)
    out = out.masked_fill(eye, 0.0)
    return out.unsqueeze(1)


def positions_from_mask(valid: torch.Tensor) -> torch.Tensor:
    """position_ids that skip padding (pads get 0)."""
    return (valid.long().cumsum(-1) - 1).clamp(min=0) * valid.long()
