"""Group-normalised advantages and zero-variance group filtering."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def group_index(uids: Sequence) -> torch.Tensor:
    ids = {}
    return torch.tensor([ids.setdefault(str(u), len(ids)) for u in uids], dtype=torch.long)


def hybrid_advantages(rewards: torch.Tensor, groups: torch.Tensor, std_mode: str = "batch", eps: float = 1e-4,
                      clip: float = 5.0):
    """rewards [B, M] (M answers per latent), groups [B]. std over the batch, the group, or none.

    Returns (text_adv [B, M], flow_adv [B] = mean over answers)."""
    r = rewards.float()
    B, M = r.shape
    adv = torch.zeros_like(r)
    gstd = torch.zeros(B)
    for g in groups.unique():
        idx = groups == g
        vals = r[idx]
        adv[idx] = vals - vals.mean()
        gstd[idx] = vals.std(unbiased=False) if vals.numel() > 1 else 0.0
    if std_mode == "batch":
        adv = adv / (r.std(unbiased=False) + eps)
    elif std_mode == "group":
        adv = adv / (gstd.unsqueeze(1) + eps)
    elif std_mode != "none":
        raise ValueError(std_mode)
    adv = adv.clamp(-clip, clip)
    return adv, adv.mean(dim=1)


def informative_rows(rewards: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Rows whose prompt group has non-zero reward variance."""
    keep = torch.zeros(rewards.shape[0], dtype=torch.bool)
    for g in groups.unique():
        idx = groups == g
        vals = rewards[idx]
        keep[idx] = bool(vals.max() > vals.min())
    return keep


def trim_to_multiple(idx: np.ndarray, groups: torch.Tensor, multiple: int) -> np.ndarray:
    """Drop whole groups from the end until len(idx) is a multiple of `multiple`."""
    if len(idx) % multiple == 0:
        return idx
    gs = groups[torch.as_tensor(idx)].tolist()
    order = list(dict.fromkeys(gs))
    while order and len(idx) % multiple:
        last = order.pop()
        cand = np.array([i for i, g in zip(idx, gs) if g != last])
        if len(cand) == 0:
            break
        idx, gs = cand, [g for g in gs if g != last]
    return idx[:len(idx) - len(idx) % multiple]
