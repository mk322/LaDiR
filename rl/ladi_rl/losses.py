"""Answer-token loss. L = flow_coef * L_flow (verl-omni diffusion loss) + text_coef * L_text (token-level PPO)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch


@dataclass
class HybridLossConfig:
    flow_coef: float = 1.0          # lambda (fixed mode; first update of grad_ratio mode)
    flow_coef_mode: str = "fixed"   # fixed | grad_ratio (lambda balances the backbone gradient norms, see engine)
    flow_grad_ratio: float = 1.0    # grad_ratio: ||lambda g_flow|| / ||g_text|| on the shared backbone
    flow_coef_min: float = 1.0
    flow_coef_max: float = 1.0e4
    text_coef: float = 1.0
    text_clip_low: float = 0.2
    text_clip_high: float = 0.28
    text_adv_clip: float = 5.0
    text_clip_ratio_c: float = 3.0  # dual-clip lower bound for negative advantages (verl default 3.0); <= 1 disables
    temperature: float = 1.0        # must equal the rollout sampling temperature
    text_tis_cap: float = 0.0       # >0: truncated importance sampling min(pi_old / pi_rollout, cap) per answer token
    fuse_window_steps: bool = True  # one batched forward/backward for all window steps of a micro-batch
    train_text: bool = True
    train_flow: bool = True

    @classmethod
    def from_dict(cls, d) -> "HybridLossConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self):
        return asdict(self)


def tis_weights(old_logp, rollout_logp, cap: float):
    """Truncated importance weights pi_old / pi_rollout per token."""
    return (old_logp - rollout_logp).clamp(-20, 20).exp().clamp(max=cap).detach()


def text_ppo_terms(new_logp, old_logp, advantages, mask, clip_low=0.2, clip_high=0.28, adv_clip=5.0,
                   is_weights=None, clip_ratio_c: float = 3.0):
    """Clipped PPO (clip-higher, dual-clip) summed over valid tokens. new/old_logp, mask: [B, L]; advantages: [B]."""
    mask = mask.float()
    log_ratio = (new_logp - old_logp).clamp(-20, 20)
    ratio = log_ratio.exp()
    adv = advantages.float().clamp(-adv_clip, adv_clip).unsqueeze(1)
    per_tok = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - clip_low, 1 + clip_high))
    dual = None
    if clip_ratio_c is not None and clip_ratio_c > 1.0:
        capped = torch.minimum(per_tok, -adv * clip_ratio_c)
        dual = (adv < 0) & (per_tok > -adv * clip_ratio_c)
        per_tok = torch.where(adv < 0, capped, per_tok)
    if is_weights is not None:
        per_tok = per_tok * is_weights
    loss_sum = (per_tok * mask).sum()
    with torch.no_grad():
        stats = {
            "clip_sum": (((ratio < 1 - clip_low) | (ratio > 1 + clip_high)).float() * mask).sum(),
            "clip_high_sum": ((ratio > 1 + clip_high).float() * mask).sum(),
            "kl_sum": (((ratio - 1) - log_ratio) * mask).sum(),
            "ratio_sum": (ratio * mask).sum(),
            "tokens": mask.sum(),
            "dual_clip_sum": (dual.float() * mask).sum() if dual is not None else torch.zeros(()),
            "is_weight_sum": (is_weights * mask).sum() if is_weights is not None else mask.sum(),
        }
    return loss_sum, stats
