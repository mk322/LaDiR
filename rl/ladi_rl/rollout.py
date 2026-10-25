"""Rollout core (pure PyTorch): latent trajectories with an SDE window, then answer decoding from the latents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from ladir.modeling.flow import positions_from_mask

from .common import FlowRLConfig, cfg_velocity, labels_from_timesteps


@dataclass
class LatentGroupRollout:
    final_latents: torch.Tensor     # [P*G, N, D] fp32 (z fed to the text stage after .to(model dtype))
    all_latents: torch.Tensor       # [P*G, W+1, N, D] fp32
    all_log_probs: torch.Tensor     # [P*G, W] fp32
    all_timesteps: torch.Tensor     # [P*G, W] fp32 scheduler timesteps
    window_start: int


def _labels(t_scalar: torch.Tensor, batch: int, model) -> torch.Tensor:
    return labels_from_timesteps(t_scalar.reshape(1), model.config.num_train_timesteps).expand(batch).to(
        t_scalar.device)


@torch.no_grad()
def rollout_latent_groups(model, scheduler, prompt_ids, prompt_mask, group_size: int, cfg: FlowRLConfig,
                          window_start: int, generator: Optional[torch.Generator] = None,
                          noise: Optional[torch.Tensor] = None,
                          latent_init: str = "independent") -> LatentGroupRollout:
    """P*G trajectories (row p*G + g): ODE steps [0, w), SDE window [w, w+W) with log-probs, ODE steps to the end.

    noise: [P*G, N, D] (independent) or [P, N, D] (shared_prefix); noise_level 0 runs the window as ODE steps."""
    P = prompt_ids.shape[0]
    N, D = model.config.num_latents, model.config.latent_dim
    dev = prompt_ids.device
    sig, ts = scheduler.sigmas, scheduler.timesteps
    S, W, w = len(ts), cfg.window_size, int(window_start)
    assert 0 <= w and w + W <= S, (w, W, S)

    assert latent_init in ("independent", "shared_prefix"), latent_init
    n_init = P * group_size if latent_init == "independent" else P
    if noise is None:
        noise = torch.randn(n_init, N, D, generator=generator, dtype=torch.float32,
                            device=generator.device if generator is not None else dev).to(dev)
    assert noise.shape[0] == n_init, (noise.shape, n_init, latent_init)
    x = noise.float()

    def ode(x, ids, mask, k):
        v = cfg_velocity(model, ids, mask, x, _labels(ts[k], x.shape[0], model), cfg.cfg_scale)
        return x + (sig[k + 1] - sig[k]).to(x.device) * v.float()

    ids, mask = prompt_ids.repeat_interleave(group_size, 0), prompt_mask.repeat_interleave(group_size, 0)
    if latent_init == "shared_prefix":
        for k in range(w):
            x = ode(x, prompt_ids, prompt_mask, k)
        x = x.repeat_interleave(group_size, 0)
    else:
        for k in range(w):
            x = ode(x, ids, mask, k)
    B = x.shape[0]
    states, lps, tsteps = [x], [], []
    for k in range(w, w + W):
        t_vec = ts[k].reshape(1).expand(B)
        if cfg.noise_level <= 0:
            x, lp = ode(x, ids, mask, k), torch.zeros(B, device=x.device)
        else:
            v = cfg_velocity(model, ids, mask, x, _labels(ts[k], B, model), cfg.cfg_scale)
            x, lp, _, _ = scheduler.sample_previous_step(
                sample=x, model_output=v.float(), timestep=t_vec, generator=generator, noise_level=cfg.noise_level,
                sde_type=cfg.sde_type, return_logprobs=True,
                include_logprob_normalizer=cfg.include_logprob_normalizer)
        states.append(x)
        lps.append(lp)
        tsteps.append(t_vec.float())
    for k in range(w + W, S):
        x = ode(x, ids, mask, k)

    return LatentGroupRollout(final_latents=x, all_latents=torch.stack(states, 1), all_log_probs=torch.stack(lps, 1),
                              all_timesteps=torch.stack(tsteps, 1), window_start=w)


def window_log_probs(model_fn, scheduler, all_latents, all_timesteps, cfg: FlowRLConfig, step: int):
    """Recompute one window step. model_fn(x, labels) -> velocity."""
    x = all_latents[:, step].float()
    t = all_timesteps[:, step]
    v = model_fn(x, labels_from_timesteps(t))
    _, lp, mean, std, sqrt_dt = scheduler.sample_previous_step(
        sample=x, model_output=v.float(), timestep=t, noise_level=cfg.noise_level,
        prev_sample=all_latents[:, step + 1].float(), sde_type=cfg.sde_type, return_logprobs=True,
        return_sqrt_dt=True, include_logprob_normalizer=cfg.include_logprob_normalizer)
    return lp, mean, std, sqrt_dt


@dataclass
class DecodeConfig:
    max_new_tokens: int = 768
    temperature: float = 1.0
    top_p: float = 1.0
    stop_at_code_block: bool = True
    answers_per_latent: int = 1
    batch_size: int = 64


def text_prefix_embeds(model, prompt_ids, latents):
    """[prompt, <bot>, ce_projector(z), <eot>]."""
    return model.text_prefix(prompt_ids, latents.to(model.dtype))


def _backtick_token_ids(tok) -> torch.Tensor:
    cache = getattr(tok, "_ladi_backtick_ids", None)
    if cache is None:
        vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
        cache = torch.tensor([i for i, t in enumerate(vocab) if t is not None and "`" in t], dtype=torch.long)
        tok._ladi_backtick_ids = cache
    return cache


def first_block_closed(text: str, open_fence: int) -> bool:
    """True once the first fenced code block is closed."""
    return open_fence + text.count("```") >= 2


def _sample(logits, temperature, top_p, generator):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def decode_answers(model, prompt_ids, prompt_mask, latents, dc: DecodeConfig, open_fences: Optional[List[int]] = None,
                   generator: Optional[torch.Generator] = None):
    """Sample answers given the latents (HF KV cache); stop at EOS or when the first code block closes.

    Returns (token lists, per-token sampling log-probs)."""
    from transformers.cache_utils import DynamicCache

    tok = model.tokenizer
    B, N = latents.shape[0], latents.shape[1]
    prefix = text_prefix_embeds(model, prompt_ids, latents)
    mask = torch.cat([prompt_mask, prompt_mask.new_ones(B, N + 2)], dim=1)
    pos = positions_from_mask(mask)
    cache = DynamicCache()
    out = model.backbone_llm(inputs_embeds=prefix, attention_mask=mask, position_ids=pos, past_key_values=cache,
                             use_cache=True, return_dict=True, logits_to_keep=1)
    logits = out.logits[:, -1].float()
    next_pos = pos[:, -1:] + 1
    gen = [[] for _ in range(B)]
    lps = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
    open_f = list(open_fences) if open_fences is not None else [0] * B
    bt_ids = _backtick_token_ids(tok).to(prompt_ids.device) if dc.stop_at_code_block else None
    temp = dc.temperature if dc.temperature > 0 else 1.0
    for _ in range(dc.max_new_tokens):
        nxt = _sample(logits, dc.temperature, dc.top_p, generator)
        lp = torch.log_softmax(logits / temp, -1).gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
        has_bt = torch.isin(nxt, bt_ids) if bt_ids is not None else torch.zeros_like(done)
        nxt_l, lp_l, bt_l, done_l = nxt.tolist(), lp.tolist(), has_bt.tolist(), done.tolist()
        for i in range(B):
            if done_l[i]:
                continue
            gen[i].append(nxt_l[i])
            lps[i].append(lp_l[i])
            if nxt_l[i] == tok.eos_token_id:
                done[i] = True
            elif bt_l[i] and first_block_closed(tok.decode(gen[i]), open_f[i]):
                done[i] = True
        if bool(done.all()):
            break
        mask = torch.cat([mask, (~done).long().unsqueeze(1)], dim=1)
        out = model.backbone_llm(input_ids=nxt.unsqueeze(1), attention_mask=mask, position_ids=next_pos,
                                 past_key_values=cache, use_cache=True, return_dict=True)
        logits = out.logits[:, -1].float()
        next_pos = next_pos + 1
    return gen, lps


def pad_answers(answers: List[List[int]], pad_id: int, max_len: Optional[int] = None):
    """-> (ids [B, L] long, mask [B, L] long); L >= 1."""
    L = max([len(a) for a in answers] + [1]) if max_len is None else max_len
    ids = torch.full((len(answers), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(answers), L), dtype=torch.long)
    for i, a in enumerate(answers):
        a = a[:L]
        if a:
            ids[i, :len(a)] = torch.tensor(a, dtype=torch.long)
            mask[i, :len(a)] = 1
    return ids, mask
