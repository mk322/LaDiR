"""LaDi rollout stage: G latent trajectories per prompt (ODE prefix, SDE window, ODE to the end) and M answers each.

Every output tensor has a leading request dim of 1.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .common import FlowRLConfig, pick_window_start
from .rollout import DecodeConfig, decode_answers, pad_answers, rollout_latent_groups, text_prefix_embeds


@dataclass
class StageRequest:
    prompt_ids: List[int]
    group_size: int
    flow: FlowRLConfig
    decode: DecodeConfig
    window_start: Optional[int] = None
    seed: Optional[int] = None
    open_fence: int = 0
    global_step: int = 0
    return_prefix_embeds: bool = False
    decode_answers: bool = True
    latent_init: str = "independent"        # independent | shared_prefix
    extra: dict = field(default_factory=dict)


def trajectory_seed(seed, g: int):
    """Seed of trajectory g of a request (None -> unseeded)."""
    return None if seed is None else (int(seed) * 1_000_003 + 7 * g + 1) % (2 ** 63 - 1)


def _batch_key(r: StageRequest):
    f = r.flow
    return (r.window_start, f.num_steps, f.cfg_scale, f.noise_level, f.sde_type, f.window_size, r.group_size,
            r.decode.max_new_tokens, r.decode.temperature, r.decode.top_p, r.decode.answers_per_latent,
            r.decode.stop_at_code_block, r.return_prefix_embeds, r.decode_answers, r.latent_init)


class LaDiStageRunner:
    def __init__(self, model, scheduler_factory, device):
        self.model = model
        self.scheduler_factory = scheduler_factory     # num_steps -> scheduler (on device)
        self.device = torch.device(device)
        self._schedulers = {}

    def scheduler(self, num_steps: int):
        if num_steps not in self._schedulers:
            self._schedulers[num_steps] = self.scheduler_factory(num_steps)
        return self._schedulers[num_steps]

    def _resolve_window(self, r: StageRequest):
        if r.window_start is None:
            rng = random.Random(r.seed) if r.seed is not None else None
            r.window_start = pick_window_start(r.flow, r.global_step, rng)
        return r

    @torch.no_grad()
    def run(self, requests: List[StageRequest]) -> List[dict]:
        requests = [self._resolve_window(r) for r in requests]
        out = [None] * len(requests)
        buckets = {}
        for i, r in enumerate(requests):
            buckets.setdefault(_batch_key(r), []).append(i)
        for idx in buckets.values():
            for i, o in zip(idx, self._run_bucket([requests[i] for i in idx])):
                out[i] = o
        return out

    def _run_bucket(self, reqs: List[StageRequest]) -> List[dict]:
        model, dev = self.model, self.device
        r0 = reqs[0]
        G, M, cfg = r0.group_size, r0.decode.answers_per_latent, r0.flow
        P = len(reqs)
        T = max(len(r.prompt_ids) for r in reqs)
        pad = model.tokenizer.pad_token_id
        ids = torch.full((P, T), pad, dtype=torch.long, device=dev)
        mask = torch.zeros((P, T), dtype=torch.long, device=dev)
        for i, r in enumerate(reqs):
            ids[i, :len(r.prompt_ids)] = torch.tensor(r.prompt_ids, dtype=torch.long, device=dev)
            mask[i, :len(r.prompt_ids)] = 1
        N, D = model.config.num_latents, model.config.latent_dim
        # per-request seeds keep the noise independent of request packing
        def _noise(seed):
            return torch.randn(N, D, dtype=torch.float32,
                               generator=torch.Generator().manual_seed(seed) if seed is not None else None)

        if r0.latent_init == "independent":
            noise = torch.stack([_noise(trajectory_seed(r.seed, g)) for r in reqs for g in range(G)]).to(dev)
        else:
            noise = torch.stack([_noise(None if r.seed is None else int(r.seed)) for r in reqs]).to(dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(r0.seed) + 1 if r0.seed is not None else random.randint(0, 2 ** 31 - 1))

        t0 = time.perf_counter()
        sched = self.scheduler(cfg.num_steps)
        ro = rollout_latent_groups(model, sched, ids, mask, G, cfg, r0.window_start, generator=gen, noise=noise,
                                   latent_init=r0.latent_init)
        z = ro.final_latents.to(model.dtype)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_latent = time.perf_counter() - t0

        B = P * G
        eids, emask = ids.repeat_interleave(G, 0), mask.repeat_interleave(G, 0)
        t1 = time.perf_counter()
        if r0.decode_answers:
            fences = [r.open_fence for r in reqs for _ in range(G * M)]
            di, dm, dz = eids.repeat_interleave(M, 0), emask.repeat_interleave(M, 0), z.repeat_interleave(M, 0)
            answers, lps = [], []
            bs = max(int(r0.decode.batch_size), 1)
            for j in range(0, di.shape[0], bs):
                T = int(dm[j:j + bs].sum(1).max())
                a, l = decode_answers(model, di[j:j + bs, :T], dm[j:j + bs, :T], dz[j:j + bs], r0.decode,
                                      open_fences=fences[j:j + bs], generator=gen)
                answers += a
                lps += l
        else:
            answers, lps = [[] for _ in range(B * M)], [[] for _ in range(B * M)]
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_ar = time.perf_counter() - t1
        a_ids, a_mask = pad_answers(answers, model.tokenizer.pad_token_id)
        a_lp = torch.zeros(a_ids.shape, dtype=torch.float32)
        for i, lp in enumerate(lps):
            if lp:
                a_lp[i, :len(lp)] = torch.tensor(lp, dtype=torch.float32)
        L = a_ids.shape[1]
        a_ids, a_mask, a_lp = a_ids.view(P, G, M, L), a_mask.view(P, G, M, L), a_lp.view(P, G, M, L)

        prefix = None
        if r0.return_prefix_embeds:
            prefix = text_prefix_embeds(model, eids, z)                       # [B, T + N + 2, H]
            pmask = torch.cat([emask, emask.new_ones(B, N + 2)], 1)

        timing = torch.tensor([[t_latent, t_ar]], dtype=torch.float32)
        W = cfg.window_size
        outs = []
        for p, r in enumerate(reqs):
            sl = slice(p * G, (p + 1) * G)
            o = {
                "all_latents": ro.all_latents[sl].float().cpu().unsqueeze(0),
                "all_log_probs": ro.all_log_probs[sl].float().cpu().unsqueeze(0),
                "all_timesteps": ro.all_timesteps[sl].float().cpu().unsqueeze(0),
                "final_latents": z[sl].cpu().unsqueeze(0),
                "window": torch.tensor([[r.window_start, r.window_start + W]], dtype=torch.long),
                "answer_ids": a_ids[p:p + 1].clone(),
                "answer_mask": a_mask[p:p + 1].clone(),
                "rollout_text_log_probs": a_lp[p:p + 1].clone(),
                "timing": timing.clone(),
            }
            if prefix is not None:
                o["prefix_embeds"] = prefix[sl].cpu().unsqueeze(0)
                o["prefix_mask"] = pmask[sl].cpu().unsqueeze(0)
            outs.append(o)
        return outs
