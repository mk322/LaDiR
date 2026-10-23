"""Latent sampling and answer decoding for LaDiR models."""

from dataclasses import dataclass, replace

import torch
from transformers.cache_utils import DynamicCache

from .modeling.flow import make_schedule, positions_from_mask


@dataclass(frozen=True)
class SamplerConfig:
    num_steps: int = 50
    cfg_scale: float = 4.0


def get_sampler_config(**overrides) -> SamplerConfig:
    return replace(SamplerConfig(), **overrides)


# --------------------------------------------------------------------------------------------------------------
# latent sampling
# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_latents(model, prompt_ids, prompt_mask, sc: SamplerConfig, generator: torch.Generator = None,
                   noise: torch.Tensor = None):
    """Euler integration of the flow from Gaussian noise with classifier-free guidance.

    prompt_ids/prompt_mask: [B, T] right-padded. Returns latents [B, N, D] in the model dtype.
    """
    B = prompt_ids.shape[0]
    N, D = model.config.num_latents, model.config.latent_dim
    if noise is None:
        noise = torch.randn(B, N, D, device=prompt_ids.device, generator=generator, dtype=torch.float32)
    x = noise.to(torch.float32)
    schedule = make_schedule(sc.num_steps, model.config.num_train_timesteps)
    use_cfg = sc.cfg_scale > 1.0
    embeds = model.embed(prompt_ids)
    if use_cfg:
        embeds = torch.cat([embeds, model.null_prompt_embeds(embeds)])
        ids, mask = prompt_ids.repeat(2, 1), prompt_mask.repeat(2, 1)
    else:
        ids, mask = prompt_ids, prompt_mask
    for label, dt in schedule:
        xin = x.to(model.dtype)
        t = torch.full((ids.shape[0],), label, device=x.device, dtype=torch.long)
        v = model.flow_velocity(ids, mask, xin.repeat(2, 1, 1) if use_cfg else xin, t, prompt_embeds=embeds)
        if use_cfg:
            cond, uncond = v.chunk(2)
            v = uncond + sc.cfg_scale * (cond - uncond)
        x = x + dt * v.to(torch.float32)
    return x.to(model.dtype)


# --------------------------------------------------------------------------------------------------------------
# answer decoding
# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def decode(model, prompt_ids, prompt_mask, latents, max_new_tokens=1024, temperature=0.0, top_p=1.0,
           stop_at_code_block=True, prompt_open_fences=None, generator=None):
    """Batched decoding of [prompt, <bot>, ce_projector(z), <eot>] -> answer texts.

    stop_at_code_block: also stop a sample once its first fenced code block closes, and cut the text there.
    prompt_open_fences: per-sample 0/1 if the prompt itself ends inside an open ``` block.
    """
    tok = model.tokenizer
    B, N = latents.shape[0], latents.shape[1]
    prefix = model.text_prefix(prompt_ids, latents)
    mask = torch.cat([prompt_mask, prompt_mask.new_ones(B, N + 2)], dim=1)
    pos = positions_from_mask(mask)
    cache = DynamicCache()
    out = model.backbone_llm(inputs_embeds=prefix, attention_mask=mask, position_ids=pos, past_key_values=cache,
                             use_cache=True, return_dict=True)
    logits = out.logits[:, -1].float()
    next_pos = pos[:, -1:] + 1
    gen = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
    open_f = prompt_open_fences or [0] * B
    for step in range(max_new_tokens):
        if temperature > 0:
            nxt = _sample_token(logits, temperature, top_p, generator)
        else:
            nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
        for i in range(B):
            if not done[i]:
                if nxt[i].item() == tok.eos_token_id:
                    done[i] = True
                else:
                    gen[i].append(nxt[i].item())
        if stop_at_code_block and step % 8 == 7:
            for i in range(B):
                if not done[i] and open_f[i] + tok.decode(gen[i]).count("```") >= 2:
                    done[i] = True
        if done.all():
            break
        mask = torch.cat([mask, (~done).long().unsqueeze(1)], dim=1)
        out = model.backbone_llm(input_ids=nxt.unsqueeze(1), attention_mask=mask, position_ids=next_pos,
                                 past_key_values=cache, use_cache=True, return_dict=True)
        logits = out.logits[:, -1].float()
        next_pos = next_pos + 1
    texts = [tok.decode(g, skip_special_tokens=True) for g in gen]
    if stop_at_code_block:
        texts = [_cut_after_first_block(t, o) for t, o in zip(texts, open_f)]
    return texts


def _cut_after_first_block(text, open_fence):
    parts = text.split("```")
    need = 2 - open_fence   # fences in `text` up to and including the one that closes the first block
    if len(parts) - 1 >= need:
        return "```".join(parts[:need]) + "```"
    return text


def _sample_token(logits, temperature, top_p, generator):
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        sp = sp * keep
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
