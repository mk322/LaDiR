"""Per-token answer log-probs given [prompt, <bot>, ce_projector(z), <eot>]."""

import torch
import torch.utils.checkpoint

from ladir.modeling.flow import positions_from_mask


def _row_log_probs(logits, ids, temperature: float):
    return torch.log_softmax(logits.float() / temperature, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def answer_log_probs(model, prompt_ids, prompt_mask, latents, answer_ids, answer_mask, temperature: float = 1.0,
                     checkpoint_rows: bool = True):
    """[B, L] fp32 log-probs of answer_ids (0 where masked); latents are fixed context.

    The LM head runs on answer positions only; the per-row log_softmax is recomputed in backward."""
    B, N = latents.shape[:2]
    prefix = model.text_prefix(prompt_ids, latents.detach())
    embeds = torch.cat([prefix, model.embed(answer_ids)], dim=1)
    mask = torch.cat([prompt_mask, prompt_mask.new_ones(B, N + 2), answer_mask], dim=1)
    llm = model.backbone_llm
    hidden = llm.model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions_from_mask(mask),
                       use_cache=False, return_dict=True).last_hidden_state
    start = prefix.shape[1]
    hidden = hidden[:, start - 1:start - 1 + answer_ids.shape[1]]
    logits = llm.lm_head(hidden)
    rows = []
    for i in range(B):
        if checkpoint_rows and torch.is_grad_enabled():
            rows.append(torch.utils.checkpoint.checkpoint(_row_log_probs, logits[i], answer_ids[i], temperature,
                                                          use_reentrant=False))
        else:
            rows.append(_row_log_probs(logits[i], answer_ids[i], temperature))
    return torch.stack(rows) * answer_mask
