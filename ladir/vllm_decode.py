"""Answer decoding with vLLM from prompt embeddings.

Answer decoding is a plain causal LM pass over [prompt, <bot>, ce_projector(z), <eot>] embeddings, so it can run in
vLLM (`enable_prompt_embeds`) on the backbone exported with `export_backbone` (plain HF Qwen3, vocab 151673).
"""

import os

import torch

# Stop once the first fenced code block closes. "\n```\n" matches a closing fence but not an opening "```python\n".
CLOSE_FENCE_STOP = ["\n```\n"]


def export_backbone(model, out_dir: str):
    """Save model.backbone_llm (+ tokenizer) as a standalone HF checkpoint for vLLM."""
    if os.path.exists(os.path.join(out_dir, "model.safetensors")) or os.path.exists(
            os.path.join(out_dir, "model.safetensors.index.json")):
        return out_dir
    model.backbone_llm.config.use_cache = True
    model.backbone_llm.save_pretrained(out_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(out_dir)
    return out_dir


class VLLMDecoder:
    def __init__(self, backbone_dir: str, gpu_memory_utilization: float = 0.45, max_model_len: int = 4096, **kw):
        from vllm import LLM
        self.llm = LLM(model=backbone_dir, enable_prompt_embeds=True, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, dtype="bfloat16", enable_prefix_caching=False, **kw)

    @torch.no_grad()
    def decode(self, model, prompt_ids, prompt_mask, latents, max_new_tokens=1024, temperature=0.0, top_p=1.0,
               stop_at_code_block=True, seed=None, prompt_open_fences=None, max_rounds=4):
        """Same inputs/outputs as ladir.sampling.decode (right-padded prompts; returns list[str]).

        A stop on a bare opening fence (odd fence count) continues the sample from its own text."""
        from vllm import SamplingParams
        tok = model.tokenizer
        prefix = model.text_prefix(prompt_ids, latents)
        N, Tpad = latents.shape[1], prompt_ids.shape[1]
        base = []
        for i in range(prompt_ids.shape[0]):
            T = int(prompt_mask[i].sum())
            base.append(torch.cat([prefix[i, :T], prefix[i, Tpad:Tpad + N + 2]]).to(torch.bfloat16))
        open_f = prompt_open_fences or [0] * len(base)
        gen_ids = [[] for _ in base]
        todo = list(range(len(base)))
        for _ in range(max_rounds if stop_at_code_block else 1):
            if not todo:
                break
            reqs, budgets = [], []
            for i in todo:
                cont = model.embed(torch.tensor(gen_ids[i], device=prompt_ids.device, dtype=torch.long)) \
                    if gen_ids[i] else base[i][:0]
                reqs.append({"prompt_embeds": torch.cat([base[i], cont.to(torch.bfloat16)]).cpu()})
                budgets.append(max_new_tokens - len(gen_ids[i]))
            outs = []
            for budget in sorted(set(budgets)):   # one generate call per remaining-token budget
                idx = [j for j, b in enumerate(budgets) if b == budget]
                sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=budget, seed=seed,
                                    stop=CLOSE_FENCE_STOP if stop_at_code_block else None,
                                    include_stop_str_in_output=True, skip_special_tokens=True)
                res = self.llm.generate([reqs[j] for j in idx], sp, use_tqdm=False)
                outs += list(zip(idx, res))
            next_todo = []
            for j, o in sorted(outs):
                i = todo[j]
                gen_ids[i] += list(o.outputs[0].token_ids)
                text = tok.decode(gen_ids[i], skip_special_tokens=True)
                stopped = o.outputs[0].finish_reason == "stop"
                if stop_at_code_block and stopped and (open_f[i] + text.count("```")) % 2 == 1 \
                        and len(gen_ids[i]) < max_new_tokens:
                    next_todo.append(i)            # stopped on an opening fence: keep going
            todo = next_todo
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
        if stop_at_code_block:
            texts = [t[:-1] if t.endswith("```\n") else t for t in texts]   # drop the newline after the fence
        return texts
