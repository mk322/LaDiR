"""Answer rollout on a colocated vLLM replica fed prompt embeddings [prompt, <bot>, ce_projector(z), <eot>].

Samples stop on a closing code fence; a stop on an opening fence continues the sample. The actor syncs the backbone
into the replica after every update.
"""

from __future__ import annotations

import copy
from typing import Awaitable, Callable, List, Optional, Sequence
from uuid import uuid4

import torch
from omegaconf import OmegaConf

ROLLOUT_NAME = "ladi_vllm_ar"
BACKBONE_PREFIX = "backbone_llm."
SYNC_PREFIX = "transformer."
try:
    from ladir.vllm_decode import CLOSE_FENCE_STOP
except Exception:  # pragma: no cover
    CLOSE_FENCE_STOP = ["\n```\n"]


def backbone_weight_name(name: str) -> Optional[str]:
    """Actor weight name -> HF Qwen3 name (None for non-backbone weights)."""
    if name.startswith(SYNC_PREFIX):
        name = name[len(SYNC_PREFIX):]
    if not name.startswith(BACKBONE_PREFIX):
        return None
    return name[len(BACKBONE_PREFIX):]


def backbone_weights(per_tensor_param):
    for name, t in per_tensor_param:
        hf = backbone_weight_name(name)
        if hf is not None:
            yield hf, t


def ar_backbone_dir(config) -> str:
    import os
    lad = OmegaConf.select(config, "actor_rollout_ref.ladi", default={}) or {}
    d = lad.get("ar_model_path") if hasattr(lad, "get") else None
    if d:
        return str(d)
    return os.path.join(str(config.actor_rollout_ref.model.path), "backbone")


def ar_rollout_dict(config) -> dict:
    """verl RolloutConfig of the AR replicas, as a dict."""
    r = config.actor_rollout_ref.rollout
    lad = OmegaConf.to_container(OmegaConf.select(config, "actor_rollout_ref.ladi", default={}) or {}, resolve=True)
    ar = lad.get("ar", {}) or {}
    max_new = int(lad.get("max_new_tokens", 768))
    prompt_len = int(config.data.max_prompt_length) + 66
    return {
        "_target_": "verl.workers.config.RolloutConfig",
        "name": ROLLOUT_NAME,
        "mode": "async",
        "n_gpus_per_node": int(r.n_gpus_per_node),
        "tensor_model_parallel_size": int(ar.get("tensor_model_parallel_size", 1)),
        "gpu_memory_utilization": float(ar.get("gpu_memory_utilization", 0.3)),
        "enforce_eager": bool(ar.get("enforce_eager", False)),
        "free_cache_engine": True,
        "load_format": "dummy",
        "prompt_length": prompt_len,
        "response_length": max_new,
        "max_model_len": prompt_len + max_new + 8,
        "max_num_seqs": int(ar.get("max_num_seqs", 256)),
        "max_num_batched_tokens": int(ar.get("max_num_batched_tokens", 16384)),
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "logprobs_mode": "processed_logprobs",
        "temperature": float(lad.get("temperature", 1.0)),
        "top_p": float(lad.get("top_p", 1.0)),
        "calculate_log_probs": True,
        "engine_kwargs": {"vllm": {"enable_prompt_embeds": True, **(ar.get("engine_kwargs", {}) or {})}},
        "checkpoint_engine": {"_target_": "verl.workers.config.CheckpointEngineConfig", "backend": "naive",
                              "update_weights_bucket_megabytes":
                              int(r.checkpoint_engine.update_weights_bucket_megabytes)},
        "agent": {"_target_": "verl.workers.config.AgentLoopConfig", "num_workers": int(r.agent.num_workers)},
    }


def ar_model_dict(config) -> dict:
    return {"_target_": "verl.workers.config.HFModelConfig", "path": ar_backbone_dir(config),
            "trust_remote_code": False}


def ar_trainer_config(config):
    """Copy of the trainer config whose rollout/model sections describe the AR replicas (for LLMServerManager)."""
    import dataclasses

    from verl.utils.config import omega_conf_to_dataclass

    c = copy.deepcopy(config)
    OmegaConf.set_struct(c, False)
    rc = omega_conf_to_dataclass(OmegaConf.create(ar_rollout_dict(config)))

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            d = {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj) if not f.name.startswith("_")}
            d["_target_"] = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return d
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_dict(v) for v in obj]
        return obj

    c.actor_rollout_ref.rollout = OmegaConf.create(to_dict(rc))
    c.actor_rollout_ref.model = OmegaConf.create(ar_model_dict(config))
    return c


def needs_continuation(text: str, open_fence: int, finish_reason: Optional[str], n_gen: int, max_new: int) -> bool:
    """A stop that leaves the first code block open was an opening fence: continue the sample."""
    return finish_reason == "stop" and (open_fence + text.count("```")) % 2 == 1 and n_gen < max_new


def vllm_sampling_dict(max_tokens: int, temperature: float, top_p: float, seed: Optional[int],
                       stop_at_code_block: bool = True) -> dict:
    return {"max_tokens": int(max_tokens), "temperature": float(temperature), "top_p": float(top_p),
            "seed": seed, "logprobs": 0, "stop": list(CLOSE_FENCE_STOP) if stop_at_code_block else None,
            "include_stop_str_in_output": True, "skip_special_tokens": True, "n": 1}


async def sample_with_continuation(prefix: torch.Tensor, gen_fn: Callable[..., Awaitable[dict]],
                                   embed_fn: Callable[[List[int]], Awaitable[torch.Tensor]],
                                   decode_fn: Callable[[List[int]], str], max_new_tokens: int, temperature: float,
                                   top_p: float, seed: Optional[int], open_fence: int = 0,
                                   stop_at_code_block: bool = True, max_rounds: int = 4) -> dict:
    """One answer: repeated vLLM calls on [prefix, embed(generated so far)].

    gen_fn(prompt_embeds [L, H], sampling dict) -> {"token_ids", "log_probs", "finish_reason"}
    Returns {"token_ids", "log_probs" (sampled-token log-probs, one per token), "finish_reason", "rounds"}.
    """
    ids: List[int] = []
    lps: List[float] = []
    finish = None
    rounds = 0
    for rounds in range(1, (max_rounds if stop_at_code_block else 1) + 1):
        cont = await embed_fn(ids) if ids else prefix[:0]
        emb = torch.cat([prefix, cont.to(prefix.dtype)])
        sp = vllm_sampling_dict(max_new_tokens - len(ids), temperature, top_p,
                                None if seed is None else seed + 7919 * (rounds - 1), stop_at_code_block)
        out = await gen_fn(emb, sp)
        new = list(out["token_ids"])
        ids += new
        lp = out.get("log_probs")
        lps += list(lp) if lp is not None else [0.0] * len(new)
        finish = out.get("finish_reason")
        if not (stop_at_code_block and needs_continuation(decode_fn(ids), open_fence, finish, len(ids),
                                                          max_new_tokens)):
            break
    return {"token_ids": ids, "log_probs": lps, "finish_reason": finish, "rounds": rounds}


def _worker_extension_base():
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
    return vLLMColocateWorkerExtension


class LaDiARWorkerExtension(_worker_extension_base()):
    """verl's colocated vLLM worker extension + token embedding lookup for fence continuations."""

    def ladi_embed_tokens(self, token_ids: List[int]):
        model = self.model_runner.get_model() if hasattr(self.model_runner, "get_model") else self.model_runner.model
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if hasattr(model, "embed_input_ids"):          # vLLM 0.24 Qwen3ForCausalLM API
                out = model.embed_input_ids(ids)
            else:
                out = model.model.embed_tokens(ids)
        return pack_embeds(out)


def pack_embeds(x: torch.Tensor):
    """Embeddings as (shape, raw bf16 bytes) for vLLM's engine-core RPC."""
    x = x.detach().to(torch.bfloat16).contiguous().cpu()
    return [list(x.shape), x.view(torch.int16).numpy().tobytes()]


def unpack_embeds(obj) -> torch.Tensor:
    """Inverse of pack_embeds -> [L, H] bf16."""
    while isinstance(obj, (list, tuple)) and len(obj) == 1 and not isinstance(obj[0], (int, float)):
        obj = obj[0]
    if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[1], (bytes, bytearray, memoryview)):
        shape, buf = obj
        return torch.frombuffer(bytearray(buf), dtype=torch.int16).view(torch.bfloat16).reshape(list(shape)).clone()
    t = obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj, dtype=torch.float32)
    return t.to(torch.bfloat16).reshape(-1, t.shape[-1])


def _server_base():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    return vLLMHttpServer


class LaDiARHttpServer(_server_base()):
    def _get_worker_extension_cls(self) -> str:
        return f"{LaDiARWorkerExtension.__module__}.{LaDiARWorkerExtension.__qualname__}"

    async def _generate_once(self, prompt_embeds: torch.Tensor, sp: dict) -> dict:
        from vllm import SamplingParams
        final = None
        async for out in self.engine.generate({"prompt_embeds": prompt_embeds}, SamplingParams(**sp),
                                              request_id=uuid4().hex):
            final = out
        o = final.outputs[0]
        lps = None
        if o.logprobs is not None:
            lps = [d[t].logprob for t, d in zip(o.token_ids, o.logprobs)]
        return {"token_ids": list(o.token_ids), "log_probs": lps, "finish_reason": o.finish_reason}

    async def _embed(self, token_ids: List[int]) -> torch.Tensor:
        res = await self.engine.collective_rpc("ladi_embed_tokens", args=(list(token_ids),))
        return unpack_embeds(res[0])

    async def generate_from_embeds(self, prompt_embeds: torch.Tensor, max_new_tokens: int, temperature: float,
                                   top_p: float = 1.0, seed: Optional[int] = None, open_fence: int = 0,
                                   stop_at_code_block: bool = True) -> dict:
        tok = self.model_config.tokenizer
        return await sample_with_continuation(
            prompt_embeds.to(torch.bfloat16), self._generate_once, self._embed,
            lambda ids: tok.decode(ids, skip_special_tokens=True), max_new_tokens, temperature, top_p, seed,
            open_fence, stop_at_code_block)


def _replica_base():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
    return vLLMReplica


class LaDiARReplica(_replica_base()):
    def __init__(self, *args, **kwargs):
        import ray
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(LaDiARHttpServer)

    def _get_server_name_prefix(self) -> str:
        return f"{ROLLOUT_NAME}_"


def _client_base():
    from verl.workers.rollout.llm_server import LLMServerClient
    return LLMServerClient


class LaDiARClient(_client_base()):
    async def generate_from_embeds(self, request_id: str, **kwargs) -> dict:
        server_id, server = await self._acquire_server(request_id)
        try:
            return await server.generate_from_embeds.remote(**kwargs)
        finally:
            self._release_server(server_id)


def register():
    """Register the AR replica (driver side) and the worker-side ServerAdapter for ROLLOUT_NAME."""
    from verl.workers.rollout.base import _ROLLOUT_REGISTRY
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    RolloutReplicaRegistry.register(ROLLOUT_NAME, lambda: LaDiARReplica)
    _ROLLOUT_REGISTRY[(ROLLOUT_NAME, "async")] = "verl.workers.rollout.vllm_rollout.ServerAdapter"


def per_answer_outputs(results: Sequence[dict], max_new: int):
    """[{"token_ids", "log_probs"}] -> (ids lists, log-prob lists) truncated to max_new."""
    ids = [list(r["token_ids"])[:max_new] for r in results]
    lps = [list(r["log_probs"])[:max_new] for r in results]
    for a, b in zip(ids, lps):
        assert len(a) == len(b), (len(a), len(b))
    return ids, lps
