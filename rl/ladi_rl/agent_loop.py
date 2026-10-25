"""Agent loop: one rollout request per prompt (G latents x M answers), answers on vLLM, unit-test rewards."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await

from .common import pick_window_start
from .reward import all_pass_reward, pass_rate
from .training_adapter import flow_config_from_sampling

logger = logging.getLogger(__name__)


def ladi_settings(config) -> dict:
    lad = OmegaConf.select(config, "actor_rollout_ref.ladi", default=None)
    return OmegaConf.to_container(lad, resolve=True) if lad is not None else {}


def rollout_sampling_params(config, global_steps: int) -> dict:
    """Sampling params of a latent rollout request."""
    r = config.actor_rollout_ref.rollout
    lad = ladi_settings(config)
    sp = {
        "num_inference_steps": int(r.pipeline.num_inference_steps),
        "guidance_scale": r.pipeline.get("guidance_scale", None),
        "noise_level": float(r.algo.noise_level),
        "sde_type": str(r.algo.sde_type),
        "sde_window_size": r.algo.sde_window_size,
        "sde_window_range": list(r.algo.sde_window_range) if r.algo.sde_window_range is not None else None,
        "sample_strategy": r.algo.get("sample_strategy", "random"),
        "iters_per_group": r.algo.get("iters_per_group", 1),
        "logprobs": True,
        "global_steps": int(global_steps),
        "max_new_tokens": int(lad.get("max_new_tokens", 768)),
        "temperature": float(lad.get("temperature", 1.0)),
        "top_p": float(lad.get("top_p", 1.0)),
        "stop_at_code_block": bool(lad.get("stop_at_code_block", True)),
        "answers_per_latent": int(lad.get("answers_per_latent", 1)),
        "decode_batch_size": int(lad.get("decode_batch_size", 64)),
        "decode_answers": True,
        "ladi_request": True,
        "latent_init": str(lad.get("latent_init", "independent")),
    }
    if sp["guidance_scale"] is None:
        sp.pop("guidance_scale")
    return sp


def request_seeds(meta_info: dict, n: int, global_indices=None) -> list:
    """One seed per prompt row from the step's rollout_seed and the global row index; random if unseeded."""
    from verl_omni.agent_loop.utils import maybe_per_rollout_seeds
    if global_indices is not None:
        global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
    seeds = maybe_per_rollout_seeds(meta_info, n, global_indices)
    if seeds is None:
        return [random.randint(0, 2 ** 31 - 1) for _ in range(n)]
    return [int(s) % (2 ** 31 - 1) for s in seeds]


def window_seed(meta_info: dict, global_steps: int) -> int:
    base = meta_info.get("rollout_seed")
    return (int(base) * 7919 + 17) if base is not None else random.randint(0, 2 ** 31 - 1)


def answer_text(tokenizer, token_ids, open_fence: int, stop_at_code_block: bool = True) -> str:
    """Decoded answer cut after the first closed code block (as in evaluation)."""
    from ladir.sampling import _cut_after_first_block
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return _cut_after_first_block(text, open_fence) if stop_at_code_block else text


class LaDiAgentLoopWorker:
    def __init__(self, config, llm_client, ar_client=None, reward_loop_worker_handles=None):
        from ladir.tokenizer import load_tokenizer

        self.config = config
        self.client = llm_client
        self.ar_client = ar_client
        self.lad = ladi_settings(config)
        self.ar_backend = str(self.lad.get("ar_backend", "vllm"))
        if self.ar_backend == "vllm" and ar_client is None:
            raise ValueError("ar_backend=vllm needs the AR replicas' client")
        self.rollout_config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.tokenizer = load_tokenizer(model_path if os.path.isdir(model_path) else "Qwen/Qwen3-8B-Base")
        self.pool = ThreadPoolExecutor(int(self.lad.get("reward_workers", 16)))
        self.max_prompt_len = int(config.data.max_prompt_length)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        t_start = time.perf_counter()
        G = int(batch.meta_info.get("group_size", self.rollout_config.n))
        validate = bool(batch.meta_info.get("validate", False))
        gstep = int(batch.meta_info.get("global_steps", 0))
        sp = rollout_sampling_params(self.config, gstep)
        if validate:
            sp["noise_level"] = 0.0
            sp["temperature"] = float(self.lad.get("val_temperature", 0.0) or 0.0)
            sp["answers_per_latent"] = 1
        io_timeout = None if validate else (self.lad.get("train_io_timeout", None) or None)
        # train_reward: pass_rate | all_pass; validation reports the pass rate
        all_pass = not validate and self.lad.get("train_reward", "pass_rate") == "all_pass"
        if self.ar_backend == "vllm":
            sp["decode_answers"] = False
            sp["return_prefix_embeds"] = True
        flow = flow_config_from_sampling(sp).validate()
        seeds = request_seeds(batch.meta_info, len(batch),
                              batch.non_tensor_batch.get("_rollout_seed_global_idx"))
        rows = [{k: v[i] for k, v in batch.non_tensor_batch.items()} for i in range(len(batch))]
        step_ws = pick_window_start(flow, gstep, random.Random(window_seed(batch.meta_info, gstep)))
        tasks = []
        for i, row in enumerate(rows):
            ws = step_ws if self.lad.get("window_scope", "step") == "step" else \
                pick_window_start(flow, gstep, random.Random(seeds[i]))
            req_sp = dict(sp, num_outputs_per_prompt=G, seed=seeds[i], sde_window_start=ws)
            tasks.append(asyncio.create_task(self._one(row, req_sp, io_timeout, all_pass)))
        results = await asyncio.gather(*tasks)
        out = self._assemble(rows, results, G, int(sp["answers_per_latent"]), int(sp["max_new_tokens"]))
        batches = {r["timing"] for r in results}
        ar_s = float(sum(t[1] for t in batches))
        if self.ar_backend == "vllm":
            ar_s = float(max(r["ar_end"] for r in results) - min(r["ar_start"] for r in results))
        out.meta_info["ladi_timing"] = {
            "latent_rollout_s": float(sum(t[0] for t in batches)),
            "ar_rollout_s": ar_s,
            "reward_cpu_s": float(sum(r["reward_s"] for r in results)),
            "reward_tail_s": float(max(r["reward_done"] for r in results) - max(r["gen_done"] for r in results)),
            "worker_wall_s": time.perf_counter() - t_start,
        }
        return out

    async def _one(self, row: dict, sp: dict, io_timeout=None, all_pass: bool = False) -> dict:
        prompt = str(row["prompt"])
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_prompt_len:
            logger.warning("prompt of %d tokens truncated to %d (head kept)", len(ids), self.max_prompt_len)
            ids = ids[:self.max_prompt_len]
        from eval.benchmarks import open_fence
        sp = dict(sp, open_fence=int(open_fence(prompt)))
        t0 = time.perf_counter()
        res = await self.client.generate(request_id=uuid4().hex, prompt_ids=ids, sampling_params=sp)
        gen_done = time.perf_counter()
        ef = dict(res.extra_fields)
        if "all_log_probs" not in ef and res.log_probs is not None:
            ef["all_log_probs"] = torch.as_tensor(res.log_probs)
        ar_start = time.perf_counter()
        if self.ar_backend == "vllm" and "prefix_embeds" in ef:
            ef.update(await self._ar_answers(ef, sp))
            gen_done = time.perf_counter()
        ar_end = time.perf_counter()
        if "answer_ids" not in ef:
            raise RuntimeError(f"LaDi rollout returned no trajectory (stop_reason={res.stop_reason})")
        a_ids, a_mask = ef["answer_ids"], ef["answer_mask"]              # [G, M, L]
        G, M = a_ids.shape[0], a_ids.shape[1]
        texts = [[answer_text(self.tokenizer, a_ids[g, m][a_mask[g, m].bool()].tolist(), int(sp["open_fence"]),
                              bool(sp["stop_at_code_block"])) for m in range(M)] for g in range(G)]
        loop = asyncio.get_running_loop()
        t1 = time.perf_counter()
        score = functools.partial(pass_rate, io_timeout=io_timeout)
        futs = [loop.run_in_executor(self.pool, score, texts[g][m], prompt, row["tests"], row.get("extra"))
                for g in range(G) for m in range(M)]
        rewards = await asyncio.gather(*futs)
        if all_pass:
            rewards = [all_pass_reward(r) for r in rewards]
        reward_done = time.perf_counter()
        timing = tuple(float(x) for x in torch.as_tensor(ef["timing"]).reshape(-1).tolist())
        return {"ids": ids, "ef": ef, "texts": texts, "rewards": rewards, "timing": timing, "gen_s": gen_done - t0,
                "ar_start": ar_start, "ar_end": ar_end, "seed": int(sp.get("seed") or -1),
                "reward_s": reward_done - t1, "gen_done": gen_done, "reward_done": reward_done}

    async def _ar_answers(self, ef: dict, sp: dict) -> dict:
        """M answers for each of the G latents on the vLLM replicas."""
        pre, pmask = ef["prefix_embeds"], ef["prefix_mask"]                # [G, Lp, H], [G, Lp]
        G, M = pre.shape[0], int(sp["answers_per_latent"])
        seed0 = int(sp.get("seed") or 0)
        calls = []
        for g in range(G):
            emb = pre[g][pmask[g].bool()]
            for m in range(M):
                calls.append(self.ar_client.generate_from_embeds(
                    request_id=uuid4().hex, prompt_embeds=emb, max_new_tokens=int(sp["max_new_tokens"]),
                    temperature=float(sp["temperature"]), top_p=float(sp["top_p"]),
                    seed=(seed0 * 131 + g * M + m) % (2 ** 31), open_fence=int(sp.get("open_fence", 0)),
                    stop_at_code_block=bool(sp["stop_at_code_block"])))
        outs = await asyncio.gather(*calls)
        return pack_ar_outputs(outs, G, M, int(sp["max_new_tokens"]), self.tokenizer.pad_token_id)

    def _assemble(self, rows, results, G, M, max_new) -> DataProto:
        P = len(rows)
        B = P * G
        pad = self.tokenizer.pad_token_id
        T = self.max_prompt_len
        prompt_ids = torch.full((B, T), pad, dtype=torch.long)
        prompt_mask = torch.zeros((B, T), dtype=torch.long)
        a_ids = torch.full((B, M, max_new), pad, dtype=torch.long)
        a_mask = torch.zeros((B, M, max_new), dtype=torch.long)
        a_lp = torch.zeros((B, M, max_new), dtype=torch.float32)
        lat, lps, ts, fin, rew = [], [], [], [], []
        non = {k: [] for k in ("uid", "task_id", "group", "answer_text", "reward_info", "window_start", "gen_seconds",
                               "seed")}
        for p, (row, r) in enumerate(zip(rows, results)):
            ef = r["ef"]
            n = len(r["ids"])
            for g in range(G):
                b = p * G + g
                prompt_ids[b, :n] = torch.tensor(r["ids"], dtype=torch.long)
                prompt_mask[b, :n] = 1
                L = min(ef["answer_ids"].shape[-1], max_new)
                a_ids[b, :, :L] = ef["answer_ids"][g, :, :L]
                a_mask[b, :, :L] = ef["answer_mask"][g, :, :L]
                a_lp[b, :, :L] = ef["rollout_text_log_probs"][g, :, :L].float()
                non["uid"].append(row.get("uid", str(p)))
                non["task_id"].append(row.get("task_id", ""))
                non["group"].append(row.get("group", ""))
                non["answer_text"].append(r["texts"][g])
                non["reward_info"].append([info for _, info in r["rewards"][g * M:(g + 1) * M]])
                non["window_start"].append(int(ef["window"].reshape(-1)[0]))
                non["gen_seconds"].append(r["gen_s"])
                non["seed"].append(int(r["seed"]))
            lat.append(ef["all_latents"].float())
            lps.append(ef["all_log_probs"].float())
            ts.append(ef["all_timesteps"].float())
            fin.append(ef["final_latents"])
            rew.append(torch.tensor([[rate for rate, _ in r["rewards"][g * M:(g + 1) * M]] for g in range(G)]))
        batch = {
            "prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "prompts": prompt_ids,
            "all_latents": torch.cat(lat), "rollout_log_probs": torch.cat(lps), "all_timesteps": torch.cat(ts),
            "final_latents": torch.cat(fin).to(torch.bfloat16),
            "answer_ids": a_ids, "answer_mask": a_mask, "rollout_text_log_probs": a_lp,
            "rewards": torch.cat(rew).float(),
        }
        non_tensor = {k: np.array(v, dtype=object) for k, v in non.items()}
        metrics = [{"generate_sequences": r["gen_s"], "tool_calls": 0.0, "compute_score": r["reward_s"],
                    "num_preempted": -1} for r in results for _ in range(G)]
        return DataProto.from_dict(tensors=batch, non_tensors=non_tensor, meta_info={"metrics": metrics})


def pack_ar_outputs(outs, G: int, M: int, max_new: int, pad_id: int) -> dict:
    """[G*M] vLLM results (row g*M + m) -> answer_ids / answer_mask / rollout_text_log_probs [G, M, L]."""
    from .rollout import pad_answers
    from .vllm_ar import per_answer_outputs

    ids, lps = per_answer_outputs(outs, max_new)
    a_ids, a_mask = pad_answers(ids, pad_id)
    a_lp = torch.zeros(a_ids.shape, dtype=torch.float32)
    for i, lp in enumerate(lps):
        if lp:
            a_lp[i, :len(lp)] = torch.tensor(lp, dtype=torch.float32)
    L = a_ids.shape[1]
    return {"answer_ids": a_ids.view(G, M, L), "answer_mask": a_mask.view(G, M, L),
            "rollout_text_log_probs": a_lp.view(G, M, L),
            "ar_rounds": torch.tensor([o.get("rounds", 1) for o in outs]).view(G, M)}


class LaDiAgentLoopManager(AgentLoopManager):
    """verl AgentLoopManager with LaDi workers (output rows = input rows x G)."""

    def __init__(self, config, llm_client, ar_client=None, **kwargs):
        self.agent_loop_workers_class = ray.remote(LaDiAgentLoopWorker)
        self.ar_client = ar_client
        super().__init__(config, llm_client, **kwargs)

    async def _init_agent_loop_workers(self):
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        self.agent_loop_workers = []
        node_ids = [n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0]
        for i in range(self.rollout_config.agent.num_workers):
            self.agent_loop_workers.append(self.agent_loop_workers_class.options(
                name=f"ladi_agent_loop_worker_{i}_{uuid4().hex[:8]}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_ids[i % len(node_ids)], soft=True),
            ).remote(self.config, self.llm_client, self.ar_client))

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        n = len(self.agent_loop_workers)
        chunks = prompts.chunk(n) if len(prompts) % n == 0 else _split(prompts, n)
        outputs = await asyncio.gather(*[w.generate_sequences.remote(c)
                                         for w, c in zip(self.agent_loop_workers, chunks) if len(c) > 0])
        timings = [o.meta_info.pop("ladi_timing") for o in outputs]
        for o in outputs:
            o.meta_info.pop("metrics", None)
        output = DataProto.concat(outputs)
        agg = {}
        for k in timings[0]:
            vals = [t[k] for t in timings]
            agg[f"ladi_rollout/{k}/sum"] = float(np.sum(vals))
            agg[f"ladi_rollout/{k}/max"] = float(np.max(vals))
        output.meta_info = {"timing": agg}
        return output


def _split(data: DataProto, n: int):
    sizes = [len(data) // n + (1 if i < len(data) % n else 0) for i in range(n)]
    out, s = [], 0
    for k in sizes:
        out.append(data[s:s + k])
        s += k
    return out
