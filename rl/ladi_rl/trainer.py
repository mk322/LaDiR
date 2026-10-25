"""LaDi-RL trainer: verl-omni's PolicyGradientRayTrainer with a latent-window + answer-token data flow."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pprint import pprint

import numpy as np
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.ray_utils import auto_await

from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer

from .algo import group_index, hybrid_advantages, informative_rows, trim_to_multiple
from .losses import HybridLossConfig

TRAIN_TENSOR_KEYS = ("prompt_ids", "prompt_mask", "all_latents", "all_timesteps", "final_latents", "answer_ids",
                     "answer_mask")


def select_train_rows(rewards, groups, filter_zero_variance: bool, divisor: int):
    """Drop zero-variance groups, then whole groups until the row count divides `divisor`."""
    keep = informative_rows(rewards, groups)
    idx = np.arange(rewards.shape[0])
    if filter_zero_variance:
        idx = idx[keep.numpy()]
    return trim_to_multiple(idx, groups, divisor), keep


def train_row_divisor(config, dp_size: int) -> int:
    a = config.actor_rollout_ref
    mbs = [int(a.actor.ppo_micro_batch_size_per_gpu)]
    lp = a.rollout.get("log_prob_micro_batch_size_per_gpu", None)
    if lp:
        mbs.append(int(lp))
    return dp_size * math.lcm(*mbs)


def _mini_batch_rows(n_rows: int, target: int, unit: int) -> int:
    """Rows per mini-batch: a multiple of `unit` that splits n_rows into ~n_rows / target updates."""
    if n_rows <= 0:
        return max(n_rows, 1)
    units = max(n_rows // unit, 1)
    k = max(1, min(units, round(n_rows / max(target, 1))))
    while units % k:
        k -= 1
    return n_rows // k


def check_supported(config):
    lad = _ladi(config)
    if bool(config.actor_rollout_ref.actor.get("use_kl_loss", False)):
        raise NotImplementedError("actor.use_kl_loss is not supported")
    if lad.get("window_scope", "step") != "step" and config.actor_rollout_ref.model.algorithm == "grpo_guard":
        raise ValueError("grpo_guard needs ladi.window_scope=step")


def rollout_settings(config) -> dict:
    r, lad = config.actor_rollout_ref.rollout, _ladi(config)
    gs = r.pipeline.get("guidance_scale", None)
    return {"num_steps": int(r.pipeline.num_inference_steps), "cfg_scale": float(gs) if gs is not None else 1.0,
            "sde_type": str(r.algo.sde_type), "noise_level": float(r.algo.noise_level),
            "window_size": int(r.algo.sde_window_size or 2), "window_range": list(r.algo.sde_window_range or []),
            "window_scope": lad.get("window_scope", "step"), "latent_init": lad.get("latent_init", "independent"),
            "group_size": int(r.n), "answers_per_latent": int(lad.get("answers_per_latent", 1)),
            "temperature": float(lad.get("temperature", 1.0)), "top_p": float(lad.get("top_p", 1.0)),
            "max_new_tokens": int(lad.get("max_new_tokens", 768)), "ar_backend": lad.get("ar_backend", "vllm"),
            "flow_loss": str(config.actor_rollout_ref.model.algorithm),
            "flow_clip_ratio": float(config.actor_rollout_ref.actor.diffusion_loss.clip_ratio)}


@auto_await
async def _sleep_all(replicas):
    import asyncio
    await asyncio.gather(*[r.sleep() for r in replicas])


def _ladi(config) -> dict:
    lad = OmegaConf.select(config, "actor_rollout_ref.ladi", default=None)
    return OmegaConf.to_container(lad, resolve=True) if lad is not None else {}


def _group_metrics(rows):
    acc = {}
    for row in rows:
        if row["group"]:
            acc.setdefault(row["group"], []).append(row["reward"])
    out = {}
    for g, v in sorted(acc.items()):
        out[f"val/{g}/pass_rate_mean"] = float(np.mean(v))
        out[f"val/{g}/num_problems"] = len(v)
    return out


class LaDiRayTrainer(PolicyGradientRayTrainer):

    def _init_online_rollout_stack(self, actor_rollout_resource_pool):
        """Rollout replicas, the LaDi agent loop (which also computes rewards) and the checkpoint engine."""
        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.rollout.llm_server import LLMServerManager

        from .agent_loop import LaDiAgentLoopManager

        self.reward_loop_manager = None
        self.enable_agent_reward_loop = False
        self.async_rollout_mode = True
        self.llm_server_manager = LLMServerManager.create(config=self.config, worker_group=self.actor_rollout_wg,
                                                          rollout_resource_pool=actor_rollout_resource_pool)
        replicas = list(self.llm_server_manager.get_replicas())
        ar_client = None
        self.ar_server_manager = None
        if _ladi(self.config).get("ar_backend", "vllm") == "vllm":
            from .vllm_ar import LaDiARClient, ar_trainer_config, register
            register()
            _sleep_all(replicas)  # free stage-0 memory before vLLM sizes its KV cache
            self.ar_server_manager = LLMServerManager.create(config=ar_trainer_config(self.config),
                                                             worker_group=self.actor_rollout_wg,
                                                             rollout_resource_pool=actor_rollout_resource_pool)
            ar_client = self.ar_server_manager.get_client(client_cls=LaDiARClient)
            replicas += list(self.ar_server_manager.get_replicas())
        self.async_rollout_manager = LaDiAgentLoopManager.create(config=self.config,
                                                                 llm_client=self.llm_server_manager.get_client(),
                                                                 ar_client=ar_client)
        ckpt_cfg = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(config=ckpt_cfg, actor_wg=self.actor_rollout_wg,
                                                          replicas=replicas)
        if self.ar_server_manager is None:
            self.checkpoint_manager.sleep_replicas()
        else:
            # stage-0 replicas are already asleep and must not be slept twice
            _sleep_all(list(self.ar_server_manager.get_replicas()))

    def _loss_config(self) -> dict:
        lad = _ladi(self.config)
        hl = HybridLossConfig.from_dict(lad.get("loss", {}))
        hl.temperature = float(lad.get("temperature", 1.0))
        return hl.to_dict()

    def _dp_size(self) -> int:
        return int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)

    def _actor_td(self, batch: DataProto, keys):
        return DataProto.from_dict(tensors={k: batch.batch[k] for k in keys if k in batch.batch.keys()}).to_tensordict()

    def _trim_answers(self, batch: DataProto):
        """Trim padded answers / prompts to the longest one in the batch."""
        L = int(batch.batch["answer_mask"].sum(-1).max().clamp(min=1))
        for k in ("answer_ids", "answer_mask", "rollout_text_log_probs", "old_text_log_probs"):
            if k in batch.batch.keys():
                batch.batch[k] = batch.batch[k][..., :L].contiguous()
        T = int(batch.batch["prompt_mask"].sum(-1).max().clamp(min=1))
        for k in ("prompt_ids", "prompt_mask", "prompts"):
            if k in batch.batch.keys():
                batch.batch[k] = batch.batch[k][:, :T].contiguous()
        return batch

    def _compute_old_log_prob_ladi(self, batch: DataProto) -> dict:
        td = self._actor_td(batch, TRAIN_TENSOR_KEYS)
        guard = self.config.actor_rollout_ref.actor.diffusion_loss.get("loss_mode") == "grpo_guard"
        tu.assign_non_tensor(td, compute_loss=False, ladi_loss=self._loss_config(), return_prev_sample_mean=guard)
        out = self.actor_rollout_wg.infer_actor_batch(td)
        res = {"old_log_probs": tu.get(out, "log_probs").float(),
               "old_text_log_probs": tu.get(out, "text_log_probs").float()}
        if guard:
            res["old_prev_sample_mean"] = tu.get(out, "prev_sample_mean").float()
        return res

    def _update_actor_ladi(self, batch: DataProto) -> dict:
        keys = TRAIN_TENSOR_KEYS + ("old_log_probs", "advantages", "old_text_log_probs", "text_advantages",
                                    "old_prev_sample_mean")
        if self._loss_config().get("text_tis_cap", 0) > 0:
            keys = keys + ("rollout_text_log_probs",)
        td = self._actor_td(batch, keys)
        a = self.config.actor_rollout_ref.actor
        G = int(self.config.actor_rollout_ref.rollout.n)
        mini = _mini_batch_rows(len(batch), int(a.ppo_mini_batch_size) * G, train_row_divisor(self.config, self._dp_size()))
        tu.assign_non_tensor(td, global_batch_size=mini, mini_batch_size=mini, epochs=int(a.ppo_epochs),
                             seed=int(a.get("data_loader_seed", 42) or 42),
                             dataloader_kwargs={"shuffle": bool(a.get("shuffle", False))},
                             ladi_loss=self._loss_config())
        out = self.actor_rollout_wg.update_actor(td)
        return rename_dict(tu.get(out, "metrics"), "actor/")

    def _dump(self, batch: DataProto, step: int):
        d = self.config.trainer.get("rollout_data_dir", None)
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        nt = batch.non_tensor_batch
        with open(os.path.join(d, f"{step}.jsonl"), "w") as f:
            for i in range(len(batch)):
                f.write(json.dumps({"task_id": str(nt["task_id"][i]), "uid": str(nt["uid"][i]),
                                    "window_start": int(nt["window_start"][i]),
                                    "seed": int(nt["seed"][i]) if "seed" in nt else None,
                                    "rewards": batch.batch["rewards"][i].tolist(),
                                    "answers": list(nt["answer_text"][i])}) + "\n")

    def _write_metrics(self, metrics: dict):
        path = os.path.join(self.config.trainer.default_local_dir, "metrics.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = {"step": self.global_steps}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        keys = ("reward/mean", "val/pass_rate_mean", "actor/text_pg_loss", "actor/pg_clipfrac", "actor/grad_norm",
                "timing_s/step", "timing_s/gen", "timing_s/update_actor")
        print("[ladi] step %d %s" % (self.global_steps, " ".join(f"{k}={row[k]:.4g}" for k in keys if k in row)),
              flush=True)

    def _validate(self):
        """Greedy validation (deterministic latent ODE + greedy answer); reward = test pass rate."""
        t0 = time.perf_counter()
        rates, rows = [], []
        offset = 0
        for vb in self.val_dataloader:
            batch = DataProto.from_single_dict(vb)
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
            batch.non_tensor_batch["_rollout_seed_global_idx"] = np.arange(offset, offset + len(batch), dtype=np.int64)
            offset += len(batch)
            batch.meta_info.update({"validate": True, "global_steps": self.global_steps, "group_size": 1,
                                    "rollout_seed": 20251015})
            out = self.async_rollout_manager.generate_sequences(batch)
            r = out.batch["rewards"][:, 0]
            rates += r.tolist()
            nt = out.non_tensor_batch
            for i in range(len(out)):
                rows.append({"task_id": str(nt["task_id"][i]), "group": str(nt["group"][i]) if "group" in nt else "", "reward": float(r[i]),
                             "answer": nt["answer_text"][i][0]})
        d = self.config.trainer.get("validation_data_dir", None)
        if d:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"val_{self.global_steps}.jsonl"), "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        out = {"val/pass_rate_mean": float(np.mean(rates)) if rates else 0.0,
               "val/num_problems": len(rates), "timing_s/validation": time.perf_counter() - t0}
        out.update(_group_metrics(rows))
        return out

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        lad = _ladi(self.config)
        check_supported(self.config)
        settings = rollout_settings(self.config)
        print(f"[ladi] rollout settings: {settings}")
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        with open(os.path.join(self.config.trainer.default_local_dir, "rollout_settings.json"), "w") as f:
            json.dump(settings, f, indent=2)
        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        if self.config.trainer.get("val_before_train", False):
            vm = self._validate()
            print(f"[ladi] validation before training: {vm}", flush=True)
            logger.log(data=vm, step=0)
            self._write_metrics(vm)
        self.global_steps += 1
        start_epoch = (self.global_steps - 1) // max(len(self.train_dataloader), 1)
        divisor = train_row_divisor(self.config, self._dp_size())
        saved = False

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics, timing = {}, {}
                is_last = self.global_steps >= self.total_training_steps
                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))],
                                                         dtype=object)
                batch.meta_info["global_steps"] = self.global_steps
                batch.non_tensor_batch["_rollout_seed_global_idx"] = np.arange(len(batch), dtype=np.int64)
                seed = self.config.actor_rollout_ref.rollout.get("seed")
                if seed is not None:
                    batch.meta_info["rollout_seed"] = int(seed) + self.global_steps - 1
                with marked_timer("step", timing):
                    with marked_timer("gen", timing):
                        out = self.async_rollout_manager.generate_sequences(batch)
                        self.checkpoint_manager.sleep_replicas()
                    timing.update(out.meta_info.get("timing", {}))
                    out = self._trim_answers(out)
                    with marked_timer("adv", timing):
                        groups = group_index(out.non_tensor_batch["uid"])
                        rewards = out.batch["rewards"]
                        text_adv, flow_adv = hybrid_advantages(
                            rewards, groups, std_mode=lad.get("adv_std", "batch"),
                            eps=float(lad.get("adv_eps", 1e-4)), clip=float(lad.get("adv_clip", 5.0)))
                        W = out.batch["all_timesteps"].shape[1]
                        out.batch["text_advantages"] = text_adv
                        out.batch["advantages"] = flow_adv.unsqueeze(1).expand(-1, W).contiguous()
                        idx, keep = select_train_rows(rewards, groups, lad.get("filter_zero_variance_groups", True),
                                                      divisor)
                        metrics["ladi/informative_group_frac"] = float(keep.float().mean())
                        metrics["ladi/train_rows"] = len(idx)
                        self._log_rollout_metrics(out, metrics)
                        self._dump(out, self.global_steps)
                    if len(idx) == 0:
                        print(f"[ladi] step {self.global_steps}: no informative groups, skipping update")
                    else:
                        train = out.select_idxs(idx) if len(idx) < len(out) else out
                        with marked_timer("old_log_prob", timing):
                            train.batch.update(self._compute_old_log_prob_ladi(train))
                        self._log_consistency(train, metrics)
                        with marked_timer("update_actor", timing):
                            actor_metrics = self._update_actor_ladi(train)
                        metrics.update(reduce_metrics(actor_metrics))
                    saved = False
                    if self.config.trainer.save_freq > 0 and (is_last or
                                                             self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing):
                            self._save_checkpoint()
                        saved = True
                    with marked_timer("update_weights", timing):
                        self.checkpoint_manager.update_weights(self.global_steps)
                    if lad.get("log_memory", True):
                        mem = self.actor_rollout_wg.ladi_memory_stats()
                        for k in mem[0]:
                            metrics[f"mem/actor_{k}_max"] = max(m[k] for m in mem)
                tf = int(self.config.trainer.get("test_freq", -1) or -1)
                if tf > 0 and (is_last or self.global_steps % tf == 0):
                    metrics.update(self._validate())
                for k, v in timing.items():
                    metrics[k if k.startswith("ladi_rollout/") else f"timing_s/{k}"] = v
                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                logger.log(data=metrics, step=self.global_steps)
                self._write_metrics(metrics)
                if is_last:
                    pprint({k: v for k, v in metrics.items() if k.startswith(("timing_s/", "reward", "actor/"))})
                    return
                self.global_steps += 1

        if self.config.trainer.save_freq > 0 and not saved:
            self.global_steps -= 1
            self._save_checkpoint()

    @staticmethod
    def _log_rollout_metrics(out: DataProto, metrics: dict):
        r = out.batch["rewards"]
        metrics["reward/mean"] = float(r.mean())
        metrics["reward/max"] = float(r.max())
        allp = [info.get("all_passed", False) for row in out.non_tensor_batch["reward_info"] for info in row]
        metrics["reward/all_tests_passed_frac"] = float(np.mean(allp)) if allp else 0.0
        lens = out.batch["answer_mask"].sum(-1).float()
        metrics["response/len_mean"] = float(lens.mean())
        metrics["response/len_max"] = float(lens.max())
        metrics["adv/text_abs_mean"] = float(out.batch["text_advantages"].abs().mean())
        metrics["adv/flow_abs_mean"] = float(out.batch["advantages"][:, 0].abs().mean())
        metrics["flow/window_start_mean"] = float(np.mean(out.non_tensor_batch["window_start"].astype(float)))

    @staticmethod
    def _log_consistency(batch: DataProto, metrics: dict):
        """Rollout vs actor log-probs at identical weights."""
        d = (batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"]).abs()
        metrics["flow/rollout_vs_actor_logp_absdiff_mean"] = float(d.mean())
        metrics["flow/rollout_vs_actor_logp_absdiff_max"] = float(d.max())
        m = batch.batch["answer_mask"].float()
        dt = (batch.batch["old_text_log_probs"] - batch.batch["rollout_text_log_probs"]).abs() * m
        metrics["text/rollout_vs_actor_logp_absdiff_mean"] = float(dt.sum() / m.sum().clamp(min=1))
        metrics["text/rollout_vs_actor_logp_absdiff_max"] = float(dt.max())
