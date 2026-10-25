"""FSDP engine for the LaDi-RL objective: latent-window diffusion loss + token-level PPO on the answers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
import torch.distributed
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.metric import AggregationType, Metric
from verl.utils.py_functional import append_to_dict
from verl.workers.engine.utils import prepare_micro_batches

from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine

from .ladi_model import is_backbone_param
from .losses import HybridLossConfig, text_ppo_terms, tis_weights


def _metric(v):
    return Metric(value=float(v), aggregation=AggregationType.MEAN)


def _device():
    d = get_device_id()
    if isinstance(d, str) and not d.isdigit():
        return torch.device(d)
    return torch.device(f"cuda:{int(d)}") if torch.cuda.is_available() else torch.device("cpu")


def _batchify(x: torch.Tensor, B: int) -> torch.Tensor:
    if x.dim() == 0:
        return x.reshape(1).expand(B).clone()
    if x.shape[0] != B:
        return x.reshape(1, *x.shape).expand(B, *x.shape).clone()
    return x


class LaDiHybridFSDPEngine(PPODiffusersFSDPEngine):
    def _build_optimizer(self, module):
        """Param groups: backbone (optim.lr), flow head (ladi.flow_head_lr), other heads (optim.lr * head_lr_mult)."""
        from verl.workers.config.optimizer import build_optimizer

        from .ladi_model import FLOW_HEAD

        mult = float(getattr(self, "_ladi_head_lr_mult", 1.0) or 1.0)
        flow_lr = getattr(self, "_ladi_flow_head_lr", None)
        lr = self.optimizer_config.lr
        if mult == 1.0 and flow_lr is None:
            return build_optimizer([p for p in module.parameters() if p.requires_grad], self.optimizer_config)
        groups = {"backbone": [], "flow_head": [], "heads": []}
        for name, p in module.named_parameters():
            if p.requires_grad:
                key = "backbone" if is_backbone_param(name) else ("flow_head" if name.startswith(FLOW_HEAD) else "heads")
                groups[key].append(p)
        lrs = {"backbone": lr, "heads": lr * mult,
               "flow_head": float(flow_lr) if flow_lr is not None else lr * mult}
        pg = [{"params": ps, "lr": lrs[k], "name": k} for k, ps in groups.items() if ps]
        return build_optimizer(pg, self.optimizer_config)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, adapter_name=None, **kwargs):
        """Per-tensor gather that also works with the FSDP2 CPU offload policy."""
        if not getattr(self.engine_config, "offload_policy", False):
            return super().get_per_tensor_param(layered_summon=layered_summon, base_sync_done=base_sync_done,
                                                adapter_name=adapter_name, **kwargs)
        from torch.distributed.tensor import DTensor

        dev = _device()
        module = self.module

        def gen():
            for name, p in module.named_parameters():
                if isinstance(p, DTensor):
                    full = DTensor.from_local(p.to_local().to(dev, non_blocking=True), p.device_mesh,
                                              p.placements, shape=p.shape, stride=p.stride()).full_tensor()
                else:
                    full = p.detach().to(dev)
                yield f"transformer.{name}", full.to(torch.bfloat16)

        return gen(), None

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        """With the FSDP2 CPU offload policy, save the shards where they are."""
        if not getattr(self.engine_config, "offload_policy", False):
            return super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep, **kwargs)
        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step,
                                                max_ckpt_to_keep=max_ckpt_to_keep)
        torch.distributed.barrier()

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        if not getattr(self.engine_config, "offload_policy", False):
            out = super().load_checkpoint(local_path, hdfs_path, del_local_after_load, **kwargs)
        else:
            out = self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path,
                                                          del_local_after_load=del_local_after_load)
            torch.distributed.barrier()
        self._apply_config_lrs()
        return out

    def _config_lrs(self):
        mult = float(getattr(self, "_ladi_head_lr_mult", 1.0) or 1.0)
        flow_lr = getattr(self, "_ladi_flow_head_lr", None)
        lr = self.optimizer_config.lr
        return {"backbone": lr, "heads": lr * mult,
                "flow_head": float(flow_lr) if flow_lr is not None else lr * mult}

    def _apply_config_lrs(self):
        """On resume, take learning rates from the config rather than the restored optimizer state."""
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return
        lrs = self._config_lrs()
        new = [lrs.get(g.get("name"), self.optimizer_config.lr) for g in opt.param_groups]
        sched = getattr(self, "lr_scheduler", None)
        for g, v in zip(opt.param_groups, new):
            scale = g["lr"] / g["initial_lr"] if g.get("initial_lr") else 1.0   # keep the schedule's position
            g["initial_lr"] = v
            g["lr"] = v * scale
        if sched is not None and hasattr(sched, "base_lrs"):
            sched.base_lrs = list(new)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only: bool = False):
        hl = HybridLossConfig.from_dict(tu.get_non_tensor_data(data, "ladi_loss", default=None))
        want_velocity = bool(tu.get_non_tensor_data(data, "return_velocity", default=False))
        want_mean = bool(tu.get_non_tensor_data(data, "return_prev_sample_mean", default=True))
        W = int(data["all_timesteps"].shape[1])
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size, use_dynamic_bsz=False)
        micro_batches, _ = prepare_micro_batches(data=data, dp_group=self.get_data_parallel_group(),
                                                 same_micro_num_in_dp=True)
        n_micro = len(micro_batches)
        dp_group = self.get_data_parallel_group()
        dp_size = torch.distributed.get_world_size(dp_group) if torch.distributed.is_initialized() else 1
        has_text = "answer_ids" in data.keys()

        total_tokens = None
        if not forward_only and has_text and hl.train_text:
            local = sum(float(mb["answer_mask"].sum()) for mb in micro_batches)
            t = torch.tensor([local], device=get_device_id(), dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(t, group=dp_group)
            total_tokens = max(float(t.item()), 1.0)

        outputs = {k: [] for k in ("log_probs", "prev_sample_mean", "std_dev_t", "sqrt_dt", "text_log_probs",
                                   "velocity")}
        losses, metrics = [], {}
        do_flow = forward_only or hl.train_flow
        ctx = torch.no_grad() if forward_only else nullcontext()
        flow_coef = self._effective_flow_coef(hl) if not forward_only else hl.flow_coef
        # flow part of every micro-batch first, then the text part, so each loss's backbone gradient can be measured
        text_mbs = []
        for mb in micro_batches:
            mb = mb.to(get_device_id())
            B = mb.batch_size[0]
            tu.assign_non_tensor(mb, gradient_accumulation_steps=n_micro * W)
            step_out = {k: [] for k in ("log_probs", "prev_sample_mean", "std_dev_t", "sqrt_dt", "velocity")}
            with ctx:
                fn = None if forward_only else loss_function
                if do_flow and hl.fuse_window_steps and W > 1:
                    loss, metas = self._flow_steps_fused(mb, fn, forward_only, W, want_velocity, hl)
                    if not forward_only:
                        (flow_coef * loss).backward()
                else:
                    metas = None
                for step in range(W if do_flow else 0):
                    if metas is not None:
                        meta = metas[step]
                    else:
                        loss, meta = self._flow_step(mb, fn, forward_only, step, want_velocity, hl)
                        if not forward_only:
                            (flow_coef * loss).backward()
                    for k in step_out:
                        if k == "prev_sample_mean" and not want_mean:
                            continue
                        x = meta["model_output"].get(k)
                        if x is not None:
                            step_out[k].append(_batchify(x.detach(), B))
                    losses.append(flow_coef * meta["loss"])
                    append_to_dict(metrics, meta["metrics"])
                for k, v in step_out.items():
                    if v:
                        outputs[k].append(torch.stack(v, dim=1))
                if has_text and forward_only:
                    outputs["text_log_probs"].append(
                        self._text_slots(mb, hl, forward_only, total_tokens, dp_size, losses, metrics))
            if has_text and not forward_only:
                text_mbs.append(mb)
        if not forward_only:
            g_flow = self._grad_sq_norms()
            for mb in text_mbs:
                outputs["text_log_probs"].append(
                    self._text_slots(mb, hl, forward_only, total_tokens, dp_size, losses, metrics))
            g_all = self._grad_sq_norms()
            gm = {
                "grad/backbone_flow_norm": g_flow["backbone"] ** 0.5,
                "grad/backbone_flow_norm_unscaled": g_flow["backbone"] ** 0.5 / max(flow_coef, 1e-12),
                "grad/backbone_text_norm_est": max(g_all["backbone"] - g_flow["backbone"], 0.0) ** 0.5,
                "grad/backbone_total_norm": g_all["backbone"] ** 0.5,
                "grad/flow_head_norm": g_flow["flow_head"] ** 0.5,
                "grad/text_head_norm": max(g_all["text_head"] - g_flow["text_head"], 0.0) ** 0.5,
                "loss/flow_coef": flow_coef,
            }
            self._last_grad_norms = gm
            append_to_dict(metrics, {k: _metric(v) for k, v in gm.items()})
        model_output = {k: torch.cat(v, 0) for k, v in outputs.items() if v}
        return {"model_output": model_output, "loss": losses, "metrics": metrics}

    def _effective_flow_coef(self, hl: HybridLossConfig) -> float:
        """grad_ratio mode: scale the flow loss so its backbone gradient is flow_grad_ratio x the text one."""
        if hl.flow_coef_mode == "fixed":
            return hl.flow_coef
        last = getattr(self, "_last_grad_norms", None)
        if not last or last["grad/backbone_flow_norm_unscaled"] <= 0 or last["grad/backbone_text_norm_est"] <= 0:
            return hl.flow_coef
        lam = hl.flow_grad_ratio * last["grad/backbone_text_norm_est"] / last["grad/backbone_flow_norm_unscaled"]
        return float(min(max(lam, hl.flow_coef_min), hl.flow_coef_max))

    def _grad_sq_norms(self) -> dict:
        """Squared gradient norms of backbone / flow head / text head (all-reduced)."""
        from torch.distributed.tensor import DTensor

        from .ladi_model import FLOW_HEAD, TEXT_HEAD
        dev = _device()
        acc = torch.zeros(3, dtype=torch.float64, device=dev)
        for name, p in self.module.named_parameters():
            g = p.grad
            if g is None:
                continue
            g = g.to_local() if isinstance(g, DTensor) else g
            sq = g.detach().double().pow(2).sum().to(dev)
            idx = 0 if name.startswith("backbone_llm.") else (1 if name.startswith(FLOW_HEAD) else
                                                                (2 if name.startswith(TEXT_HEAD) else None))
            if idx is not None:
                acc[idx] += sq
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(acc, group=self.get_data_parallel_group())
        return {"backbone": float(acc[0]), "flow_head": float(acc[1]), "text_head": float(acc[2])}

    def _flow_steps_fused(self, mb, loss_function, forward_only, W, want_velocity, hl: HybridLossConfig):
        """All window steps of a micro-batch in one batched forward."""
        lat, ts = mb["all_latents"], mb["all_timesteps"]
        if str(self.model_config.algorithm) == "grpo_guard":
            for step in range(W):
                if not bool((ts[:, step] == ts[0, step]).all()):
                    raise ValueError("grpo_guard needs one SDE window per micro-batch (set ladi.window_scope=step)")
        per_step = [self.prepare_model_inputs(micro_batch=mb, step=k)[0] for k in range(W)]
        inputs = dict(per_step[0])
        for key in ("prompt_ids", "prompt_mask", "latents", "labels"):
            inputs[key] = torch.cat([p[key] for p in per_step], 0)
        v_all = self.module(**inputs)[0]
        B = lat.shape[0]
        total, metas = None, []
        for k in range(W):
            v = v_all[k * B:(k + 1) * B]
            _, lp, mean, std, sqrt_dt = self.scheduler.sample_previous_step(
                sample=lat[:, k].float(), model_output=v.float(), timestep=ts[:, k],
                noise_level=self.model_config.algo.noise_level, prev_sample=lat[:, k + 1].float(),
                sde_type=self.model_config.algo.sde_type, return_logprobs=True, return_sqrt_dt=True,
                include_logprob_normalizer=False)
            mo = {"log_probs": lp, "prev_sample_mean": mean, "std_dev_t": std, "sqrt_dt": sqrt_dt}
            if want_velocity:
                mo["velocity"] = v.float()
            if forward_only or loss_function is None:
                metas.append({"model_output": mo, "loss": 0.0, "metrics": {}})
                continue
            data = tu.get_tensordict({"old_log_probs": mb["old_log_probs"][:, k], "advantages": mb["advantages"][:, k]})
            for key in ("old_prev_sample_mean", "ref_log_prob", "ref_prev_sample_mean", "teacher_prev_sample_mean",
                        "rollout_is_weights"):
                if mb.get(key, None) is not None:
                    data[key] = mb[key][:, k]
            ga = tu.get_non_tensor_data(mb, "gradient_accumulation_steps", default=1)
            tu.assign_non_tensor(data, gradient_accumulation_steps=ga, sp_size=1)
            loss, mets = loss_function(model_output=mo, data=data, dp_group=self.get_data_parallel_group())
            total = loss if total is None else total + loss
            metas.append({"model_output": mo, "loss": float(loss.detach()), "metrics": mets})
        return total, metas

    def _flow_step(self, mb, loss_function, forward_only, step, want_velocity, hl: HybridLossConfig):
        """One window step: flow forward -> scheduler log-prob -> registered diffusion loss."""
        lat, ts = mb["all_latents"], mb["all_timesteps"]
        if str(self.model_config.algorithm) == "grpo_guard" and not bool((ts[:, step] == ts[0, step]).all()):
            raise ValueError("grpo_guard needs one SDE window per micro-batch (set ladi.window_scope=step)")
        model_inputs, _ = self.prepare_model_inputs(micro_batch=mb, step=step)
        v = self.module(**model_inputs)[0]
        _, lp, mean, std, sqrt_dt = self.scheduler.sample_previous_step(
            sample=lat[:, step].float(), model_output=v.float(), timestep=ts[:, step],
            noise_level=self.model_config.algo.noise_level, prev_sample=lat[:, step + 1].float(),
            sde_type=self.model_config.algo.sde_type, return_logprobs=True, return_sqrt_dt=True,
            include_logprob_normalizer=False)
        model_output = {"log_probs": lp, "prev_sample_mean": mean, "std_dev_t": std, "sqrt_dt": sqrt_dt}
        if want_velocity:
            model_output["velocity"] = v.float()
        if forward_only or loss_function is None:
            return None, {"model_output": model_output, "loss": 0.0, "metrics": {}}
        data = tu.get_tensordict({"old_log_probs": mb["old_log_probs"][:, step],
                                  "advantages": mb["advantages"][:, step]})
        for k in ("old_prev_sample_mean", "ref_log_prob", "ref_prev_sample_mean", "teacher_prev_sample_mean",
                  "rollout_is_weights"):
            if mb.get(k, None) is not None:
                data[k] = mb[k][:, step]
        ga = tu.get_non_tensor_data(mb, "gradient_accumulation_steps", default=1)
        tu.assign_non_tensor(data, gradient_accumulation_steps=ga, sp_size=1)
        loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
        return loss, {"model_output": model_output, "loss": float(loss.detach()), "metrics": metrics}

    def _text_slots(self, mb, hl: HybridLossConfig, forward_only, total_tokens, dp_size, losses, metrics):
        a_ids, a_mask = mb["answer_ids"], mb["answer_mask"]          # [b, M, L]
        M = a_ids.shape[1]
        lps = []
        for m in range(M):
            ids_m, mask_m = a_ids[:, m], a_mask[:, m]
            if not forward_only and not hl.train_text:
                lps.append(torch.zeros_like(ids_m, dtype=torch.float32))
                continue
            lp = self.module(mode="text", prompt_ids=mb["prompt_ids"], prompt_mask=mb["prompt_mask"],
                             latents=mb["final_latents"], answer_ids=ids_m, answer_mask=mask_m,
                             temperature=hl.temperature)
            lps.append(lp.detach().float())
            if forward_only:
                continue
            isw = None
            if hl.text_tis_cap > 0 and mb.get("rollout_text_log_probs", None) is not None:
                isw = tis_weights(mb["old_text_log_probs"][:, m], mb["rollout_text_log_probs"][:, m],
                                  hl.text_tis_cap)
            loss_sum, st = text_ppo_terms(lp, mb["old_text_log_probs"][:, m], mb["text_advantages"][:, m], mask_m,
                                          hl.text_clip_low, hl.text_clip_high, hl.text_adv_clip, is_weights=isw,
                                          clip_ratio_c=hl.text_clip_ratio_c)
            # FSDP averages over DP ranks; scale to a global token mean
            loss = hl.text_coef * loss_sum * dp_size / total_tokens
            loss.backward()
            losses.append(float(loss.detach()))
            n = max(float(st["tokens"]), 1.0)
            append_to_dict(metrics, {
                "actor/text_pg_loss": _metric(float(loss_sum.detach()) / n),
                "actor/text_clipfrac": _metric(float(st["clip_sum"]) / n),
                "actor/text_clipfrac_higher": _metric(float(st["clip_high_sum"]) / n),
                "actor/text_approx_kl": _metric(float(st["kl_sum"]) / n),
                "actor/text_ratio_mean": _metric(float(st["ratio_sum"]) / n),
                "actor/text_is_weight_mean": _metric(float(st["is_weight_sum"]) / n),
                "actor/text_dual_clipfrac": _metric(float(st["dual_clip_sum"]) / n),
                "actor/text_tokens": Metric(value=float(st["tokens"]), aggregation=AggregationType.SUM),
            })
        return torch.stack(lps, 1)
