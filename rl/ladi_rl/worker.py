"""Actor/rollout worker: verl-omni's ActorRolloutRefWorker with the LaDi engine, AR weight sync and LaDiR export."""

from __future__ import annotations

import contextlib
import json
import logging
import os

import torch
import torch.distributed
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine.base import EngineRegistry

from verl_omni.workers.engine_workers import ActorRolloutRefWorker

from .engine import LaDiHybridFSDPEngine

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def scoped_engine(model_type: str, engine_cls):
    """Temporarily route `model_type` to `engine_cls` in verl's engine registry."""
    table = EngineRegistry._engines[model_type]
    saved = {b: dict(v) for b, v in table.items()}
    try:
        for backend, by_dev in table.items():
            for key in list(by_dev):
                by_dev[key] = engine_cls
        yield
    finally:
        for backend in list(table):
            table[backend] = saved[backend]


def write_ladi_export(module, state: dict, out_dir: str, fmt: str = "split"):
    """split: <out>/backbone-hf/ (bf16 Qwen3) + ladi_heads.safetensors (fp32); single: one fp32 model.safetensors."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    cfg = getattr(module, "config", None)
    if fmt == "single":
        save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    elif fmt == "split":
        bb = {k[len("backbone_llm."):]: v.to(torch.bfloat16) for k, v in state.items() if k.startswith("backbone_llm.")}
        heads = {k: v for k, v in state.items() if not k.startswith("backbone_llm.")}
        hf_dir = os.path.join(out_dir, "backbone-hf")
        os.makedirs(hf_dir, exist_ok=True)
        save_file(bb, os.path.join(hf_dir, "model.safetensors"), metadata={"format": "pt"})
        bcfg = module.backbone_llm.config
        bcfg.use_cache = True
        bcfg.save_pretrained(hf_dir)
        module.tokenizer.save_pretrained(hf_dir)
        save_file(heads, os.path.join(out_dir, "ladi_heads.safetensors"), metadata={"format": "pt"})
    else:
        raise ValueError(f"unknown export_format {fmt!r}")
    if cfg is not None:
        with open(os.path.join(out_dir, "ladi_config.json"), "w") as f:
            json.dump(dict(cfg.__dict__), f, indent=2, default=str)


class LaDiActorRolloutRefWorker(ActorRolloutRefWorker):
    def _ladi(self) -> dict:
        from omegaconf import OmegaConf
        lad = self.config.get("ladi", None)
        return OmegaConf.to_container(lad, resolve=True) if lad is not None else {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        lad = self._ladi()
        engine_cls = type("LaDiHybridFSDPEngineConfigured", (LaDiHybridFSDPEngine,),
                          {"_ladi_head_lr_mult": lad.get("head_lr_mult", 1.0),
                           "_ladi_flow_head_lr": lad.get("flow_head_lr", None)})
        from .training_adapter import LaDiDiffusionBase
        LaDiDiffusionBase._ladi_trainable = lad.get("trainable", "all")
        with scoped_engine("diffusion_model", engine_cls):
            super().init_model()
        self.ar_rollout = None
        if lad.get("ar_backend", "vllm") == "vllm" and "rollout" in self.role:
            from omegaconf import OmegaConf
            from verl.utils.config import omega_conf_to_dataclass
            from verl.workers.rollout.base import get_rollout_class

            from .vllm_ar import ROLLOUT_NAME
            from .vllm_ar import register as register_ar_backend
            register_ar_backend()
            self.ar_rollout = get_rollout_class(ROLLOUT_NAME, "async")(
                config=omega_conf_to_dataclass(OmegaConf.create(lad["_ar_rollout"])),
                model_config=omega_conf_to_dataclass(OmegaConf.create(lad["_ar_model"])), device_mesh=None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Sync the stage-0 rollout, then the backbone into the stage-1 vLLM replica."""
        from verl.utils.memory_utils import aggressive_empty_cache
        aggressive_empty_cache(force_sync=True)  # release the update's cached blocks before the rollout wakes
        await super().update_weights(global_steps=global_steps, mode=mode)
        if getattr(self, "ar_rollout", None) is not None:
            await self._update_ar_weights(global_steps)

    async def _update_ar_weights(self, global_steps):
        from verl.utils.device import set_expandable_segments
        from verl.utils.memory_utils import aggressive_empty_cache

        from .vllm_ar import backbone_weight_name, backbone_weights

        expected = {backbone_weight_name(n) for n, _ in self.actor.engine.module.named_parameters()} - {None}
        sent = set()

        def counted(gen):
            for name, t in gen:
                sent.add(name)
                yield name, t

        set_expandable_segments(False)
        await self.ar_rollout.resume(tags=["weights"])
        params, _ = self.actor.engine.get_per_tensor_param()
        await self.ar_rollout.update_weights(counted(backbone_weights(params)), global_steps=global_steps)
        if sent != expected:
            raise RuntimeError(f"AR weight sync mismatch: missing={sorted(expected - sent)[:5]} "
                               f"unexpected={sorted(sent - expected)[:5]}")
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)
        await self.ar_rollout.resume(tags=["kv_cache"])
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def ladi_memory_stats(self, reset: bool = True):
        """Peak torch memory since the last call and current device usage (GB)."""
        if not torch.cuda.is_available():
            return {"max_allocated_gb": 0.0, "max_reserved_gb": 0.0, "device_used_gb": 0.0}
        free, total = torch.cuda.mem_get_info()
        out = {"max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
               "max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
               "device_used_gb": (total - free) / 2**30}
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return out

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        lad = self._ladi()
        if lad.get("save_fsdp_checkpoint", True):
            super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        if lad.get("export_ladi_checkpoint", True):
            self._export_ladi(os.path.join(local_path, "ladi"))

    def _export_ladi(self, out_dir: str):
        """Gather the FSDP2 state tensor by tensor and write it on rank 0."""
        from torch.distributed.tensor import DTensor
        from verl.utils.device import get_device_id

        module = self.actor.engine.module
        dev = torch.device(f"cuda:{get_device_id()}")
        rank0 = torch.distributed.get_rank() == 0
        state = {}
        for name, t in list(module.state_dict(keep_vars=True).items()):
            t = t.detach() if hasattr(t, "detach") else t
            if isinstance(t, DTensor):
                full = DTensor.from_local(t.to_local().to(dev), t.device_mesh, t.placements, shape=t.shape,
                                          stride=t.stride()).full_tensor()
            else:
                full = t
            if rank0:
                state[name] = full.to(torch.float32).cpu().contiguous()
            del full
        if rank0:
            write_ladi_export(module, state, out_dir, self._ladi().get("export_format", "split"))
            logger.warning("exported LaDi checkpoint (%d tensors) to %s", len(state), out_dir)
        del state
        torch.distributed.barrier()
