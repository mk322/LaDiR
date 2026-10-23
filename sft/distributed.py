"""Process-group setup and FSDP2 wrapping for SFT.

FSDP2 (`torch.distributed.fsdp.fully_shard`) with
  * one FSDP unit per Qwen3DecoderLayer (+ the root LaDiRModel for embeddings, lm_head, projectors, special embeds)
  * full_shard / hybrid_shard (shard inside a node, replicate across nodes) / no_shard via a 2-D device mesh
  * fp32 sharded master weights and optimizer state, `param_dtype` compute, `reduce_dtype` gradient reduction
  * non-reentrant activation checkpointing of every decoder layer (wrapped *before* sharding)
"""

import datetime
import functools
import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, CheckpointWrapper, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setup_distributed(device=None) -> DistInfo:
    """Init the default process group (torchrun env, or a single-process group when run directly)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        timeout = datetime.timedelta(minutes=60)
        if "RANK" in os.environ:
            dist.init_process_group(backend, timeout=timeout)
        else:
            dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1,
                                    timeout=timeout)
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    if device.type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        _enable_fsdp2_on_cpu()
    return DistInfo(rank, world, local_rank, local_world, device)


def _enable_fsdp2_on_cpu():
    """Make FSDP2 usable on CPU/gloo (CPU-only runs): no-op stream events and SUM-then-divide gradient reduction."""
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives

    if getattr(collectives, "_ladir_cpu_patch", False):
        return

    class _Event:
        def synchronize(self):
            pass

        def wait(self, stream=None):
            pass

    torch.cpu.Stream.record_event = lambda self, event=None: _Event()
    torch.cpu.Stream.wait_event = lambda self, event: None
    default_factors = collectives._get_gradient_divide_factors

    def divide_factors(reduce_scatter_group, all_reduce_group, reduce_dtype):
        if dist.get_backend(reduce_scatter_group) != "gloo":
            return default_factors(reduce_scatter_group, all_reduce_group, reduce_dtype)
        n = reduce_scatter_group.size() * (all_reduce_group.size() if all_reduce_group is not None else 1)
        return 1.0, float(n)

    collectives._get_gradient_divide_factors = divide_factors
    collectives._ladir_cpu_patch = True


def make_mesh(info: DistInfo, sharding: str):
    """2-D (replicate, shard) mesh. full_shard: (1, W); hybrid_shard: (W / local, local); no_shard: (W, 1)."""
    W = info.world_size
    if sharding == "full_shard":
        shape = (1, W)
    elif sharding == "hybrid_shard":
        if W % info.local_world_size:
            raise ValueError(f"world size {W} not divisible by local world size {info.local_world_size}")
        shape = (W // info.local_world_size, info.local_world_size)
    elif sharding == "no_shard":
        shape = (W, 1)
    else:
        raise ValueError(sharding)
    return init_device_mesh(info.device.type, shape, mesh_dim_names=("replicate", "shard"))


def shard_model(model: torch.nn.Module, layer_cls, mesh, param_dtype: str, reduce_dtype: str,
                activation_checkpointing: bool):
    """Wrap in place: AC(layer) -> fully_shard(layer) for every `layer_cls`, then fully_shard(root)."""
    if activation_checkpointing:
        wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=lambda m: isinstance(m, layer_cls))
        units = [m for m in model.modules() if isinstance(m, CheckpointWrapper)]
    else:
        units = [m for m in model.modules() if isinstance(m, layer_cls)]
    if not units:
        raise RuntimeError(f"no {layer_cls.__name__} modules found to shard")
    mp = MixedPrecisionPolicy(param_dtype=DTYPES[param_dtype], reduce_dtype=DTYPES[reduce_dtype])
    for unit in units:
        fully_shard(unit, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def clean_fqn(name: str) -> str:
    """Strip wrapper prefixes that activation checkpointing inserts into parameter names."""
    return name.replace("_checkpoint_wrapped_module.", "")
