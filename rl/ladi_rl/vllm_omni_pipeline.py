"""vllm-omni rollout pipeline for LaDi (custom pipeline; weights arrive as `transformer.<key>` from the actor sync)."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase

from .common import ALGORITHMS, ARCHITECTURE, build_scheduler
from .rollout import DecodeConfig
from .stage import LaDiStageRunner, StageRequest
from .training_adapter import flow_config_from_sampling

logger = logging.getLogger(__name__)

WEIGHT_PREFIX = "transformer."


def _local_device():
    try:
        from vllm_omni.diffusion.distributed.utils import get_local_device
        return get_local_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get(d, k, default=None):
    v = d.get(k, default) if isinstance(d, dict) else default
    return default if v is None else v


class LaDiPipelineWithLogProb(nn.Module):
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig = None, prefix: str = "", model=None, device=None):
        super().__init__()
        self.od_config = od_config
        self.device = torch.device(device) if device is not None else _local_device()
        if model is None:
            from .ladi_model import LaDiForTraining
            model = LaDiForTraining.from_pretrained(od_config.model, torch_dtype=torch.bfloat16, device="cpu")
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._param_names = {n for n, _ in self.model.named_parameters()} | {n for n, _ in self.model.named_buffers()}
        self.runner = LaDiStageRunner(self.model, lambda s: build_scheduler(s, self.device), self.device)
        self._sync_counts = {}
        self._expected = set(self.model.state_dict().keys())
        self._received = set()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Optional[set]:
        """Copy actor tensors into the rollout model; an incomplete sync raises at the next request."""
        state = self.model.state_dict()
        n = 0
        for name, tensor in weights:
            key = name[len(WEIGHT_PREFIX):] if name.startswith(WEIGHT_PREFIX) else name
            if key not in state:
                raise KeyError(f"LaDi rollout: unexpected weight {name!r}")
            with torch.no_grad():
                state[key].copy_(tensor.to(device=state[key].device, dtype=state[key].dtype))
            n += 1
            self._received.add(key)
        if self._received >= self._expected:
            self._sync_counts["complete_syncs"] = self._sync_counts.get("complete_syncs", 0) + 1
            self._received = set()
        self._sync_counts["last_bucket"] = n
        self._sync_counts["total"] = self._sync_counts.get("total", 0) + n
        return None

    def _to_stage_request(self, req: OmniDiffusionRequest) -> Optional[StageRequest]:
        prompt = req.prompt
        ids = prompt.get("prompt_token_ids") if isinstance(prompt, dict) else None
        if ids is None:
            text = prompt.get("prompt") if isinstance(prompt, dict) else prompt
            if not text:
                return None
            ids = self.model.tokenizer(text, add_special_tokens=False)["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        sp = req.sampling_params
        ex = dict(sp.extra_args or {})
        if not ex.get("ladi_request", False):
            return None  # engine warm-up request
        flat = {**ex, "num_inference_steps": sp.num_inference_steps,
                "guidance_scale": sp.guidance_scale if getattr(sp, "guidance_scale_provided", True) else None}
        flow = flow_config_from_sampling(flat).validate()
        dc = DecodeConfig(
            max_new_tokens=int(_get(ex, "max_new_tokens", 768)),
            temperature=float(_get(ex, "temperature", 1.0)),
            top_p=float(_get(ex, "top_p", 1.0)),
            stop_at_code_block=bool(_get(ex, "stop_at_code_block", True)),
            answers_per_latent=int(_get(ex, "answers_per_latent", 1)),
            batch_size=int(_get(ex, "decode_batch_size", 64)),
        )
        ws = ex.get("sde_window_start")
        return StageRequest(
            prompt_ids=[int(i) for i in ids], group_size=max(int(sp.num_outputs_per_prompt or 1), 1), flow=flow,
            decode=dc, window_start=None if ws is None else int(ws), seed=sp.seed,
            open_fence=int(_get(ex, "open_fence", 0)), global_step=int(_get(ex, "global_steps", 0)),
            return_prefix_embeds=bool(_get(ex, "return_prefix_embeds", False)),
            decode_answers=bool(_get(ex, "decode_answers", True)),
            latent_init=str(_get(ex, "latent_init", "independent")))

    def check_weight_sync(self):
        if self._received:
            missing = sorted(self._expected - self._received)
            raise RuntimeError(f"LaDi rollout: incomplete weight sync, {len(missing)} tensors missing, e.g. "
                               f"{missing[:5]}")

    @torch.no_grad()
    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch, **kwargs):
        batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        stage_reqs = [self._to_stage_request(r) for r in batch.requests]
        live = [i for i, r in enumerate(stage_reqs) if r is not None]
        if batch.is_dummy_run() or not live:
            outs = [DiffusionOutput(output=None, custom_output={}) for _ in batch.requests]
            return outs if isinstance(req, DiffusionRequestBatch) else outs[0]
        self.check_weight_sync()
        results = self.runner.run([stage_reqs[i] for i in live])
        outs = [DiffusionOutput(output=None, custom_output={}) for _ in batch.requests]
        for i, res in zip(live, results):
            outs[i] = DiffusionOutput(output=res["window"].float(), custom_output=res)
        return outs if isinstance(req, DiffusionRequestBatch) else outs[0]


for _alg in ALGORITHMS:
    VllmOmniPipelineBase.register(ARCHITECTURE, algorithm=_alg)(LaDiPipelineWithLogProb)
