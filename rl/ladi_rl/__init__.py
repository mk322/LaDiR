"""LaDi-RL on verl-omni. Importing the package registers the LaDiR training adapter and rollout pipeline."""

from .common import ALGORITHMS, ARCHITECTURE, FlowRLConfig  # noqa: F401
from .training_adapter import LaDiDiffusion  # noqa: F401

try:
    from .vllm_omni_pipeline import LaDiPipelineWithLogProb  # noqa: F401
except ImportError as e:  # pragma: no cover
    import logging

    logging.getLogger(__name__).warning("LaDi vllm-omni pipeline not registered: %s", e)

__all__ = ["ARCHITECTURE", "ALGORITHMS", "FlowRLConfig", "LaDiDiffusion", "LaDiPipelineWithLogProb"]
