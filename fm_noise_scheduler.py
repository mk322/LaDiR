"""
Flow Matching Euler Discrete Scheduler implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union


class StepOutput:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample


class FlowMatchEulerDiscreteScheduler:
    """
    Flow Matching Euler Discrete Scheduler.

    Implements the forward process (add_noise) as linear interpolation
    between data x_0 and noise epsilon, and the reverse process (step)
    as an Euler integration step of the learned velocity field.
    """

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.config = {"prediction_type": "flow"}
        self.timesteps = None
        self.sigmas = None
        self._step_index = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int):
        """Set the timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps)
        self.timesteps = timesteps.long()

        # sigmas = t / T, normalized to [0, 1] range
        self.sigmas = timesteps.float() / self.num_train_timesteps
        self._step_index = None

    def add_noise(self, original_samples, noise, timesteps):
        """
        Flow matching interpolation: x_t = (1 - t/T) * x_0 + (t/T) * noise

        Args:
            original_samples: x_0, the clean data
            noise: epsilon ~ N(0, I)
            timesteps: integer timesteps in [0, num_train_timesteps)

        Returns:
            Noisy samples x_t
        """
        t = timesteps.float() / self.num_train_timesteps  # normalize to [0, 1]
        # Reshape t for broadcasting: (B,) -> (B, 1, 1) etc.
        while t.dim() < original_samples.dim():
            t = t.unsqueeze(-1)
        noisy = (1 - t) * original_samples + t * noise
        # Store the timestep for potential use by _forward
        self.timestep = timesteps
        return noisy

    def step(self, model_output, timestep, sample, generator=None):
        """
        Euler step: x_{t-dt} = x_t - dt * v_predicted

        Args:
            model_output: predicted velocity v_theta(x_t, t)
            timestep: current timestep (integer)
            sample: current noisy sample x_t

        Returns:
            StepOutput with prev_sample = x_{t-dt}
        """
        dt = 1.0 / self.num_inference_steps  # uniform step size
        prev_sample = sample - dt * model_output

        # Track step index
        if self._step_index is None:
            self._step_index = 0
        else:
            self._step_index += 1

        return StepOutput(prev_sample)

    def _init_step_index(self, timestep):
        """Initialize step index."""
        if self._step_index is None:
            self._step_index = 0
