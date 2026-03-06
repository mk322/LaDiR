"""
GRPO (Group Relative Policy Optimization) trainer for latent diffusion.

Treats the T-step denoising process as a multi-step MDP:
- State: (x_t, t, question)
- Action: velocity prediction v_theta(x_t, t, q)
- Reward: exact-match correctness after VAE decode of final x_0

Uses clipped surrogate objective (PPO-style) with group-relative advantages.
Supports FSDP: all forward passes go through the FSDP wrapper via model(mode="velocity").
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler


class GRPODiffusionTrainer:
    def __init__(self, model, ref_model, autoencoder, reward_fn, tokenizer, optimizer, cfg):
        """
        Args:
            model: trainable LMFusionModel (FSDP-wrapped if distributed)
            ref_model: frozen LMFusionModel (FSDP-wrapped if distributed)
            autoencoder: frozen VAE for encoding/decoding (NOT FSDP-wrapped)
            reward_fn: ExactMatchReward instance
            tokenizer: text tokenizer
            optimizer: optimizer created AFTER FSDP wrapping
            cfg: OmegaConf config with grpo.* fields
        """
        self.model = model
        self.ref_model = ref_model
        self.autoencoder = autoencoder
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.cfg = cfg

        grpo_cfg = cfg.grpo
        self.group_size = grpo_cfg.group_size
        self.num_denoise_steps = grpo_cfg.num_denoise_steps
        self.trajectory_subsample = grpo_cfg.trajectory_subsample
        self.clip_epsilon = grpo_cfg.clip_epsilon
        self.kl_coeff = grpo_cfg.kl_coeff
        self.policy_variance = grpo_cfg.policy_variance
        self.num_ppo_epochs = grpo_cfg.num_ppo_epochs
        self.max_grad_norm = grpo_cfg.max_grad_norm
        self.tht_token_dim = 128

        self.is_distributed = dist.is_initialized()

        # Scheduler for trajectory sampling (same config as model's sample_scheduler)
        self.sample_scheduler = FlowMatchEulerDiscreteScheduler(1000)

    def _call_velocity(self, model, input_ids_q, x_t, timestep):
        """
        Compute velocity by calling through the FSDP wrapper.
        This ensures all params (including root-level ae_to_latent, output_projection,
        time_embed, etc.) are properly all-gathered before use.
        """
        return model(
            input_ids_q=input_ids_q,
            mode="velocity",
            grpo_x_t=x_t,
            grpo_timestep=timestep,
        )

    def sample_trajectory(self, input_ids_q, thought_shape, num_steps, subsample_every):
        """
        Run the denoising loop through the FSDP-wrapped model, recording
        states/actions at subsampled steps.

        All forward passes go through self.model(..., mode="velocity") so that
        FSDP properly all-gathers root-level parameters.
        """
        with torch.no_grad():
            B = input_ids_q.shape[0]
            device = input_ids_q.device
            x = torch.randn(B, *thought_shape, dtype=torch.bfloat16, device=device)

            self.sample_scheduler._step_index = None
            self.sample_scheduler.set_timesteps(num_inference_steps=num_steps)

            states, actions, timesteps_recorded = [], [], []

            for step_idx, t in enumerate(self.sample_scheduler.timesteps):
                record = (step_idx % subsample_every == 0)
                if record:
                    states.append(x.clone())
                    timesteps_recorded.append(t.clone())

                timestep = torch.full((B,), t, device=device)
                v_pred = self._call_velocity(self.model, input_ids_q, x, timestep)

                if record:
                    actions.append(v_pred.clone())

                x = self.sample_scheduler.step(v_pred, t, x).prev_sample

        return {
            "states": states,
            "actions": actions,
            "timesteps": timesteps_recorded,
            "x_0": x,
        }

    def sample_and_evaluate(self, batch):
        """
        Sample G trajectories per question, compute rewards.

        Args:
            batch: dict with 'input_ids_q' (B, T) and 'gt_solutions' (list of B strings)

        Returns:
            trajectory: dict with states, actions, timesteps, x_0
            rewards: (B, G) tensor of rewards
            input_ids_q_expanded: (B*G, T) expanded question IDs
        """
        input_ids_q = batch["input_ids_q"]
        gt_solutions = batch["gt_solutions"]
        B = input_ids_q.shape[0]
        G = self.group_size

        # Repeat each question G times: [q1,q1,...,q2,q2,...]
        input_ids_q_expanded = input_ids_q.repeat_interleave(G, dim=0)  # (B*G, T)

        # Sample trajectories through FSDP wrapper
        trajectory = self.sample_trajectory(
            input_ids_q_expanded,
            thought_shape=(3, self.tht_token_dim),
            num_steps=self.num_denoise_steps,
            subsample_every=self.trajectory_subsample,
        )

        # Decode final latents via VAE (not FSDP-wrapped, runs locally)
        decoded_texts = self.autoencoder.decode_text_batch(trajectory["x_0"])

        # Expand gt_solutions to match B*G
        gt_solutions_expanded = [s for s in gt_solutions for _ in range(G)]

        # Compute rewards
        rewards = self.reward_fn.compute_rewards(decoded_texts, gt_solutions_expanded)
        rewards = rewards.to(input_ids_q.device).view(B, G)

        return trajectory, rewards, input_ids_q_expanded, decoded_texts

    def compute_advantages(self, rewards):
        """
        Group-relative advantage normalization.

        Args:
            rewards: (B, G) tensor

        Returns:
            advantages: (B, G) tensor with mean ~0 per group
        """
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - mean_r) / (std_r + 1e-8)
        return advantages

    def compute_grpo_loss(self, trajectory, advantages, input_ids_q):
        """
        Compute GRPO clipped surrogate loss + KL penalty.
        All velocity calls go through the FSDP wrapper.

        Args:
            trajectory: dict with states, actions, timesteps from sampling
            advantages: (B, G) advantages, will be flattened to (B*G,)
            input_ids_q: (B*G, T) expanded question IDs

        Returns:
            total loss scalar
        """
        B_G = input_ids_q.shape[0]
        sigma_sq = self.policy_variance
        eps_clip = self.clip_epsilon
        beta = self.kl_coeff

        adv_flat = advantages.reshape(-1)  # (B*G,)

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_steps = len(trajectory["states"])

        for i in range(num_steps):
            x_t = trajectory["states"][i].detach()    # (B*G, 3, 128)
            v_old = trajectory["actions"][i].detach()  # (B*G, 3, 128)
            t = trajectory["timesteps"][i]

            timestep = torch.full((B_G,), t, device=x_t.device)

            # New policy velocity (WITH gradients) — through FSDP wrapper
            v_new = self._call_velocity(self.model, input_ids_q, x_t, timestep)

            # Reference policy velocity (no gradients) — through FSDP wrapper
            with torch.no_grad():
                v_ref = self._call_velocity(self.ref_model, input_ids_q, x_t, timestep)

            # Log-prob ratio:
            # log pi_old(a|s) = 0 since a = v_old exactly
            # log pi_new(a|s) = -||v_new - v_old||^2 / (2 * sigma^2)
            diff_sq = ((v_new - v_old) ** 2).sum(dim=(1, 2))  # (B*G,)
            log_ratio = -diff_sq / (2 * sigma_sq)
            ratio = torch.exp(log_ratio)  # (B*G,), <= 1

            # Clipped surrogate
            clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
            surr1 = ratio * adv_flat
            surr2 = clipped_ratio * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty via velocity MSE vs reference
            kl_loss = ((v_new - v_ref) ** 2).sum(dim=(1, 2)).mean()

            total_policy_loss += policy_loss
            total_kl_loss += kl_loss

        total_policy_loss /= num_steps
        total_kl_loss /= num_steps

        return total_policy_loss + beta * total_kl_loss

    def train_step(self, batch):
        """
        One GRPO training step:
        1. Sample trajectories and compute rewards (no grad)
        2. Compute group-relative advantages (local per-question)
        3. Run PPO inner epochs with clipped surrogate loss

        Args:
            batch: dict with 'input_ids_q' and 'gt_solutions'

        Returns:
            dict with loss, mean_reward, reward_std for logging
        """
        # 1. Sample trajectories and compute rewards
        trajectory, rewards, input_ids_q_expanded, decoded_texts = self.sample_and_evaluate(batch)

        mean_reward = rewards.mean().item()
        reward_std = rewards.std().item()

        # 2. Compute advantages (local per-group normalization — each rank
        #    processes different questions via DistributedSampler, and advantages
        #    are computed per-question, so no cross-rank sync needed)
        advantages = self.compute_advantages(rewards)

        # 3. PPO inner epochs
        total_loss = 0.0
        for epoch in range(self.num_ppo_epochs):
            self.optimizer.zero_grad()
            loss = self.compute_grpo_loss(
                trajectory, advantages, input_ids_q_expanded
            )
            loss.backward()
            # clip_grad_norm_ works with FSDP use_orig_params=True
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()
            total_loss += loss.item()

        return {
            "loss": total_loss / self.num_ppo_epochs,
            "mean_reward": mean_reward,
            "reward_std": reward_std,
        }
