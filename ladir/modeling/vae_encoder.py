"""Encoder half of the frozen VAE used by stage-wise LaDiR SFT.

Stage-wise checkpoints store it under `vae.*`:
  vae.encoder_model.*   full Qwen3-4B-Base CausalLM (vocab 151936, not resized)
  vae.latents_query     [1, num_latents, 2560]
  vae.fc_mu / fc_var    Linear(2560, latent_dim)
Only `mu` is used downstream. The full VAE (with decoder) lives in the vae/ package.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .flow import positions_from_mask


class VAEEncoder(nn.Module):
    def __init__(self, encoder_model_name: str = "Qwen/Qwen3-4B-Base", num_latents: int = 64, latent_dim: int = 2560,
                 mask_mode: str = "causal", pretrained: bool = False):
        """mask_mode:
        "causal": 2D padding mask, causal attention everywhere (used to encode stage-wise SFT targets).
        "hybrid": latent queries attend to each other bidirectionally (as in VAE training).
        """
        super().__init__()
        self.num_latents = num_latents
        if pretrained:
            self.encoder_model = AutoModelForCausalLM.from_pretrained(encoder_model_name, torch_dtype=torch.bfloat16)
        else:
            self.encoder_model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(encoder_model_name),
                                                                  torch_dtype=torch.bfloat16)
        hidden = self.encoder_model.config.hidden_size
        self.latents_query = nn.Parameter(torch.randn(1, num_latents, hidden))
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)
        self.mask_mode = mask_mode
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask: [B, T] right-padded. Returns mu [B, num_latents, latent_dim]."""
        B, T = input_ids.shape
        N = self.num_latents
        embeds = self.encoder_model.get_input_embeddings()(input_ids)
        embeds = torch.cat([embeds, self.latents_query.expand(B, -1, -1).to(embeds.dtype)], dim=1)
        full_mask = torch.cat([attention_mask, attention_mask.new_ones(B, N)], dim=1)
        if self.mask_mode == "causal":
            mask = full_mask
        else:
            L = T + N
            m = torch.tril(torch.ones(L, L, dtype=torch.bool, device=input_ids.device))
            m[T:, T:] = True
            m = m.unsqueeze(0).unsqueeze(0) & full_mask.bool()[:, None, None, :]
            mask = torch.zeros(m.shape, dtype=embeds.dtype, device=m.device).masked_fill(~m, torch.finfo(embeds.dtype).min)
        # pad-skipping positions: independent of batch padding
        hidden = self.encoder_model.model(inputs_embeds=embeds, attention_mask=mask,
                                          position_ids=positions_from_mask(full_mask), use_cache=False,
                                          return_dict=True).last_hidden_state[:, -N:]
        return self.fc_mu(hidden)
