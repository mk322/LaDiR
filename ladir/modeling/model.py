"""LaDiR model: one LLM backbone used in two modes.

  flow mode : [prompt, <bot>, <timestep>, flow_projector(z_t) x N]  -> output_projection -> velocity  (latent diffusion)
  text mode : [prompt, <bot>, ce_projector(z) x N, <eot>, answer]    -> next-token logits             (answer decoding)

Latent targets z come from a frozen VAE encoder (`vae.*`).

Attribute names define the checkpoint format of the released checkpoint (mk322/LaDiR-Code-8B).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import add_noise, flow_target, flow_sequence_mask, positions_from_mask
from .layers import MLPConnector, TimeEmbedding, FinalLayer


@dataclass
class LaDiRConfig:
    backbone: str = "Qwen/Qwen3-8B-Base"
    hidden_size: int = 4096
    num_latents: int = 64
    latent_dim: int = 2560
    num_train_timesteps: int = 1000
    # "vae": build the frozen VAE encoder (encoding training targets); "none": inference only
    latent_source: str = "none"
    vae_encoder_name: str = "Qwen/Qwen3-4B-Base"
    vae_mask_mode: str = "causal"
    flow_mask: str = "train_matched"            # flow-mode attention mask, see flow.flow_sequence_mask
    # training
    text_drop_prob: float = 0.1
    diffusion_batch_mul: int = 4
    diffusion_loss_weight: float = 5.0
    ce_loss_weight: float = 1.0
    diffusion_forcing: bool = True
    diffusion_forcing_prob: float = 0.5
    diffusion_forcing_noise_scale: float = 0.3


class LaDiRModel(nn.Module):
    def __init__(self, config: LaDiRConfig, backbone_llm: nn.Module, tokenizer, vae: nn.Module = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.backbone_llm = backbone_llm
        self.backbone_llm.config.use_cache = False
        self.vae = vae
        H, D = config.hidden_size, config.latent_dim

        self.ce_projector = MLPConnector(D, H)
        self.flow_projector = MLPConnector(D, H)
        self.output_projection = FinalLayer(H, D)
        self.bot_embed = nn.Parameter(torch.randn(1, H) * 0.02)
        self.eot_embed = nn.Parameter(torch.randn(1, H) * 0.02)
        self.null_embed = nn.Parameter(torch.randn(1, H) * 0.02)
        self.time_embed = TimeEmbedding(H)   # input <timestep> token
        self.t_embedder = TimeEmbedding(H)   # AdaLN conditioning of the output head
        self.d_embed = TimeEmbedding(H)      # unused; kept for checkpoint compatibility
        self._init_new_weights()

    def _init_new_weights(self):
        for module in [self.ce_projector, self.flow_projector, self.output_projection]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------------------------------------------------
    def embed(self, input_ids):
        return self.backbone_llm.get_input_embeddings()(input_ids)

    def _special(self, param, B, dtype):
        return param.expand(B, 1, -1).to(dtype)

    def null_prompt_embeds(self, prompt_embeds):
        """Unconditional prompt for CFG: every prompt position except the first -> null_embed."""
        out = prompt_embeds.clone()
        out[:, 1:] = self.null_embed.to(out.dtype)
        return out

    # ------------------------------------------------------------------------------------------------------------
    # flow mode
    # ------------------------------------------------------------------------------------------------------------
    def flow_velocity(self, prompt_ids, prompt_mask, latents, timesteps, prompt_embeds=None):
        """Velocity prediction over [prompt, <bot>, <timestep>, flow_projector(z_t)].

        prompt_ids/prompt_mask: [B, T] right-padded; latents: [B, N, D]; timesteps: [B] integer labels.
        prompt_embeds: optional override (e.g. null prompt for CFG); defaults to embed(prompt_ids).
        """
        tok = self.tokenizer
        B, T = prompt_ids.shape
        N = latents.shape[1]
        if prompt_embeds is None:
            prompt_embeds = self.embed(prompt_ids)
        dtype = prompt_embeds.dtype
        inputs_embeds = torch.cat([
            prompt_embeds,
            self._special(self.bot_embed, B, dtype),
            self.time_embed(timesteps, dtype).unsqueeze(1),
            self.flow_projector(latents).to(dtype),
        ], dim=1)
        tail = torch.tensor([tok.bot_token_id, tok.time_token_id] + [tok.latent_token_id] * N, device=prompt_ids.device)
        full_ids = torch.cat([prompt_ids, tail.unsqueeze(0).expand(B, -1)], dim=1)
        valid = torch.cat([prompt_mask, prompt_mask.new_ones(B, N + 2)], dim=1)
        mask = flow_sequence_mask(full_ids, valid, tok.latent_token_id, self.config.flow_mask, dtype)
        hidden = self.backbone_llm(inputs_embeds=inputs_embeds, attention_mask=mask,
                                   position_ids=positions_from_mask(valid), output_hidden_states=True,
                                   use_cache=False, return_dict=True).hidden_states[-1][:, -N:]
        return self.output_projection(hidden, self.t_embedder(timesteps, hidden.dtype))

    # ------------------------------------------------------------------------------------------------------------
    # text mode
    # ------------------------------------------------------------------------------------------------------------
    def text_prefix(self, prompt_ids, latents):
        """[prompt, <bot>, ce_projector(z), <eot>] embeddings."""
        prompt_embeds = self.embed(prompt_ids)
        B, dtype = prompt_ids.shape[0], prompt_embeds.dtype
        return torch.cat([prompt_embeds, self._special(self.bot_embed, B, dtype),
                          self.ce_projector(latents).to(dtype), self._special(self.eot_embed, B, dtype)], dim=1)

    # ------------------------------------------------------------------------------------------------------------
    # latent targets
    # ------------------------------------------------------------------------------------------------------------
    def encode_reasoning(self, reasoning_ids, reasoning_mask):
        """Latent targets z [B, N, D] for a reasoning block."""
        if self.config.latent_source != "vae" or self.vae is None:
            raise ValueError("encoding reasoning needs the VAE encoder (latent_source='vae')")
        return self.vae.encode(reasoning_ids, reasoning_mask).to(torch.bfloat16)

    @property
    def dtype(self):
        return self.bot_embed.dtype

    # ------------------------------------------------------------------------------------------------------------
    # training losses
    # ------------------------------------------------------------------------------------------------------------
    def compute_flow_loss(self, prompt_ids, prompt_mask, z):
        cfg = self.config
        k = cfg.diffusion_batch_mul
        z, prompt_ids, prompt_mask = z.repeat(k, 1, 1), prompt_ids.repeat(k, 1), prompt_mask.repeat(k, 1)
        noise = torch.randn_like(z)
        t = torch.randint(0, cfg.num_train_timesteps, (z.shape[0],), device=z.device)
        z_t = add_noise(z, noise, t, cfg.num_train_timesteps)
        prompt_embeds = self.embed(prompt_ids)
        if self.training and cfg.text_drop_prob > 0:  # per-sample CFG dropout
            drop = torch.rand(z.shape[0], device=z.device) < cfg.text_drop_prob
            prompt_embeds = torch.where(drop[:, None, None], self.null_prompt_embeds(prompt_embeds), prompt_embeds)
        v = self.flow_velocity(prompt_ids, prompt_mask, z_t, t, prompt_embeds=prompt_embeds)
        return F.mse_loss(v.float(), flow_target(z, noise).float())

    def apply_diffusion_forcing(self, z):
        cfg = self.config
        if not (self.training and cfg.diffusion_forcing):
            return z
        B = z.shape[0]
        scale = torch.rand(B, device=z.device) * cfg.diffusion_forcing_noise_scale
        scale = scale * (torch.rand(B, device=z.device) < cfg.diffusion_forcing_prob).float()
        return (z + torch.randn_like(z) * scale.view(B, 1, 1)).to(z.dtype)

    def compute_ce_loss(self, prompt_ids, prompt_mask, z, answer_ids, answer_mask):
        """CE on answer tokens given [prompt, <bot>, z, <eot>]. Masks come from sequence lengths, so the final EOS
        is trained even though Qwen's pad token equals its EOS token."""
        B, N = z.shape[0], z.shape[1]
        prefix = self.text_prefix(prompt_ids, self.apply_diffusion_forcing(z))
        embeds = torch.cat([prefix, self.embed(answer_ids)], dim=1)
        mask = torch.cat([prompt_mask, prompt_mask.new_ones(B, N + 2), answer_mask], dim=1)
        logits = self.backbone_llm(inputs_embeds=embeds, attention_mask=mask, position_ids=positions_from_mask(mask),
                                   use_cache=False, return_dict=True).logits
        start = prefix.shape[1]
        pred = logits[:, start - 1:start - 1 + answer_ids.shape[1]]
        labels = answer_ids.masked_fill(answer_mask == 0, -100)
        return F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(), labels.reshape(-1), ignore_index=-100)

    def forward(self, prompt_input_ids, prompt_attention_mask, reasoning_input_ids, reasoning_attention_mask,
                answer_input_ids, answer_attention_mask, gt_latents=None, **_):
        cfg = self.config
        if gt_latents is None:
            gt_latents = self.encode_reasoning(reasoning_input_ids, reasoning_attention_mask)
        flow_loss = self.compute_flow_loss(prompt_input_ids, prompt_attention_mask, gt_latents)
        ce_loss = self.compute_ce_loss(prompt_input_ids, prompt_attention_mask, gt_latents,
                                       answer_input_ids, answer_attention_mask)
        loss = cfg.diffusion_loss_weight * flow_loss + cfg.ce_loss_weight * ce_loss
        return {"loss": loss, "flow_loss": flow_loss.detach(), "ce_loss": ce_loss.detach()}
