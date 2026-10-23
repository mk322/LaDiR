"""Small building blocks. Module/attribute names are part of the checkpoint format; do not rename."""

import math

import torch
import torch.nn as nn


class Projector(nn.Module):
    """Two-layer MLP with SiLU."""

    def __init__(self, dim: int, mlp_dim: int = None, out_dim: int = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim or dim)
        self.fc2 = nn.Linear(mlp_dim or dim, out_dim or dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MLPConnector(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear, mapping latents (latent_dim) to the LLM hidden size."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, hidden_dim, bias=False),
        )

    def forward(self, x):
        return self.mlp(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of an integer timestep label followed by a Projector."""

    def __init__(self, embed_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj = Projector(freq_dim, embed_dim, embed_dim)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.freq_dim // 2
        freq = torch.arange(half, dtype=torch.float32, device=timestep.device)
        freq = freq.mul(-math.log(10000) / half).exp().unsqueeze(0)
        emb = timestep.unsqueeze(-1).float() * freq
        return self.proj(torch.cat([emb.cos(), emb.sin()], dim=-1).to(dtype=dtype))


class FinalLayer(nn.Module):
    """AdaLN output head: hidden states -> velocity in latent space, modulated by the timestep embedding."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)
