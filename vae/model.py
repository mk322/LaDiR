"""Reasoning-block VAE: LM encoder with learnable latent queries + LM decoder trained with CE.

  encoder : [text_0 .. text_{T-1}, q_0 .. q_{N-1}]      -> hidden(q) -> fc_mu / fc_var
  decoder : [W z_0 .. W z_{N-1}, ae_speical_token, text_0 .. text_{T-1}]
            hidden at position N + i predicts text_i      (position N is the AE token, it predicts text_0)

Mask modes (`encoder_mask_mode`, `decoder_mask_mode`):
  "hybrid": latent positions attend to each other bidirectionally, everything else causal.
  "causal": plain causal attention with a 2D padding mask.

Encoder position ids (`position_mode`):
  "pad_skip": text token i gets position i, latent query j gets len(text) + j (independent of batch padding).
  "padded":   arange(T_padded + N); the latent queries' positions depend on the padded batch length.

Parameter names match the `vae.*` block of stage-wise LaDiR checkpoints.
"""

import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MASK_MODES = ("hybrid", "causal")
POSITION_MODES = ("pad_skip", "padded")
VAE_PREFIX_KEYS = ("latents_query", "fc_mu.", "fc_var.", "encoder_model.", "decoder_model.", "latent_to_decoder.",
                   "ae_speical_token")


@dataclass
class VAEConfig:
    encoder_name: str = "Qwen/Qwen3-4B-Base"
    decoder_name: str = "Qwen/Qwen3-8B-Base"
    num_latents: int = 64
    latent_dim: int = 2560
    encoder_mask_mode: str = "hybrid"
    decoder_mask_mode: str = "hybrid"
    position_mode: str = "pad_skip"
    load_decoder: bool = True
    # --- trainable parts
    train_encoder_lm: bool = False
    train_encoder_queries: bool = True     # latents_query
    train_encoder_heads: bool = True       # fc_mu, fc_var
    train_decoder_lm: bool = True          # decoder_model
    train_decoder_adapter: bool = True     # latent_to_decoder, ae_speical_token
    # --- posterior
    sample_posterior: bool = True          # z = mu + eps*std in training; eval always uses z = mu
    extra_noise_std: float = 0.0           # extra N(0, std^2) noise on z during training
    logvar_clamp: Tuple[float, float] = (-30.0, 20.0)
    # --- loss
    ce_fp32: bool = True                   # lm_head matmul + log-softmax in fp32
    ce_chunk_tokens: int = 2048            # valid tokens per lm_head chunk (activation-checkpointed); 0 = no chunking
    query_init_std: float = 1.0
    gradient_checkpointing: bool = False

    def __post_init__(self):
        assert self.encoder_mask_mode in MASK_MODES and self.decoder_mask_mode in MASK_MODES
        assert self.position_mode in POSITION_MODES
        self.logvar_clamp = tuple(self.logvar_clamp)


# ----------------------------------------------------------------------------------------------------- masks
def additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool [B,1,L,L] (True = attend) -> additive float mask (0 / finfo.min), the format `VAEEncoder` uses."""
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


def build_mask(mode: str, key_mask: torch.Tensor, latent_start: int, num_latents: int, dtype: torch.dtype):
    """key_mask: [B, L] 1 = real token. Returns what to pass as `attention_mask` to the HF model."""
    if mode == "causal":
        return key_mask
    L = key_mask.shape[1]
    m = torch.tril(torch.ones(L, L, dtype=torch.bool, device=key_mask.device))
    m[latent_start:latent_start + num_latents, latent_start:latent_start + num_latents] = True
    m = m[None, None] & key_mask.bool()[:, None, None, :]
    return additive_mask(m, dtype)


def encoder_position_ids(attention_mask: torch.Tensor, num_latents: int) -> torch.Tensor:
    """Pad-skipping positions for [text (right-padded), queries]: text i -> i, query j -> len + j."""
    lengths = attention_mask.sum(1, keepdim=True)
    text_pos = (attention_mask.long().cumsum(1) - 1).clamp(min=0)
    q_pos = lengths + torch.arange(num_latents, device=attention_mask.device)[None]
    return torch.cat([text_pos, q_pos], dim=1)


class FP32LMHead(nn.Linear):
    """nn.Linear whose forward runs in fp32 (also under autocast). Swapped in via __class__ so parameter names,
    weight tying and DeepSpeed ZeRO-3 hooks of the lm_head are kept."""

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            b = None if self.bias is None else self.bias.float()
            return F.linear(x.float(), self.weight.float(), b)


# ----------------------------------------------------------------------------------------------------- model
class ReasoningVAE(nn.Module):
    def __init__(self, config: VAEConfig, encoder_model: nn.Module, decoder_model: Optional[nn.Module] = None):
        """encoder_model / decoder_model: HF CausalLMs (see `build_vae`)."""
        super().__init__()
        self.config = config
        self.num_latents = config.num_latents
        self.encoder_model = encoder_model
        enc_hidden = encoder_model.config.hidden_size
        self.latents_query = nn.Parameter(torch.randn(1, config.num_latents, enc_hidden) * config.query_init_std)
        self.fc_mu = nn.Linear(enc_hidden, config.latent_dim)
        self.fc_var = nn.Linear(enc_hidden, config.latent_dim)
        if decoder_model is not None:
            dec_hidden = decoder_model.config.hidden_size
            self.decoder_model = decoder_model
            self.latent_to_decoder = nn.Linear(config.latent_dim, dec_hidden)
            self.ae_speical_token = nn.Parameter(torch.randn(1, dec_hidden))
            if config.ce_fp32 and type(decoder_model.lm_head) is nn.Linear:
                decoder_model.lm_head.__class__ = FP32LMHead
        else:
            self.decoder_model = self.latent_to_decoder = self.ae_speical_token = None
        if config.gradient_checkpointing:
            for m in (self.encoder_model, self.decoder_model):
                if m is not None:
                    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.set_trainable()

    # ------------------------------------------------------------------ trainable parts
    def set_trainable(self):
        c = self.config
        self.encoder_model.requires_grad_(c.train_encoder_lm)
        self.latents_query.requires_grad_(c.train_encoder_queries)
        self.fc_mu.requires_grad_(c.train_encoder_heads)
        self.fc_var.requires_grad_(c.train_encoder_heads)
        if self.decoder_model is not None:
            self.decoder_model.requires_grad_(c.train_decoder_lm)
            self.latent_to_decoder.requires_grad_(c.train_decoder_adapter)
            self.ae_speical_token.requires_grad_(c.train_decoder_adapter)

    def trainable_summary(self) -> dict:
        out = {}
        for name, p in self.named_parameters():
            top = name.split(".")[0]
            d = out.setdefault(top, [0, 0])
            d[0] += p.numel()
            d[1] += p.numel() if p.requires_grad else 0
        return {k: {"params": v[0], "trainable": v[1]} for k, v in out.items()}

    # ------------------------------------------------------------------ encoder
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """input_ids/attention_mask [B, T], right-padded (mask None = no padding). Returns mu, log_var [B, N, D]."""
        mask_mode = self.config.encoder_mask_mode
        position_mode = self.config.position_mode
        B, T = input_ids.shape
        N = self.num_latents
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeds = self.encoder_model.get_input_embeddings()(input_ids)
        embeds = torch.cat([embeds, self.latents_query.expand(B, -1, -1).to(embeds.dtype)], dim=1)
        key_mask = torch.cat([attention_mask, attention_mask.new_ones(B, N)], dim=1)
        mask = build_mask(mask_mode, key_mask, T, N, embeds.dtype)
        pos = encoder_position_ids(attention_mask, N) if position_mode == "pad_skip" else None
        # base model only (== hidden_states[-1] of the CausalLM, without the unused full-vocab logits)
        out = self.encoder_model.model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=False,
                                       return_dict=True)
        hidden = out.last_hidden_state[:, -N:]
        hidden = hidden.to(self.fc_mu.weight.dtype)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden).clamp(*self.config.logvar_clamp)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if not self.training:
            return mu
        z = mu
        if self.config.sample_posterior:
            z = z + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        if self.config.extra_noise_std > 0:
            z = z + torch.randn_like(z) * self.config.extra_noise_std
        return z

    # ------------------------------------------------------------------ decoder
    def decoder_prefix(self, z: torch.Tensor) -> torch.Tensor:
        """[B, N, D] -> [B, N+1, H]: projected latents followed by the AE token."""
        z = z.to(self.latent_to_decoder.weight.dtype)
        zp = self.latent_to_decoder(z)
        emb_dtype = self.decoder_model.get_input_embeddings().weight.dtype
        ae = self.ae_speical_token[None].expand(z.shape[0], -1, -1)
        return torch.cat([zp.to(emb_dtype), ae.to(emb_dtype)], dim=1)

    def decoder_hidden(self, z, input_ids, attention_mask):
        """Decoder final hidden states (after norm) at the positions that predict text_0..text_{T-1}: [B, T, H]."""
        mask_mode = self.config.decoder_mask_mode
        B, T = input_ids.shape
        N = self.num_latents
        prefix = self.decoder_prefix(z)
        embeds = torch.cat([prefix, self.decoder_model.get_input_embeddings()(input_ids)], dim=1)
        key_mask = torch.cat([attention_mask.new_ones(B, N + 1), attention_mask], dim=1)
        mask = build_mask(mask_mode, key_mask, 0, N, embeds.dtype)
        out = self.decoder_model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True)
        return out.last_hidden_state[:, N:N + T]

    def ce_loss(self, hidden: torch.Tensor, labels: torch.Tensor):
        """hidden [B, T, H] (already aligned: hidden[:, i] predicts labels[:, i]); labels -100 = ignore.
        Returns (mean CE over valid tokens, n_valid, n_correct). lm_head + log-softmax in fp32 when ce_fp32."""
        valid = labels != -100
        h = hidden[valid]
        y = labels[valid]
        n = y.numel()
        if n == 0:
            return hidden.sum() * 0.0, 0, 0
        head = self.decoder_model.lm_head

        def chunk_fn(hc, yc):
            logits = head(hc).float()
            return F.cross_entropy(logits, yc, reduction="sum"), (logits.argmax(-1) == yc).sum()

        step = self.config.ce_chunk_tokens if self.config.ce_chunk_tokens > 0 else n
        total = hidden.new_zeros((), dtype=torch.float32)
        correct = 0
        for s in range(0, n, step):
            hc, yc = h[s:s + step], y[s:s + step]
            if torch.is_grad_enabled() and hc.requires_grad and step < n:
                loss_c, corr_c = checkpoint(chunk_fn, hc, yc, use_reentrant=False)
            else:
                loss_c, corr_c = chunk_fn(hc, yc)
            total = total + loss_c
            correct += int(corr_c)
        return total / n, n, correct

    # ------------------------------------------------------------------ training forward
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None,
                kl_weight: float = 0.0):
        """Returns dict(loss, ce_loss, kl_loss, n_tokens, n_correct, mu, log_var)."""
        if labels is None:
            labels = input_ids.masked_fill(attention_mask == 0, -100)
        mu, log_var = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, log_var)
        hidden = self.decoder_hidden(z, input_ids, attention_mask)
        ce, n, correct = self.ce_loss(hidden, labels)
        # mean over all latent elements
        kl = -0.5 * torch.mean(1 + log_var.float() - mu.float().pow(2) - log_var.float().exp())
        return {"loss": ce + kl_weight * kl, "ce_loss": ce, "kl_loss": kl, "n_tokens": n, "n_correct": correct,
                "mu": mu, "log_var": log_var}

    # ------------------------------------------------------------------ state dict helpers
    def load_vae_state(self, state: dict) -> dict:
        """Load a VAE state dict (optional 'module.' or 'vae.' key prefixes). Missing decoder keys are allowed only
        when this model has no decoder; tied lm_head/embedding duplicates are tolerated."""
        state = _strip_prefixes(state)
        expected = self.state_dict()
        missing = [k for k in expected if k not in state]
        unexpected = [k for k in state if k not in expected]
        tied = _tied_keys(self)
        missing = [k for k in missing if k not in tied]
        if self.decoder_model is None:
            unexpected = [k for k in unexpected if not k.startswith(("decoder_model.", "latent_to_decoder.",
                                                                     "ae_speical_token"))]
        if missing or unexpected:
            raise RuntimeError(f"VAE state mismatch: missing={missing[:10]} (n={len(missing)}) "
                               f"unexpected={unexpected[:10]} (n={len(unexpected)})")
        keep = {k: v for k, v in state.items() if k in expected}
        self.load_state_dict(keep, strict=False)
        return {"missing": missing, "unexpected": unexpected}


def _strip_prefixes(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        for p in ("module.", "vae."):
            if k.startswith(p) and k[len(p):].startswith(VAE_PREFIX_KEYS):
                k = k[len(p):]
        out[k] = v
    return out


def _tied_keys(model: nn.Module) -> set:
    """state_dict keys that alias another key's storage (e.g. tied lm_head)."""
    seen, tied = {}, set()
    for k, v in model.state_dict(keep_vars=True).items():   # tied weights are the same Parameter object
        if id(v) in seen:
            tied.update((k, seen[id(v)]))
        else:
            seen[id(v)] = k
    return tied


# ----------------------------------------------------------------------------------------------------- builders
def build_vae(config: VAEConfig, pretrained: bool = True, dtype=torch.bfloat16) -> ReasoningVAE:
    """Build from HF names. pretrained=False -> random init from the configs (for loading a full checkpoint)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    def lm(name):
        if pretrained:
            return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
        return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), torch_dtype=dtype,
                                                attn_implementation="sdpa")

    enc = lm(config.encoder_name)
    dec = lm(config.decoder_name) if config.load_decoder else None
    model = ReasoningVAE(config, enc, dec)
    # the new heads/queries are created in fp32 by nn defaults; cast only them (the LMs may already be
    # partitioned by DeepSpeed zero.Init, where a whole-module .to() is unsafe)
    for m in (model.fc_mu, model.fc_var, model.latent_to_decoder):
        if m is not None:
            m.to(dtype)
    for name in ("latents_query", "ae_speical_token"):
        p = getattr(model, name)
        if p is not None:
            p.data = p.data.to(dtype)
    return model


def load_state_file(path: str) -> dict:
    """Read a .pt (`torch.save(state_dict)`) or .safetensors VAE state dict to CPU."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return state


def read_stage2_vae_state(path_or_repo: str) -> dict:
    """Read only the `vae.*` tensors of a stage-2 LaDiR checkpoint (local model.safetensors file, a directory that
    contains it, or an HF repo id such as 'mk322/LaDiR-Code-8B'). Returns keys without the 'vae.'
    prefix: encoder_model.*, latents_query [1,N,2560], fc_mu.*, fc_var.* (no decoder)."""
    from safetensors import safe_open
    path = path_or_repo
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    elif not os.path.isfile(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(path_or_repo, "model.safetensors")
    out = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.startswith("vae."):
                out[k[len("vae."):]] = f.get_tensor(k)
    if not out:
        raise RuntimeError(f"{path} has no vae.* tensors (not a stage-wise checkpoint?)")
    return out


def config_to_dict(cfg: VAEConfig) -> dict:
    d = asdict(cfg)
    d["logvar_clamp"] = list(d["logvar_clamp"])
    return d
