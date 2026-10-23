"""Offline VAE latent pre-encoding for stage-wise SFT.

Encodes every reasoning block of a pretokenized dataset with the frozen VAE encoder once, so stage-wise training
feeds `gt_latents` to the model and never loads the VAE.

  * batch-invariant: per-sample masks from lengths (EOS included, as in VAE training) and pad-skipping position
    ids, so every latent equals encoding that sample alone
  * output shard k is aligned 1:1 with data shard k: latents_0000k.pt = {"latents": [n, N, D] (bf16 by default)}
  * resumable and multi-GPU: shard k is handled by rank k % world_size; finished shards are skipped
  * meta.json (written last) records the data fingerprint, VAE source and encoding settings

Usage (8 GPUs):
    torchrun --nproc_per_node=8 -m sft.precompute_latents --data_dir $DATA/ling_coder_tok \
        --out_dir $DATA/ling_coder_latents --vae mk322/LaDiR-Code-8B
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from .data import META, TokenShards, latent_shard_name
from .vae_weights import build_vae, vae_mask_mode

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def encode_shard(vae, tokens: TokenShards, shard: int, batch_size: int, pad_id: int, device, store_dtype):
    start, end = int(tokens.starts[shard]), int(tokens.starts[shard + 1])
    seqs = [tokens.get(i)["reasoning_input_ids"] for i in range(start, end)]
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))      # length-sorted batches: less padding
    out = torch.empty(len(seqs), vae.num_latents, vae.fc_mu.out_features, dtype=store_dtype)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        batch = [seqs[i] for i in idx]
        lens = torch.tensor([len(x) for x in batch])
        ids = pad_sequence(batch, batch_first=True, padding_value=pad_id)
        mask = (torch.arange(ids.shape[1])[None] < lens[:, None]).long()
        mu = vae.encode(ids.to(device), mask.to(device))       # batch-invariant (pad-skipping positions)
        out[idx] = mu.to("cpu", store_dtype)
    return out


def run(data_dir: str, out_dir: str, vae_source: str, vae_encoder: str = "Qwen/Qwen3-4B-Base",
        num_latents: int = 64, latent_dim: int = 2560, mask_mode: str = "auto", batch_size: int = 32,
        store_dtype: str = "bf16", compute_dtype: str = "bf16", device=None, rank: int = 0, world_size: int = 1):
    if mask_mode == "auto":
        mask_mode = vae_mask_mode(vae_source)
    if device is None:
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    tokens = TokenShards(data_dir)
    meta = tokens.meta
    info = {
        "data_fingerprint": meta["fingerprint"],
        "num_samples": meta["num_samples"],
        "vae_source": vae_source,
        "vae_encoder": vae_encoder,
        "num_latents": num_latents,
        "latent_dim": latent_dim,
        "mask_mode": mask_mode,
        "positions": "pad_skipping",
        "dtype": store_dtype,
        "compute_dtype": compute_dtype,
    }
    os.makedirs(out_dir, exist_ok=True)
    final, partial = os.path.join(out_dir, META), os.path.join(out_dir, "meta.partial.json")
    for existing in (final, partial):   # refuse to mix shards from a different dataset / VAE / setting
        if os.path.exists(existing):
            with open(existing) as f:
                if json.load(f) != info:
                    raise RuntimeError(f"{existing} describes a different encoding; use a fresh --out_dir")
    if world_size > 1:
        dist.barrier()
    if rank == 0 and not os.path.exists(final):
        with open(partial, "w") as f:
            json.dump(info, f, indent=2)
    vae = build_vae(vae_source, vae_encoder, num_latents, latent_dim, mask_mode, device, DTYPES[compute_dtype])
    for k in range(rank, len(meta["shards"]), world_size):
        path = os.path.join(out_dir, latent_shard_name(k))
        if os.path.exists(path):
            continue
        latents = encode_shard(vae, tokens, k, batch_size, meta["pad_token_id"], device, DTYPES[store_dtype])
        torch.save({"latents": latents}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"[rank {rank}] shard {k}: {latents.shape[0]} samples -> {path}", flush=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0 and os.path.exists(partial):
        os.replace(partial, final)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_dir", required=True, help="pretokenized dataset (python -m sft.data)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vae", required=True,
                   help="LaDiR checkpoint (dir / HF repo) or a vae.train export (weights/step_N.safetensors)")
    p.add_argument("--vae_encoder", default="Qwen/Qwen3-4B-Base", help="architecture config of the VAE encoder LLM")
    p.add_argument("--num_latents", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=2560)
    p.add_argument("--mask_mode", default="auto", choices=["auto", "causal", "hybrid"],
                   help="auto: the mask a vae.train export was trained with, causal for a LaDiR checkpoint")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES), help="storage dtype")
    p.add_argument("--compute_dtype", default="bf16", choices=list(DTYPES))
    a = p.parse_args(argv)
    rank, world = 0, 1
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    run(a.data_dir, a.out_dir, a.vae, a.vae_encoder, a.num_latents, a.latent_dim, a.mask_mode, a.batch_size,
        a.dtype, a.compute_dtype, rank=rank, world_size=world)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
