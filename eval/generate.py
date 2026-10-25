"""Generate raw benchmark outputs with a LaDiR checkpoint (scoring is separate: eval/score.py).

    python -m eval.generate --checkpoint mk322/LaDiR-Code-8B --benchmarks humaneval,mbpp,lcb_v6

Generation setting (fixed): CFG scale 4, 50 flow steps, greedy decoding, 1 sample per task, at most 768 new tokens.

Output: <out_root>/<ckpt_name>/<bench>.jsonl, one row per task:
    {task_id, sample_idx, prompt, raw, noise_seed, <GEN_FIELDS of the benchmark>}
plus <bench>.meta.json with the run settings (checkpoint, sampler config, decode backend, git sha, ...).
Runs are resumable: rows already present are skipped; resuming with different settings is an error.

Pipeline: latents are sampled in batches (--latent_batch_size), then decoded by a backend (--decode_backend):
  hf    ladir.sampling.decode (PyTorch KV cache, --batch_size); batch-invariant
  vllm  ladir.vllm_decode.VLLMDecoder on the exported HF backbone (faster)
  auto  (default) vllm when importable and on GPU, else hf. The backend is recorded in <bench>.meta.json.

The initial latent noise of each task comes from its own seeded generator, so outputs do not depend on batch sizes or
order.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import zlib
from typing import List, Optional

import torch

from .benchmarks import GEN_FIELDS, canonical_name, load_benchmark, open_fence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The evaluation setting: CFG 4, 50 flow steps, greedy decoding, one sample per task.
CFG_SCALE = 4.0
NUM_STEPS = 50
MAX_NEW_TOKENS = 768


def sampler_config(**overrides):
    """The fixed eval sampler config."""
    from ladir.sampling import get_sampler_config
    return get_sampler_config(**{"cfg_scale": CFG_SCALE, "num_steps": NUM_STEPS, **overrides})


# ----------------------------------------------------------------------------------------------------------------
# naming / metadata
# ----------------------------------------------------------------------------------------------------------------
def ckpt_name(checkpoint: str) -> str:
    return os.path.basename(os.path.normpath(checkpoint))


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", "ladir", "eval"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def noise_seed(task_id: str) -> int:
    """Seed of the initial latent noise of one task: crc32(task_id)."""
    return zlib.crc32(task_id.encode())


def make_noise(seed: int, num_latents: int, latent_dim: int) -> torch.Tensor:
    return torch.randn(num_latents, latent_dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


# ----------------------------------------------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------------------------------------------
def _read_rows(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:   # torn last line from a killed run
                        pass
    return rows


def _repair_tail(path):
    """Drop a partial last line left by a killed run so appended rows start on a fresh line."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def _check_meta(meta_path, meta):
    keys = ("checkpoint", "sampler", "max_new_tokens", "decode_backend")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prev = json.load(f)
        diff = {k: (prev.get(k), meta.get(k)) for k in keys if prev.get(k) != meta.get(k)}
        if diff:
            raise RuntimeError(f"{meta_path}: existing run has different settings {diff}; use another --out_root")
        meta = {**prev, "resumed": prev.get("resumed", []) + [{"time": meta["time"], "git_sha": meta["git_sha"]}]}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------------------------------------------------
# decode backends: decoder.decode(model, ids, mask, latents, items, max_new_tokens) -> [str]   (greedy)
# ----------------------------------------------------------------------------------------------------------------
class HFDecoder:
    """ladir.sampling.decode (PyTorch, KV cache). Exactly batch-invariant."""
    name = "hf"

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size

    def decode(self, model, ids, mask, latents, items, max_new_tokens):
        from ladir.sampling import decode
        out = []
        for j in range(0, len(items), self.batch_size):
            sl = slice(j, j + self.batch_size)
            T = int(mask[sl].sum(1).max())                     # trim padding to this sub-batch
            out += decode(model, ids[sl, :T], mask[sl, :T], latents[sl], max_new_tokens=max_new_tokens,
                          prompt_open_fences=[open_fence(it["prompt"]) for it in items[sl]])
        return out


class VLLMBackendDecoder:
    """ladir.vllm_decode.VLLMDecoder (prompt embeddings). Needs vLLM."""
    name = "vllm"

    def __init__(self, backbone_dir: str, gpu_memory_utilization: float = 0.45, **kw):
        from ladir.vllm_decode import VLLMDecoder
        self.impl = VLLMDecoder(backbone_dir, gpu_memory_utilization=gpu_memory_utilization, **kw)

    def decode(self, model, ids, mask, latents, items, max_new_tokens):
        return self.impl.decode(model, ids, mask, latents, max_new_tokens=max_new_tokens, temperature=0.0,
                                stop_at_code_block=True,
                                prompt_open_fences=[open_fence(it["prompt"]) for it in items])


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        import ladir.vllm_decode  # noqa: F401
        return True
    except Exception:
        return False


def default_backbone_dir(checkpoint: str) -> str:
    """Exported HF backbone: <ckpt>/backbone-hf (RL export), <ckpt_dir>-backbone-hf next to a local checkpoint,
    else $LADIR_BACKBONE_ROOT/<repo name>-backbone-hf (default outputs/backbones)."""
    if os.path.isdir(os.path.join(checkpoint, "backbone-hf")):
        return os.path.join(checkpoint, "backbone-hf")
    if os.path.isdir(checkpoint):
        return os.path.normpath(checkpoint) + "-backbone-hf"
    root = os.environ.get("LADIR_BACKBONE_ROOT", "outputs/backbones")
    return os.path.join(root, os.path.basename(os.path.normpath(checkpoint)) + "-backbone-hf")


# ----------------------------------------------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------------------------------------------
def _pad(tok, prompts, device):
    enc = [tok(p)["input_ids"] for p in prompts]
    T = max(map(len, enc))
    ids = torch.tensor([e + [tok.pad_token_id] * (T - len(e)) for e in enc], device=device)
    mask = torch.tensor([[1] * len(e) + [0] * (T - len(e)) for e in enc], device=device)
    return ids, mask


# The flow pass uses a float attention mask, so attention memory is O(batch * L^2). Cap batch * (CFG ? 2 : 1) * L_max^2
# per latent-sampling batch (default ~ 16 sequences of 600 tokens).
LATENT_ATTN_BUDGET = float(os.environ.get("LADIR_LATENT_ATTN_BUDGET", 16 * 600 ** 2))


def _latent_chunks(tok, items, max_batch, cfg_mult, num_latents):
    chunk, lmax = [], 0
    for it in items:
        L = len(tok(it["prompt"])["input_ids"]) + num_latents + 2
        new_lmax = max(lmax, L)
        if chunk and (len(chunk) >= max_batch or (len(chunk) + 1) * cfg_mult * new_lmax ** 2 > LATENT_ATTN_BUDGET):
            yield chunk
            chunk, new_lmax = [], L
        chunk.append(it)
        lmax = new_lmax
    if chunk:
        yield chunk


@torch.no_grad()
def generate_rows(model, items: List[dict], sampler_config, batch_size: int = 8, max_new_tokens: int = MAX_NEW_TOKENS,
                  decoder=None, latent_batch_size: Optional[int] = None) -> List[str]:
    """items: [{prompt, noise_seed}] -> raw texts (same order).

    Latents are sampled in batches of latent_batch_size (the expensive, CFG-doubled flow passes), then decoded by
    `decoder` (default HFDecoder(batch_size)). Per-item seeded noise makes the result independent of both batch sizes.
    """
    from ladir.sampling import sample_latents
    decoder = decoder or HFDecoder(batch_size)
    lbs = latent_batch_size or batch_size
    tok, device = model.tokenizer, model.bot_embed.device
    N, D = model.config.num_latents, model.config.latent_dim
    out = []
    for chunk in _latent_chunks(tok, items, lbs, 2 if sampler_config.cfg_scale > 1.0 else 1, N):
        ids, mask = _pad(tok, [it["prompt"] for it in chunk], device)
        noise = torch.stack([make_noise(it["noise_seed"], N, D) for it in chunk]).to(device)
        lat = sample_latents(model, ids, mask, sampler_config, noise=noise)
        out += decoder.decode(model, ids, mask, lat, chunk, max_new_tokens)
    return out


def generate_benchmark(model, bench: str, out_dir: str, sampler_config, meta: dict, batch_size: int = 8,
                       max_new_tokens: int = MAX_NEW_TOKENS, limit: Optional[int] = None,
                       task_ids: Optional[List[str]] = None, decoder=None, latent_batch_size: Optional[int] = None,
                       log=print) -> str:
    """Generate one greedy sample per task of one benchmark into out_dir/<bench>.jsonl (resumable). Returns the path."""
    b = canonical_name(bench)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, b + ".jsonl")
    _check_meta(os.path.join(out_dir, b + ".meta.json"), {**meta, "benchmark": b})
    problems = load_benchmark(b)
    if task_ids is not None:
        keep = set(task_ids)
        problems = [p for p in problems if p["task_id"] in keep]
    if limit:
        problems = problems[:limit]
    _repair_tail(path)
    done = {r["task_id"] for r in _read_rows(path)}
    tok = model.tokenizer
    todo = []
    for p in problems:
        if p["task_id"] not in done:
            row = {"task_id": p["task_id"], "sample_idx": 0, "prompt": p["prompt"],
                   "noise_seed": noise_seed(p["task_id"])}
            row.update({k: p.get(k) for k in GEN_FIELDS[b]})
            todo.append((len(tok(p["prompt"])["input_ids"]), row))
    todo.sort(key=lambda x: (-x[0], x[1]["task_id"]))   # longest first: OOM shows up early
    todo = [r for _, r in todo]
    decoder = decoder or HFDecoder(batch_size)
    lbs = latent_batch_size or batch_size
    log(f"[{b}] {len(problems)} tasks, {len(problems) - len(todo)} done, {len(todo)} to generate "
        f"(decode={decoder.name}, latent batch {lbs})")
    step = max(lbs, getattr(decoder, "batch_size", lbs))
    with open(path, "a") as f:
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            texts = generate_rows(model, chunk, sampler_config, batch_size, max_new_tokens,
                                  decoder=decoder, latent_batch_size=lbs)
            for row, text in zip(chunk, texts):
                f.write(json.dumps({**row, "raw": text}) + "\n")
            f.flush()
            log(f"[{b}] {min(i + step, len(todo))}/{len(todo)}")
    return path


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="HF repo id (mk322/LaDiR-Code-8B) or local checkpoint dir")
    ap.add_argument("--benchmarks", default="humaneval,mbpp,lcb_v6")
    ap.add_argument("--batch_size", type=int, default=8, help="HF decode batch size")
    ap.add_argument("--latent_batch_size", type=int, default=64, help="prompts per latent-sampling batch")
    ap.add_argument("--decode_backend", default="auto", choices=["auto", "hf", "vllm"],
                    help="auto = vllm if importable and on GPU, else hf")
    ap.add_argument("--vllm_backbone_dir", default=None,
                    help="exported HF backbone (default: <ckpt>-backbone-hf; exported on first use)")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45)
    ap.add_argument("--limit", type=int, default=None, help="first K tasks per benchmark (quick runs)")
    ap.add_argument("--out_root", default=os.path.join(REPO, "results"))
    ap.add_argument("--run_name", default=None, help="override <ckpt_name>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    return ap


def make_decoder(args, model):
    backend = args.decode_backend
    if backend == "auto":
        backend = "vllm" if (str(args.device).startswith("cuda") and vllm_available()) else "hf"
    if backend == "hf":
        return HFDecoder(args.batch_size)
    from ladir.vllm_decode import export_backbone
    bdir = args.vllm_backbone_dir or default_backbone_dir(args.checkpoint)
    export_backbone(model, bdir)          # no-op if already exported
    return VLLMBackendDecoder(bdir, gpu_memory_utilization=args.vllm_gpu_memory_utilization)


def run(args, model=None, sampler=None, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Load the checkpoint once (unless `model` is given) and generate every requested benchmark. Returns the output
    dir. `sampler` / `max_new_tokens` default to the fixed eval setting."""
    sc = sampler or sampler_config()
    out_dir = os.path.join(args.out_root, args.run_name or ckpt_name(args.checkpoint))
    if model is None:
        from ladir.checkpoint import load_ladir
        model = load_ladir(args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype))
    decoder = make_decoder(args, model)
    meta = {
        "checkpoint": args.checkpoint,
        "sampler": dict(sc.__dict__),
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy",
        "samples_per_task": 1,
        "batch_size": args.batch_size,
        "latent_batch_size": args.latent_batch_size,
        "decode_backend": decoder.name,
        "dtype": args.dtype,
        "git_sha": git_sha(),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
    }
    for bench in [b for b in args.benchmarks.split(",") if b]:
        generate_benchmark(model, bench, out_dir, sc, meta, batch_size=args.batch_size,
                           max_new_tokens=max_new_tokens, limit=args.limit, decoder=decoder,
                           latent_batch_size=args.latent_batch_size)
    print(out_dir)
    return out_dir


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    main()
