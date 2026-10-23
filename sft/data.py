"""SFT data: pretokenization CLI, dataset (+ optional precomputed latents) and collator.

Raw data: a JSON list of items with
    original_messages (or messages): [{"role": "HUMAN" | "user", "content": <instruction>}, ...]
    reasoning_block: str     -> encoded into latents (VAE or in-LLM encoder)
    text_blocks:     str     -> answer, trained with CE
    language:        str     (only items of this language are kept, default "Python")
Items whose reasoning_block / text_blocks are shorter than 10 characters (after strip) or that have no instruction
are dropped.

Pretokenized layout (directory):
    meta.json                 tokenizer, max lengths, per-shard sample counts, content fingerprint
    data_00000.pt ...         {prompt,reasoning,answer}_{ids,offsets} (flat int32 ids + int64 offsets), source_index
Truncation keeps the head of each field; reasoning and answer are truncated to max_len - 1 tokens and then EOS is
appended, so every reasoning/answer sequence ends with EOS.

Usage:
    python -m sft.data --input /path/sft_data.json --out_dir /path/ling_coder_tok \
        --tokenizer Qwen/Qwen3-8B-Base
"""

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

FIELDS = ("prompt", "reasoning", "answer")
FORMAT_VERSION = 1
META = "meta.json"


# ----------------------------------------------------------------------------------------------------------------
# raw json -> examples
# ----------------------------------------------------------------------------------------------------------------
def _instruction(item: dict) -> str:
    for key in ("original_messages", "messages"):
        for msg in item.get(key) or []:
            if msg.get("role") in ("HUMAN", "user"):
                return msg.get("content") or ""
    return ""


def parse_raw(items: List[dict], language: Optional[str] = "Python"):
    """Yield (source_index, instruction, reasoning_block, text_blocks) for usable items."""
    for i, item in enumerate(items):
        if language and item.get("language", "") != language:
            continue
        reasoning, answer = (item.get("reasoning_block") or "").strip(), (item.get("text_blocks") or "").strip()
        if len(reasoning) < 10 or len(answer) < 10:
            continue
        instruction = _instruction(item)
        if not instruction:
            continue
        yield i, instruction, reasoning, answer


def tokenize_examples(tokenizer, examples, max_prompt_length: int, max_reasoning_length: int,
                      max_answer_length: int, batch_size: int = 1024):
    """examples: list of (source_index, instruction, reasoning, answer) -> dict of lists of id lists."""
    eos = tokenizer.eos_token_id
    out = {"source_index": [], "prompt": [], "reasoning": [], "answer": []}
    for s in range(0, len(examples), batch_size):
        chunk = examples[s:s + batch_size]
        enc = [tokenizer([ex[k] for ex in chunk], add_special_tokens=False)["input_ids"] for k in (1, 2, 3)]
        for j, ex in enumerate(chunk):
            out["source_index"].append(ex[0])
            out["prompt"].append(enc[0][j][:max_prompt_length])
            out["reasoning"].append(enc[1][j][:max_reasoning_length - 1] + [eos])
            out["answer"].append(enc[2][j][:max_answer_length - 1] + [eos])
    return out


def _pack(seqs: List[List[int]]):
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.int64)
    offsets = torch.zeros(len(seqs) + 1, dtype=torch.int64)
    offsets[1:] = lens.cumsum(0)
    ids = torch.tensor([t for s in seqs for t in s], dtype=torch.int32)
    return ids, offsets


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def pretokenize(input_path: str, out_dir: str, tokenizer_name: str, max_prompt_length: int = 512,
                max_reasoning_length: int = 1024, max_answer_length: int = 1024, language: Optional[str] = "Python",
                num_samples: Optional[int] = None, shard_size: int = 10000, tokenizer=None) -> dict:
    from ladir.tokenizer import load_tokenizer
    tokenizer = tokenizer or load_tokenizer(tokenizer_name)
    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)
    examples = list(parse_raw(items, language))
    if num_samples is not None:
        examples = examples[:num_samples]
    if not examples:
        raise ValueError(f"no usable examples in {input_path}")
    os.makedirs(out_dir, exist_ok=True)
    shards, digests = [], []
    for k, s in enumerate(range(0, len(examples), shard_size)):
        tok = tokenize_examples(tokenizer, examples[s:s + shard_size], max_prompt_length, max_reasoning_length,
                                max_answer_length)
        shard = {"source_index": torch.tensor(tok["source_index"], dtype=torch.int64)}
        for name in FIELDS:
            shard[f"{name}_ids"], shard[f"{name}_offsets"] = _pack(tok[name])
        fname = f"data_{k:05d}.pt"
        torch.save(shard, os.path.join(out_dir, fname))
        shards.append({"file": fname, "num_samples": len(tok["source_index"])})
        digests.append(_sha256(os.path.join(out_dir, fname)))
    meta = {
        "format_version": FORMAT_VERSION,
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_prompt_length": max_prompt_length,
        "max_reasoning_length": max_reasoning_length,
        "max_answer_length": max_answer_length,
        "language": language,
        "num_samples": sum(sh["num_samples"] for sh in shards),
        "shards": shards,
        "source": {"path": os.path.abspath(input_path), "num_items": len(items)},
        "fingerprint": hashlib.sha256("".join(digests).encode()).hexdigest(),
    }
    with open(os.path.join(out_dir, META), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def read_meta(directory: str) -> dict:
    with open(os.path.join(directory, META)) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------------------------
class TokenShards:
    """Random access over pretokenized shards (memory-mapped)."""

    def __init__(self, directory: str):
        self.meta = read_meta(directory)
        if self.meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{directory}: unsupported format_version {self.meta.get('format_version')}")
        self.shards = [torch.load(os.path.join(directory, sh["file"]), mmap=True, weights_only=True)
                       for sh in self.meta["shards"]]
        counts = torch.tensor([sh["num_samples"] for sh in self.meta["shards"]])
        self.starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])

    def __len__(self):
        return int(self.starts[-1])

    def locate(self, idx: int):
        k = int(torch.searchsorted(self.starts, idx, right=True)) - 1
        return k, idx - int(self.starts[k])

    def get(self, idx: int) -> Dict[str, torch.Tensor]:
        k, j = self.locate(idx)
        sh = self.shards[k]
        out = {}
        for name in FIELDS:
            o = sh[f"{name}_offsets"]
            out[f"{name}_input_ids"] = sh[f"{name}_ids"][int(o[j]):int(o[j + 1])].long()
        return out


class LatentShards:
    """Precomputed latents aligned 1:1 with the token shards (see sft/precompute_latents.py)."""

    def __init__(self, directory: str, token_meta: dict):
        with open(os.path.join(directory, META)) as f:
            self.meta = json.load(f)
        if self.meta["data_fingerprint"] != token_meta["fingerprint"]:
            raise ValueError(f"{directory} was computed for different pretokenized data "
                             f"({self.meta['data_fingerprint']} != {token_meta['fingerprint']})")
        counts = [sh["num_samples"] for sh in token_meta["shards"]]
        self.shards = []
        for k, n in enumerate(counts):
            path = os.path.join(directory, latent_shard_name(k))
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing latent shard {path} (precompute_latents incomplete?)")
            sh = torch.load(path, mmap=True, weights_only=True)
            if sh["latents"].shape[0] != n:
                raise ValueError(f"{path}: {sh['latents'].shape[0]} latents for {n} samples")
            self.shards.append(sh["latents"])


def latent_shard_name(k: int) -> str:
    return f"latents_{k:05d}.pt"


class SFTDataset(Dataset):
    def __init__(self, train_dir: str, latents_dir: Optional[str] = None, num_samples: Optional[int] = None,
                 seed: int = 0, expected_lengths: Optional[Dict[str, int]] = None,
                 expected_vocab_size: Optional[int] = None):
        self.tokens = TokenShards(train_dir)
        meta = self.tokens.meta
        for key, value in (expected_lengths or {}).items():
            if meta[key] != value:
                raise ValueError(f"{train_dir} was pretokenized with {key}={meta[key]}, config expects {value}")
        if expected_vocab_size is not None and meta["vocab_size"] != expected_vocab_size:
            raise ValueError(f"{train_dir}: tokenizer vocab {meta['vocab_size']} != {expected_vocab_size}")
        self.latents = LatentShards(latents_dir, meta) if latents_dir else None
        n = len(self.tokens)
        if num_samples is None:
            self.indices = None
        else:
            if num_samples > n:
                raise ValueError(f"data.num_samples={num_samples} > {n} available samples")
            g = torch.Generator().manual_seed(seed)
            self.indices = torch.randperm(n, generator=g)[:num_samples].sort().values

    def __len__(self):
        return len(self.tokens) if self.indices is None else len(self.indices)

    def __getitem__(self, i: int):
        idx = i if self.indices is None else int(self.indices[i])
        item = self.tokens.get(idx)
        if self.latents is not None:
            k, j = self.tokens.locate(idx)
            item["gt_latents"] = self.latents.shards[k][j].clone()
        return item


class SFTCollator:
    """Right-pads ids and builds attention masks from sequence lengths (never from `ids != pad`: Qwen's pad token is
    its EOS token, and the trailing EOS must be attended to and trained)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: List[dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        for name in FIELDS:
            seqs = [it[f"{name}_input_ids"] for it in items]
            lens = torch.tensor([len(s) for s in seqs])
            if (lens == 0).any():
                raise ValueError(f"empty {name} sequence in batch")
            ids = pad_sequence(seqs, batch_first=True, padding_value=self.pad_token_id)
            batch[f"{name}_input_ids"] = ids
            batch[f"{name}_attention_mask"] = (torch.arange(ids.shape[1])[None] < lens[:, None]).long()
        if "gt_latents" in items[0]:
            batch["gt_latents"] = torch.stack([it["gt_latents"] for it in items])
        return batch


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Pretokenize raw LaDiR SFT json into sharded .pt files")
    p.add_argument("--input", required=True, help="raw json list")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_reasoning_length", type=int, default=1024)
    p.add_argument("--max_answer_length", type=int, default=1024)
    p.add_argument("--language", default="Python", help="'' keeps all languages")
    p.add_argument("--num_samples", type=int, default=None, help="keep only the first N usable items")
    p.add_argument("--shard_size", type=int, default=10000)
    a = p.parse_args(argv)
    meta = pretokenize(a.input, a.out_dir, a.tokenizer, a.max_prompt_length, a.max_reasoning_length,
                       a.max_answer_length, a.language or None, a.num_samples, a.shard_size)
    print(f"wrote {meta['num_samples']} samples in {len(meta['shards'])} shards to {a.out_dir} "
          f"(fingerprint {meta['fingerprint'][:12]})")


if __name__ == "__main__":
    main()
