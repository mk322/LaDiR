"""Reasoning-block data for VAE training.

Raw items use the same format as SFT (instruction / reasoning_block / text_blocks, plus `language`); the VAE
reconstructs `reasoning_block.strip() + eos`. Items are tokenized once (optionally cached to a .pt file) and
collated with explicit, length-based attention masks, so a trailing EOS stays a real (trained) token even though
pad_token_id == eos_token_id.
"""

import json
import os
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset


def read_items(path: str) -> list:
    """JSON list or JSONL file."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def reasoning_texts(items: list, language: Optional[str] = "Python", min_chars: int = 10) -> List[str]:
    out = []
    for it in items:
        if language and it.get("language", language) != language:
            continue
        rb = (it.get("reasoning_block") or "").strip()
        if len(rb) > min_chars:
            out.append(rb)
    return out


def tokenize_texts(texts: List[str], tokenizer, max_tokens: int = 512, append_eos: bool = True,
                   overlength: str = "drop") -> List[List[int]]:
    """overlength: 'drop' (length counts the EOS) or 'truncate' (keeps the EOS as last token)."""
    eos = [tokenizer.eos_token_id] if append_eos else []
    out = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"] + eos
        if len(ids) > max_tokens:
            if overlength == "drop":
                continue
            ids = ids[:max_tokens - len(eos)] + eos
        out.append(ids)
    return out


class TokenizedDataset(Dataset):
    def __init__(self, ids: List[List[int]]):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i]}


class VAECollator:
    """Right-pads to the longest sequence. attention_mask comes from lengths, labels = ids with pads -> -100."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


def build_datasets(data_path: str, tokenizer, max_tokens: int = 512, language: Optional[str] = "Python",
                   val_size: int = 100, split_seed: Optional[int] = None, cache_path: Optional[str] = None,
                   overlength: str = "drop"):
    """Returns (train, val) TokenizedDatasets. split_seed None = last `val_size` examples are validation; otherwise
    shuffle with that seed first."""
    if cache_path and os.path.exists(cache_path):
        ids = torch.load(cache_path, weights_only=True)["ids"]
    else:
        ids = tokenize_texts(reasoning_texts(read_items(data_path), language), tokenizer, max_tokens,
                             overlength=overlength)
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save({"ids": ids, "source": data_path, "max_tokens": max_tokens}, cache_path)
    if split_seed is not None:
        ids = list(ids)
        random.Random(split_seed).shuffle(ids)
    val_size = min(val_size, max(len(ids) - 1, 0))
    return TokenizedDataset(ids[:len(ids) - val_size]), TokenizedDataset(ids[len(ids) - val_size:])
