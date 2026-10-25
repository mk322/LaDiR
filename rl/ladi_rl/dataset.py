"""RL dataset: parquet rows (id, prompt, tests, extra[, benchmark])."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

COLUMNS = ("task_id", "prompt", "tests", "extra", "group")


class LaDiRLDataset(Dataset):
    def __init__(self, files: Sequence[str] | str, max_samples: int = -1):
        files = [files] if isinstance(files, str) else list(files)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.rename(columns={"id": "task_id", "benchmark": "group"})
        missing = [c for c in ("task_id", "prompt", "tests", "extra") if c not in df.columns]
        if missing:
            raise ValueError(f"RL parquet missing columns {missing}")
        if "group" not in df.columns:
            df["group"] = ""
        df["group"] = df["group"].fillna("")
        if max_samples is not None and max_samples > 0:
            df = df.iloc[:max_samples]
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {"index": torch.tensor(i, dtype=torch.long), **{c: str(r[c]) for c in COLUMNS}}


def collate_fn(items: List[dict]) -> dict:
    out = {"index": torch.stack([it["index"] for it in items])}
    for c in COLUMNS:
        out[c] = np.array([it[c] for it in items], dtype=object)
    return out
