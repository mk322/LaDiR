"""Download the RL data from the Hub: python -m rl.data.fetch_hf --out_dir data/ladi_rl"""

import argparse
import glob
import os

import pandas as pd
from huggingface_hub import snapshot_download

REPO = "mk322/LaDi-RL-Code-Data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    src = snapshot_download(a.repo, repo_type="dataset")
    os.makedirs(a.out_dir, exist_ok=True)
    for split in ("train", "validation"):
        files = sorted(glob.glob(os.path.join(src, "**", f"{split}*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no {split} parquet in {a.repo}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.to_parquet(os.path.join(a.out_dir, f"{split}.parquet"), index=False)
        print(f"{split}: {len(df)} rows -> {a.out_dir}/{split}.parquet")


if __name__ == "__main__":
    main()
