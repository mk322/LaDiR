"""RL model dir for verl-omni next to a LaDiR checkpoint: links the weights, adds model_index.json, the LaDiR config
and a backbone/ config + tokenizer for the answer vLLM replica.

    python -m rl.ladi_rl.model_dir mk322/LaDiR-Code-8B outputs/rl_model
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from typing import Optional

from ladir.modeling.model import LaDiRConfig

from .common import ARCHITECTURE


def read_model_dir(ckpt_dir: str, config: Optional[LaDiRConfig] = None):
    """-> (LaDiRConfig, backbone HF config or None, tokenizer path)."""
    from transformers import AutoConfig

    cfg = config
    cfg_path = os.path.join(ckpt_dir, "ladi_config.json")
    if cfg is None and os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            raw = json.load(f)
        known = {f.name for f in fields(LaDiRConfig)}
        cfg = LaDiRConfig(**{k: v for k, v in raw.items() if k in known})
    cfg = cfg or LaDiRConfig()
    backbone_config = None
    bb = cfg.backbone
    if not os.path.isabs(bb) and os.path.isdir(os.path.join(ckpt_dir, bb)):
        bb = os.path.join(ckpt_dir, bb)
    if os.path.isdir(bb):
        backbone_config = AutoConfig.from_pretrained(bb)
    has_tok = any(os.path.isfile(os.path.join(ckpt_dir, n)) for n in ("tokenizer.json", "tokenizer_config.json"))
    tok_path = ckpt_dir if has_tok else cfg.backbone
    return cfg, backbone_config, tok_path


def prepare_model_dir(ckpt: str, out: str, backbone: str = "Qwen/Qwen3-8B-Base", ladi_config: dict = None) -> str:
    from transformers import AutoConfig

    from ladir.checkpoint import HEADS_FILE, is_split, resolve
    from ladir.tokenizer import load_tokenizer

    src = os.path.abspath(resolve(ckpt))
    os.makedirs(out, exist_ok=True)
    names = ["backbone-hf", HEADS_FILE] if is_split(src) else ["model.safetensors"]
    for name in names:
        link, target = os.path.join(out, name), os.path.join(src, name)
        if os.path.lexists(link):
            if os.path.realpath(link) != os.path.realpath(target):
                raise FileExistsError(f"{link} points elsewhere")
        else:
            os.symlink(target, link)
    tok = load_tokenizer(backbone)
    tok.save_pretrained(out)
    bb_cfg = AutoConfig.from_pretrained(backbone)
    bb_cfg.vocab_size = len(tok)
    bb_cfg.architectures = bb_cfg.architectures or ["Qwen3ForCausalLM"]
    bb_cfg.save_pretrained(os.path.join(out, "backbone"))
    tok.save_pretrained(os.path.join(out, "backbone"))
    cfg = {**LaDiRConfig().__dict__, **(ladi_config or {}), "backbone": "backbone", "source_checkpoint": ckpt}
    with open(os.path.join(out, "ladi_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "model_index.json"), "w") as f:
        json.dump({"_class_name": ARCHITECTURE}, f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--backbone", default="Qwen/Qwen3-8B-Base")
    a = ap.parse_args(argv)
    print(prepare_model_dir(a.checkpoint, a.out, a.backbone))


if __name__ == "__main__":
    main()
