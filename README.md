<div align="center">

<img src="assets/banner.svg" alt="LaDiR: Latent Diffusion Reasoner" width="100%">

[![arXiv](https://img.shields.io/badge/arXiv-2510.04573-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2510.04573)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-7c3aed.svg?style=flat-square)](https://arxiv.org/abs/2510.04573)
[![Model](https://img.shields.io/badge/🤗%20Model-LaDiR--Code--8B-ffd21e.svg?style=flat-square)](https://huggingface.co/mk322/LaDiR-Code-8B)

</div>

**LaDiR** lets an LLM reason in a continuous latent space. A VAE compresses a reasoning chain into 64 latent tokens;
the LLM generates this latent block with flow matching, then decodes the answer conditioned on it.

## Install

```bash
pip install -e .
```

## Evaluate

HumanEval(+), MBPP(+) and LiveCodeBench v6:

```bash
python -m eval.generate --checkpoint mk322/LaDiR-Code-8B
python -m eval.score results/LaDiR-Code-8B
```

## Train

```bash
export LADIR_DATA=/path/to/data LADIR_OUTPUT=/path/to/runs LADIR_RAW_JSON=/path/to/sft.json

# 1. reasoning VAE (skip to reuse the VAE inside mk322/LaDiR-Code-8B)
scripts/train_vae.sh data.path=$LADIR_RAW_JSON
export LADIR_VAE=outputs/vae/vae_4b_8b_64/weights/step_<N>.safetensors

# 2. VAE latents → stage 1 (projectors) → stage 2 (full model)
scripts/train_sft_stagewise.sh
```

## Citation

```bibtex
@inproceedings{kang2026ladir,
  title     = {LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning},
  author    = {Kang, Haoqiang and Zhang, Yizhe and Kuang, Nikki Lijing and Majamaki, Nicklas and Jaitly, Navdeep and Ma, Yi-An and Qin, Lianhui},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.04573}
}
```
