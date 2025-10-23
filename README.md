# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning

Official repository for the paper:  
**[LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)**  

---

## 🧠 Overview

**LaDiR (Latent Diffusion Reasoner)** introduces a new reasoning framework that unifies the expressiveness of **continuous latent representations** with the **iterative refinement capability** of diffusion models for large language models (LLMs).

Instead of generating reasoning chains autoregressively, LaDiR performs **latent diffusion over thought tokens**, enabling:

- Iterative semantic self-refinement  
- Diverse parallel reasoning trajectories  
- A flexible trade-off between accuracy and test-time compute  

---


## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Training the VAE Model

1. **Prepare your dataset** in JSONL format with the following structure:
   ```json
   {"input": "question text", "output": "reasoning chain"}
   ```

2. **Configure training parameters** in `configs/cd_formal_8B_VAE_conn.yaml`

3. **Run VAE training**:
   ```bash
   cd vae
   bash ..scripts/train_vae.sh
   ```

### Training the Diffusion Model
   ```bash
   bash scripts/train_vae.sh
   ```

## ⚙️ Configuration

The model can be configured through YAML files in the `configs/` directory. Key parameters include:

- **Model**: Base language model path, LoRA configuration
- **Training**: Learning rate, batch size, number of steps
- **VAE**: Compression rate, memory size, beta for KL loss
- **Dataset**: Training file paths, data processing options

---

If you find this work useful, please consider citing:

```bibtex
@article{kang2025ladir,
  title={LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning},
  author={Kang, Haoqiang and Zhang, Yizhe and Kuang, Nikki Lijing and Majamäki, Nicklas and Jaitly, Navdeep and Ma, Yi-An and Qin, Lianhui},
  journal={arXiv preprint arXiv:2510.08558},
  year={2025}
}
