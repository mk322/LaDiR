"""
Example usage script for the VAE-based diffusion model.
This script demonstrates how to use the model for training and inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vae.model_vae import VAE
from vae.vae_args import parse_args
from peft import LoraConfig
import argparse


def example_vae_training():
    """Example of how to train the VAE model."""
    print("=== VAE Training Example ===")
    
    # Parse arguments (you would typically pass these via command line)
    args = argparse.Namespace(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        output_dir="checkpoints/vae_example",
        input_type="full_format",
        test_size=10,
        max_steps=1000,
        learning_rate=2e-5,
        lora_r=128,
        lora_alpha=32,
        lora_dropout=0.05,
        beta=1e-5,
        train=True,
        notes="Example VAE training"
    )
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("LoRA configuration created successfully")
    print(f"Target model: {args.model_name_or_path}")
    print(f"Output directory: {args.output_dir}")


def example_model_loading():
    """Example of how to load a pre-trained model."""
    print("\n=== Model Loading Example ===")
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer loaded: {model_name}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        print(f"Model loaded: {model_name}")
        
        # Example text processing
        text = "What is 2 + 2?"
        tokens = tokenizer(text, return_tensors="pt")
        print(f"Tokenized text: {tokens}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: You may need to authenticate with Hugging Face Hub")


def example_config_usage():
    """Example of how to use configuration files."""
    print("\n=== Configuration Example ===")
    
    # Example configuration
    config = {
        "model": {
            "llm_model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            "use_flash_attn": True,
            "freeze_tokenizer": True,
        },
        "ae": {
            "lora_r": 512,
            "lora_alpha": 256,
            "lora_dropout": 0.05,
            "mean_compression_rate": 1,
            "fixed_mem_size": 3,
            "icae_ckpt": "checkpoints/vae_model/model.safetensors",
        },
        "trainer": {
            "per_device_train_batch_size": 64,
            "learning_rate": 1e-4,
            "max_steps": 5000,
            "save_strategy": "steps",
            "save_steps": 1000,
        }
    }
    
    print("Example configuration structure:")
    for section, params in config.items():
        print(f"  {section}:")
        for key, value in params.items():
            print(f"    {key}: {value}")


def main():
    """Main example function."""
    print("VAE-based Diffusion Model - Example Usage")
    print("=" * 50)
    
    example_vae_training()
    example_model_loading()
    example_config_usage()
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Prepare your dataset in JSONL format")
    print("2. Update configuration files in configs/")
    print("3. Run VAE training: python vae/train_vae.py [args]")
    print("4. Run diffusion training: python train.py configs/your_config.yaml")


if __name__ == "__main__":
    main()
