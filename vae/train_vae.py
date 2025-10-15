import transformers
import torch
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    LoraConfig,
)
import argparse
import json

from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding, train_model
from model_vae import VAE

from data_vae import load_data
import os
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    lora_r: int = field(
        default=128,
        metadata={"help": "lora rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )
    beta: float = field(
        default=1e-5,
        metadata={"help": "Beta for VAE's KL loss"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_mem_size: int = field(
        default=3,
        metadata={"help": "Enabling the fixed mem size."},
    )
    ignore_data_skip: bool = field(
        default=True,
        metadata={"help": "Enabling the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=128*4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    report_to: str = field(
        default="wandb"
    )
    max_steps: int = field(
        default=15000
    )
    save_strategy: str = field(
        default="steps"
    )
    save_steps: int = field(
        default=2500
    )
    logging_strategy: str = field(
        default="steps"
    )
    logging_steps: int = field(
        default=1
    )
    learning_rate: float = field(
        default=2.5e-5
    )
    lr_scheduler_type: str = field(
        default="cosine"
    )
    lr_scheduler_kwargs: dict = field(default_factory=dict)
    warmup_steps: int = field(
        default=0
    )
    weight_decay: float = field(
        default=0.0
    )
    eval_strategy: str = field(
        default="no"
    )
    num_train_epochs: int = field(
        default=1
    )
    bf16: bool = field(
        default=True
    )

def main(model_args, training_args, args, notes):    

    train_dataset, eval_dataset, lines = load_data(args.test_size, args.input_type)
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Loading model...")
    model = VAE(model_args, training_args, lora_config).to("cuda")
    model = model.to(torch.bfloat16)

    print("Model loaded successfully...")
    
    memory_size = training_args.fixed_mem_size
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    print("Tokenizing train/eval datasets...")

    train_fn_kwargs = {
        "tokenizer": model.tokenizer, 
        "model_max_length": model.training_args.model_max_length,
        "mem_size": model.mem_size,
        "min_tokens_for_lm": model.training_args.min_tokens_for_lm,
        "mean_compression_rate": model.mean_compression_rate,
        "add_special_token_for_lm": model.training_args.add_special_token_for_lm,
        "leave_tokens_for_lm": model.training_args.leave_tokens_for_lm,
        "ae_token_id": model.ae_token_id,
        "eos_id": model.eos_id,
        #"lm_token_id": model.lm_token_id,
        "mem": MEM_TOKENS, 
        "input_type": args.input_type, 
        "lm_ratio": training_args.lm_ratio
    }

    eval_fn_kwargs = {
        "tokenizer": model.tokenizer, 
        "model_max_length": model.training_args.model_max_length,
        "mem_size": model.mem_size,
        "min_tokens_for_lm": model.training_args.min_tokens_for_lm,
        "mean_compression_rate": model.mean_compression_rate,
        "add_special_token_for_lm": model.training_args.add_special_token_for_lm,
        "leave_tokens_for_lm": model.training_args.leave_tokens_for_lm,
        "ae_token_id": model.ae_token_id,
        "eos_id": model.eos_id,
        #"lm_token_id": model.lm_token_id,
        "mem": MEM_TOKENS, 
        "input_type": args.input_type, 
    }
    
    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, num_proc=112, batch_size=64, fn_kwargs=train_fn_kwargs)
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, num_proc=112, batch_size=64, fn_kwargs=eval_fn_kwargs)
    print("Finished tokenizing train/eval datasets...")

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)

    print("Training model...")
    train_model(
        args,
        notes,
        model, 
        train_dataset, 
        eval_dataset, 
        model_args,
        training_args, 
        lines,
        data_collator,
    )
    print("Finished training...")

def parse_args():
    """Parses command-line arguments and initializes Model & Training arguments dynamically."""
    parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model with LoRA.")

    # Add arguments dynamically
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model path.")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate.")

    # Training Arguments (Automatically Converted to `TrainingArguments`)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--input_type", type=str, required=True, help="Can be 'cot_only', 'full_format', or 'q_q_only'.")
    parser.add_argument("--test_size", type=int, default=0.1, help="Test dataset split ratio.")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--lr_scheduler_kwargs", type=str, default="{}", help="JSON string of LR scheduler kwargs.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps.")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer type.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay coefficient.")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Saving steps.")
    parser.add_argument("--notes", type=str, help="Additional notes for the run.")
    parser.add_argument("--eval_interval", type=int, default=1000, help="evaluation interval")
    parser.add_argument("--run_name", type=str, help="Name of the run.")
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="Distributed Data Parallel backend.")
    parser.add_argument("--fsdp", type=str, default="hybrid_shard auto_wrap", help="FSDP mode/configuration.")
    parser.add_argument("--beta", type=float, default=1e-5, help="Beta value for KL loss")

    parser.add_argument(
        "--fsdp_config",
        type=str,
        default='{"backward_prefetch": "backward_pre", "forward_prefetch": true, "cpu_ram_efficient_loading": true, "sync_module_states": true, "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"], "use_orig_params": true, "activation_checkpointing": true}',
        help="FSDP configuration in JSON format."
    )

    args = parser.parse_args()

    # Convert JSON string arguments
    args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
    args.fsdp_config = json.loads(args.fsdp_config)

    skip_list = ['input_type', 'test_size', 'notes', 'eval_interval']
    
    model_args = ModelArguments(**{k: v for k, v in vars(args).items() if k in ModelArguments.__annotations__})
    training_args = TrainingArguments(**{k: v for k, v in vars(args).items() if k not in ModelArguments.__annotations__ and k not in skip_list})

    return model_args, training_args, args
    
if __name__ == "__main__":
    model_args, training_args, args = parse_args()
    main(model_args, training_args, args, args.notes)