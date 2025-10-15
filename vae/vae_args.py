import argparse
from dataclasses import dataclass, field
import json
import transformers
from typing import Optional


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
        default=False,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )
    beta: float = field(
        default=0.3,
        metadata={"help": "KL loss beta"}
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
        default=1,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    report_to: str = field(
        default="wandb"
    )
    max_steps: int = field(
        default=5000
    )
    save_strategy: str = field(
        default="no"
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
        default="epoch"
    )
    num_train_epochs: int = field(
        default=1
    )

def parse_args():
    """Parses command-line arguments and initializes Model & Training arguments dynamically."""
    parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model with LoRA.")

    # Add arguments dynamically
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Pretrained model path.")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate.")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--beta", type=float, default=0.3, help="Beta value for KL loss")

    # Training Arguments (Automatically Converted to `TrainingArguments`)
    parser.add_argument("--icae_ckpt", type=str, help="Directory to save checkpoints and logs.")
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
    parser.add_argument("--notes", type=str, help="Additional notes for the run.")
    parser.add_argument("--eval_interval", type=int, default=1000, help="evaluation interval")
    parser.add_argument("--run_name", type=str, help="Name of the run.")
    parser.add_argument("--guidance", type=float, help="CFG guidance at inference-time")
    parser.add_argument("--ckpt_path", type=str, help="Name of the CKPT.")

    args = parser.parse_args()

    # Convert JSON string arguments
    args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)

    skip_list = ['input_type', 'test_size', 'notes', 'eval_interval', 'icae_ckpt', "guidance", "ckpt_path"]
    
    model_args = ModelArguments(**{k: v for k, v in vars(args).items() if k in ModelArguments.__annotations__})
    training_args = TrainingArguments(**{k: v for k, v in vars(args).items() if k not in ModelArguments.__annotations__ and k not in skip_list})

    return model_args, training_args, args
    
