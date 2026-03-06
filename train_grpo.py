"""
GRPO training entry point for LaDiR.

Loads a pre-trained SFT diffusion checkpoint and fine-tunes it using
Group Relative Policy Optimization with exact-match reward signals.

Supports FSDP for multi-GPU training, matching the SFT training setup.
"""

import os
import sys
import copy
import functools
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    StateDictType,
)
from torch.optim import AdamW
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from omegaconf import OmegaConf as om
from safetensors.torch import load_file
from peft import LoraConfig

from dataset import ThoughtDataset, ThoughtDataCollator
from model import LMFusionModel, freeze_module
from vae.model_vae import VAE
from vae.vae_args import parse_args
from reward import ExactMatchReward
from grpo_trainer import GRPODiffusionTrainer


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_fsdp_wrap_policy():
    """Auto-wrap policy targeting LlamaDecoderLayer, same as SFT config."""
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )


def get_fsdp_mixed_precision():
    """bf16 mixed precision policy matching SFT's bf16=True."""
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def wrap_model_fsdp(model, device_id, sync_module_states=True, ignored_modules=None):
    """
    Wrap a model with FSDP using the same config as SFT training:
    - full_shard strategy
    - auto_wrap on LlamaDecoderLayer
    - use_orig_params=True (required for optimizer param groups)
    - backward_prefetch for overlap
    - bf16 mixed precision
    - ignored_modules: submodules to exclude from FSDP sharding
      (e.g. frozen VAE that is not used in the GRPO forward path)
    """
    return FSDP(
        model,
        auto_wrap_policy=get_fsdp_wrap_policy(),
        mixed_precision=get_fsdp_mixed_precision(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True,
        limit_all_gathers=True,
        ignored_modules=ignored_modules,
    )


def save_fsdp_checkpoint(model, path, rank0_only=True):
    """Save FSDP model checkpoint by gathering full state dict to rank 0."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if is_rank0():
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(state_dict, path)
        print(f"Saved checkpoint to {path}")
    if dist.is_initialized():
        dist.barrier()


def load_model_and_vae(cfg, device):
    """
    Load VAE, text LLaMA, and LMFusionModel.
    The VAE is loaded on the local device and kept outside FSDP.
    The LMFusionModel is returned un-wrapped (FSDP wrapping happens in main).
    """
    # Load frozen VAE
    ae_lora_config = LoraConfig(
        r=cfg.ae.lora_r,
        lora_alpha=cfg.ae.lora_alpha,
        lora_dropout=cfg.ae.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    ae_model_args, ae_training_args, ae_args = parse_args()
    ae = VAE(ae_model_args, ae_training_args, ae_lora_config)
    if is_rank0():
        print(f"Loading VAE checkpoint from {cfg.ae.icae_ckpt}")
    state_dict = load_file(cfg.ae.icae_ckpt)
    if "state_dict" in state_dict:
        ae.load_state_dict(state_dict["state_dict"], strict=False)
    else:
        ae.load_state_dict(state_dict, strict=False)
    freeze_module(ae)
    ae = ae.to(device, dtype=torch.bfloat16)
    ae.eval()

    # Load text LLaMA
    TEXT_LLAMA_PATH = cfg.model.llm_model_name_or_path
    text_llama_config = AutoConfig.from_pretrained(
        TEXT_LLAMA_PATH, use_flash_attention=False, _flash_attn_2_enabled=False
    )
    text_llama = AutoModelForCausalLM.from_pretrained(
        TEXT_LLAMA_PATH, config=text_llama_config, torch_dtype=torch.bfloat16
    )

    # Setup tokenizer with special tokens
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_LLAMA_PATH)
    text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
    for special_token in ["<tht_s>", "<tht>", "</tht_s>", "<timestep>"]:
        text_tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    text_tokenizer.bot_token_id = text_tokenizer.convert_tokens_to_ids("<tht_s>")
    text_tokenizer.tht_token_id = text_tokenizer.convert_tokens_to_ids("<tht>")
    text_tokenizer.eot_token_id = text_tokenizer.convert_tokens_to_ids("</tht_s>")
    text_tokenizer.time_token_id = text_tokenizer.convert_tokens_to_ids("<timestep>")
    text_tokenizer.pad_token_id = text_tokenizer.eos_token_id

    # Build LMFusionModel on CPU first (FSDP will move to device)
    model = LMFusionModel(
        text_llama=text_llama,
        thought_llama=None,
        autoencoder=ae,
        model_config=cfg,
        tokenizer=text_tokenizer,
        hidden_dim=text_llama_config.hidden_size,
        freeze_text=False,
    ).to(dtype=torch.bfloat16)

    return model, ae, text_tokenizer


def main(cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_rank0():
        print(f"GRPO Training - run name: {cfg.run_name}")

    # 1. Load model and VAE (model on CPU, VAE on device)
    model, ae, text_tokenizer = load_model_and_vae(cfg, device)

    # 2. Load SFT checkpoint BEFORE FSDP wrapping
    sft_ckpt = cfg.sft_checkpoint
    if is_rank0():
        print(f"Loading SFT checkpoint from {sft_ckpt}")
    if sft_ckpt.endswith(".safetensors"):
        sft_state = load_file(sft_ckpt)
    else:
        sft_state = torch.load(sft_ckpt, map_location="cpu", weights_only=False)
        if "state_dict" in sft_state:
            sft_state = sft_state["state_dict"]
        elif "model_state_dict" in sft_state:
            sft_state = sft_state["model_state_dict"]
    missing, unexpected = model.load_state_dict(sft_state, strict=False)
    if is_rank0():
        if missing:
            print(f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    # 3. Create frozen reference model BEFORE FSDP wrapping
    # Deep copy while still on CPU to avoid GPU OOM
    if is_rank0():
        print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    freeze_module(ref_model)
    ref_model.eval()

    # 4. Ensure autoencoder stays frozen in policy model
    freeze_module(model.autoencoder)

    # 5. Wrap both models with FSDP
    # sync_module_states=True ensures rank 0's loaded weights are broadcast to all ranks
    # Exclude the frozen autoencoder from FSDP sharding — it's not used in the
    # GRPO forward path (mode="velocity") and sharding it wastes communication.
    if is_rank0():
        print("Wrapping models with FSDP...")
    model = wrap_model_fsdp(
        model, device_id=local_rank, sync_module_states=True,
        ignored_modules=[model.autoencoder],
    )
    ref_model = wrap_model_fsdp(
        ref_model, device_id=local_rank, sync_module_states=True,
        ignored_modules=[ref_model.autoencoder],
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_rank0():
        print(f"Total params: {n_params:.2e}, Trainable: {n_trainable:.2e}")

    # 6. Create optimizer AFTER FSDP wrapping
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg.grpo.learning_rate,
        betas=(cfg.grpo.get("adam_beta1", 0.9), cfg.grpo.get("adam_beta2", 0.95)),
        weight_decay=cfg.grpo.get("weight_decay", 0.02),
    )

    # 7. Create reward function
    reward_fn = ExactMatchReward()

    # 8. Create dataset and dataloader with DistributedSampler
    train_dataset = ThoughtDataset(
        text_tokenizer,
        cfg.dataset.train_file,
    )
    data_collator = ThoughtDataCollator(pad_token_id=text_tokenizer.pad_token_id)

    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.grpo.get("batch_size", 4),
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=cfg.grpo.get("dataloader_num_workers", 4),
        pin_memory=True,
    )

    if is_rank0():
        print(f"Dataset size: {len(train_dataset)}")
        print(f"GRPO config: group_size={cfg.grpo.group_size}, "
              f"denoise_steps={cfg.grpo.num_denoise_steps}, "
              f"ppo_epochs={cfg.grpo.num_ppo_epochs}, "
              f"lr={cfg.grpo.learning_rate}")

    # 9. Create GRPO trainer (receives already-wrapped models and optimizer)
    trainer = GRPODiffusionTrainer(
        model=model,
        ref_model=ref_model,
        autoencoder=ae,
        reward_fn=reward_fn,
        tokenizer=text_tokenizer,
        optimizer=optimizer,
        cfg=cfg,
    )

    # 10. Training loop
    num_epochs = cfg.grpo.get("num_epochs", 10)
    save_steps = cfg.grpo.get("save_steps", 500)
    log_steps = cfg.grpo.get("log_steps", 1)
    output_dir = cfg.grpo.get("output_dir", f"ckpt/{cfg.run_name}_grpo")
    if is_rank0():
        os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            # Move tensors to device
            batch["input_ids_q"] = batch["input_ids_q"].to(device)
            batch["output_ids"] = batch["output_ids"].to(device)

            result = trainer.train_step(batch)
            global_step += 1

            if global_step % log_steps == 0 and is_rank0():
                print(f"[Step {global_step}] [Epoch {epoch}] "
                      f"loss={result['loss']:.4f} "
                      f"mean_reward={result['mean_reward']:.4f} "
                      f"reward_std={result['reward_std']:.4f}")

            if global_step % save_steps == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}", "model.pt")
                save_fsdp_checkpoint(model, ckpt_path)

    # Final save
    final_path = os.path.join(output_dir, "final", "model.pt")
    save_fsdp_checkpoint(model, final_path)
    if is_rank0():
        print("Training complete.")


if __name__ == "__main__":
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    args_list = sys.argv[1:]
    yaml_path = args_list[0] if args_list and args_list[0].endswith(".yaml") else "configs/grpo_config.yaml"
    if args_list and args_list[0].endswith(".yaml"):
        args_list = args_list[1:]

    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
