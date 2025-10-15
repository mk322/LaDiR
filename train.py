import torch
from transformers.training_args import TrainingArguments
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import Trainer
from config import LMFusionConfig
from dataset import ThoughtDataset, ThoughtDataCollator
from model import LMFusionModel
import os
from safetensors.torch import load_file
import pathlib
from omegaconf import OmegaConf as om
from vae.model_vae import VAE
from vae.vae_args import parse_args
from peft import LoraConfig


def main(cfg):

    print(f'run name: {cfg.run_name}')
    local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = torch.device(local_rank)

    # AE
    ae_lora_config = LoraConfig(
        r=cfg.ae.lora_r,
        lora_alpha=cfg.ae.lora_alpha,
        lora_dropout=cfg.ae.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    ae_model_args, ae_training_args, ae_args = parse_args()
    ae = VAE(ae_model_args, ae_training_args, ae_lora_config)
    print(f"Loading trained checkpoint from {cfg.ae.icae_ckpt}")

    state_dict = load_file(cfg.ae.icae_ckpt)

    if "state_dict" in state_dict:
        missing_keys, unexpected_keys = ae.load_state_dict(state_dict["state_dict"], strict=False)  # Allow missing keys if needed
    else:
        missing_keys, unexpected_keys = ae.load_state_dict(state_dict, strict=False)

    for p in ae.parameters():
        p.requires_grad = False
        
    ae = ae.to(local_rank, dtype=torch.bfloat16)

    TEXT_LLAMA_PATH = cfg.model.llm_model_name_or_path

    # Load text LLaMA
    text_llama_config = AutoConfig.from_pretrained(TEXT_LLAMA_PATH, use_flash_attention=False, _flash_attn_2_enabled=False)


    text_llama = AutoModelForCausalLM.from_pretrained(TEXT_LLAMA_PATH, 
                                                  config=text_llama_config,
                                                    torch_dtype=torch.bfloat16, # or torch.bfloat16
                                                  )

    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_LLAMA_PATH)

    text_tokenizer.pad_token_id = text_tokenizer.eos_token_id

    BOT_TOKEN = "<tht_s>"
    THT_TOKEN = "<tht>"
    EOT_TOKEN = "</tht_s>"
    TIME_TOKEN = "<timestep>"
    for special_token in [BOT_TOKEN, THT_TOKEN, EOT_TOKEN, TIME_TOKEN]:
        special_tokens_dict = {"additional_special_tokens": [special_token]}
        text_tokenizer.add_special_tokens(special_tokens_dict)
    text_tokenizer.bot_token_id = text_tokenizer.convert_tokens_to_ids(BOT_TOKEN)
    text_tokenizer.tht_token_id = text_tokenizer.convert_tokens_to_ids(THT_TOKEN)
    text_tokenizer.eot_token_id = text_tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    
    text_tokenizer.time_token_id = text_tokenizer.convert_tokens_to_ids(TIME_TOKEN)

    text_tokenizer.pad_token_id = text_tokenizer.eos_token_id

    # Freeze all autoencoder parameters
    for param in ae.parameters():
        param.requires_grad = False

    # Initialize our LMFusion model
    model = LMFusionModel(
        text_llama=text_llama,
        thought_llama=None,
        autoencoder=ae,
        model_config=cfg,
        tokenizer=text_tokenizer,
        hidden_dim=text_llama_config.hidden_size,
        freeze_text=False
    ).to(dtype=torch.bfloat16)

    cfg.n_params = sum(p.numel() for p in model.parameters())
    cfg.n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{cfg.n_params=:.6e}, {cfg.n_trainable_params=:.6e}')

    # c) Create a toy dataset
    train_dataset = ThoughtDataset(
        text_tokenizer,
        cfg.dataset.train_file,
    )

    data_collator = ThoughtDataCollator(pad_token_id=text_tokenizer.pad_token_id)

    print(len(train_dataset))

    training_args = TrainingArguments(**om.to_container(cfg.trainer, resolve=True))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainable_params = [n for n, p in trainer.model.autoencoder.named_parameters() if p.requires_grad]
    print(f"Before training Trainable AE parameters:\n{trainable_params}")

    if cfg.allow_resume and list(pathlib.Path(cfg.trainer.output_dir).glob('checkpoint*')):
        print('Resume from last checkpoint!!!!')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    print('Done.')

if __name__ == '__main__':
    import sys
    import os

    if 'RANK' in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')

    #yaml_path, args_list = sys.argv[1], sys.argv[2:]
    args_list = sys.argv[1:]
    yaml_path = "configs/cd_formal_8B_VAE_conn.yaml"
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
