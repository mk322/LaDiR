"""Tokenizer setup shared by every stage.

The four special tokens are appended in this exact order, giving ids 151669..151672 on Qwen3 and a
vocabulary of 151673. The released checkpoints depend on this order.
"""

from transformers import AutoTokenizer

SPECIAL_TOKENS = ["<bot>", "<eot>", "<latent>", "<timestep>"]


def setup_tokenizer(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
    tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
    tokenizer.latent_token_id = tokenizer.convert_tokens_to_ids("<latent>")
    tokenizer.time_token_id = tokenizer.convert_tokens_to_ids("<timestep>")
    return tokenizer


def load_tokenizer(name_or_path: str = "Qwen/Qwen3-8B-Base"):
    return setup_tokenizer(AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True))
