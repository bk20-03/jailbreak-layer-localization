"""Chat template helpers for Llama-3 generation."""

from __future__ import annotations


def apply_llama3_user_template(tokenizer, text: str) -> str:
    """Apply Llama-3 instruct chat template for a single user message."""
    messages = [{"role": "user", "content": text.strip()}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_llama3_eos_ids(tokenizer) -> list[int]:
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)
    return eos_ids


def setup_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer
