"""Chat template helpers for instruct-model generation."""

from __future__ import annotations


def apply_user_template(tokenizer, text: str) -> str:
    """Apply the tokenizer chat template for a single user message."""
    messages = [{"role": "user", "content": text.strip()}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback for templates without chat_template
    return text.strip()


# Back-compat alias used by existing callers
apply_llama3_user_template = apply_user_template


def get_eos_ids(tokenizer) -> list[int]:
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    for tok in ("<|eot_id|>", "<|im_end|>", "</s>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
            # skip UNK-like ids
            if tid == getattr(tokenizer, "unk_token_id", None):
                continue
            eos_ids.append(tid)
    return eos_ids


get_llama3_eos_ids = get_eos_ids


def setup_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer
