"""Shared text generation helpers (used by dataset pipeline and extraction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harmprobe.utils.chat_templates import apply_llama3_user_template, get_llama3_eos_ids, setup_tokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 300
    max_input_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 8
    seed: int = 42


def load_model_tokenizer(
    model_path: str,
    *,
    dtype: str = "float16",
    local_files_only: bool = False,
    device: str | None = None,
    dry_run: bool = False,
):
    """Load a causal LM + tokenizer.

    Default behavior is unchanged (``device_map="auto"``, float16) so existing
    GPU callers (extraction) are unaffected. Pass ``device="cpu"`` for a CPU-safe
    load (float32, no ``device_map``, explicit ``.to("cpu")``) — float16 generation
    is not supported on CPU.
    """
    if dry_run:
        raise RuntimeError("load_model_tokenizer must not be called in dry-run mode")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("torch and transformers required for generation") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    tokenizer = setup_tokenizer(tokenizer)
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )
        model = model.to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map.get(dtype, torch.float16),
            device_map="auto",
            local_files_only=local_files_only,
        )
    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    gen_cfg: GenerationConfig,
) -> list[str]:
    """Generate one response per prompt (sequential; batch wrapper below)."""
    import torch

    eos_ids = get_llama3_eos_ids(tokenizer)
    device = next(model.parameters()).device
    outputs: list[str] = []
    for raw in prompts:
        chat = apply_llama3_user_template(tokenizer, raw)
        inputs = tokenizer(
            chat,
            return_tensors="pt",
            truncation=True,
            max_length=gen_cfg.max_input_tokens,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=True,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )
        gen_ids = out[0, inputs.input_ids.shape[1] :]
        outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return outputs


def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    gen_cfg: GenerationConfig,
) -> list[str]:
    """Generate responses in batches of gen_cfg.batch_size."""
    results: list[str] = []
    bs = max(1, gen_cfg.batch_size)
    for i in range(0, len(prompts), bs):
        chunk = prompts[i : i + bs]
        results.extend(generate_responses(model, tokenizer, chunk, gen_cfg))
    return results


def generation_params_dict(gen_cfg: GenerationConfig) -> dict[str, Any]:
    return {
        "max_new_tokens": gen_cfg.max_new_tokens,
        "max_input_tokens": gen_cfg.max_input_tokens,
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
        "batch_size": gen_cfg.batch_size,
        "seed": gen_cfg.seed,
    }
