"""Extract generation-step hidden states from CSV prompts into HDF5."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError("h5py is required for extraction") from exc


def _layer_step_key(layer: int, step: int) -> str:
    return f"layer{layer:02d}_step{step:02d}"


def load_prompts_from_csv(cfg: dict[str, Any]) -> tuple[list[str], list[int]]:
    """Load and filter prompts from source CSV."""
    df = pd.read_csv(cfg["source_csv"])
    filt = cfg.get("filter_condition")
    if filt:
        col = filt["column"]
        val = filt["value"]
        df = df[df[col] == val].copy()

    prompt_col = cfg["prompt_column"]
    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found in {cfg['source_csv']}")

    prompts = df[prompt_col].dropna().astype(str).str.strip().tolist()

    id_col = cfg.get("prompt_id_column")
    if id_col and id_col in df.columns:
        prompt_ids = df[id_col].astype(int).tolist()[: len(prompts)]
    else:
        prompt_ids = list(range(len(prompts)))

    limit = cfg.get("max_samples") or cfg.get("n_samples")
    if limit is not None:
        limit = int(limit)
        prompts = prompts[:limit]
        prompt_ids = prompt_ids[:limit]

    if not prompts:
        raise ValueError("No prompts loaded after filtering/limiting")

    return prompts, prompt_ids


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _setup_hf_env(cfg: dict[str, Any]) -> None:
    hf_home = cfg.get("hf_home")
    if hf_home:
        os.environ.setdefault("HF_HOME", str(hf_home))
    if cfg.get("local_files_only", True):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _load_model_and_tokenizer(cfg: dict[str, Any]):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Extraction requires torch and transformers. "
            "Install with: pip install torch transformers tqdm"
        ) from exc

    _setup_hf_env(cfg)
    local_only = cfg.get("local_files_only", True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_path"],
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if tokenizer.chat_template is None and cfg["checkpoint_type"] != "base":
        from harmprobe.extraction.config_loader import DEFAULT_BASE_MODEL

        base_tok = AutoTokenizer.from_pretrained(
            DEFAULT_BASE_MODEL,
            local_files_only=local_only,
        )
        if base_tok.chat_template:
            tokenizer.chat_template = base_tok.chat_template
            print("Chat template copied from base tokenizer.")

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(cfg.get("dtype", "float16"), torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch_dtype,
        device_map="auto",
        local_files_only=local_only,
    )
    model.eval()

    if tokenizer.chat_template is None:
        raise ValueError(
            f"No chat_template on tokenizer for {cfg['model_path']}. "
            "Set tokenizer_path to the base model."
        )

    return model, tokenizer


def _get_eos_ids(tokenizer) -> list[int]:
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)
    return eos_ids


def _apply_chat_template(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text.strip()}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_hidden_states(cfg: dict[str, Any]) -> Path:
    """
    Run hidden-state extraction and write HDF5.

    Returns path to output HDF5 file.
    """
    import torch
    from tqdm import tqdm

    output_path = Path(cfg["output_h5"])
    if output_path.exists() and not cfg.get("overwrite"):
        raise FileExistsError(
            f"Output HDF5 already exists: {output_path}. Set overwrite=true to replace."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(int(cfg["seed"]))
    prompts, prompt_ids = load_prompts_from_csv(cfg)
    total = len(prompts)
    layers = [int(l) for l in cfg["layers"]]
    steps = [int(s) for s in cfg["steps"]]
    hidden_dim = int(cfg["hidden_dim"])
    canonical_class = int(cfg["canonical_class"])
    max_new_tokens = max(steps) + 1 if steps else int(cfg.get("max_new_tokens", 100))

    print(f"Loading model from {cfg['model_path']}...")
    model, tokenizer = _load_model_and_tokenizer(cfg)
    model_layers = model.config.num_hidden_layers
    model_hidden = model.config.hidden_size

    if model_hidden != hidden_dim:
        print(f"Warning: config hidden_dim={hidden_dim} but model reports {model_hidden}")
        hidden_dim = model_hidden

    device = next(model.parameters()).device
    eos_ids = _get_eos_ids(tokenizer)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    metadata = {
        "model_id": cfg["model_id"],
        "checkpoint_type": cfg["checkpoint_type"],
        "canonical_class": canonical_class,
        "source_csv": cfg["source_csv"],
        "extraction_config": {
            "experiment_id": cfg["experiment_id"],
            "layers": layers,
            "steps": steps,
            "max_samples": cfg.get("max_samples"),
            "batch_size": cfg["batch_size"],
            "seed": cfg["seed"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": total,
        "hidden_dim": hidden_dim,
        "n_layers_model": model_layers,
    }

    print(f"Extracting {total} prompts, layers={layers}, steps={steps}")
    print(f"Writing to {output_path}")

    with h5py.File(output_path, "w") as f:
        f.attrs["metadata"] = json.dumps(metadata)
        f.attrs["canonical_class"] = canonical_class
        f.attrs["model_path"] = cfg["model_path"]
        f.attrs["source_csv"] = cfg["source_csv"]

        datasets: dict[tuple[int, int], h5py.Dataset] = {}
        for layer in layers:
            for step in steps:
                datasets[(layer, step)] = f.create_dataset(
                    _layer_step_key(layer, step),
                    shape=(total, hidden_dim),
                    dtype="float32",
                    chunks=True,
                )

        labels_ds = f.create_dataset("labels", shape=(total,), dtype="int8")
        f.create_dataset("prompt_ids", data=np.array(prompt_ids, dtype=np.int64))
        prompts_ds = f.create_dataset("prompts", shape=(total,), dtype=string_dtype)

        batch_size = max(1, int(cfg["batch_size"]))
        indices = list(range(total))

        for batch_start in tqdm(range(0, total, batch_size), desc="Batches"):
            batch_idx = indices[batch_start : batch_start + batch_size]
            for idx in batch_idx:
                raw_text = prompts[idx]
                prompt = _apply_chat_template(tokenizer, raw_text)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg["max_input_len"]),
                ).to(device)

                with torch.no_grad():
                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=float(cfg["temperature"]),
                        top_p=float(cfg["top_p"]),
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=eos_ids,
                    )

                n_gen_steps = min(len(gen_out.hidden_states), max_new_tokens)
                for step_idx in range(n_gen_steps):
                    if step_idx not in steps:
                        continue
                    layer_states_tuple = gen_out.hidden_states[step_idx]
                    for layer in layers:
                        layer_idx = layer + 1  # skip embedding at index 0
                        if layer_idx >= len(layer_states_tuple):
                            continue
                        vec = (
                            layer_states_tuple[layer_idx][0, -1, :]
                            .float()
                            .cpu()
                            .numpy()
                        )
                        datasets[(layer, step_idx)][idx, :] = vec

                labels_ds[idx] = canonical_class
                prompts_ds[idx] = raw_text

                del gen_out
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Saved HDF5: {output_path}")
    return output_path
