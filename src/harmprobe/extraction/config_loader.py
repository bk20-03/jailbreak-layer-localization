"""Load extraction YAML configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from harmprobe.runners.config_loader import find_framework_root, find_workspace_root, resolve_path

DEFAULT_FT_MODEL = os.environ.get("HARMPROBE_FT_MODEL", "")
DEFAULT_HF_HOME = os.environ.get("HF_HOME") or os.environ.get("HARMPROBE_HF_HOME")

# Portable defaults: optional env overrides; prefer per-YAML model paths.
DEFAULT_BASE_MODEL = os.environ.get(
    "HARMPROBE_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_hf_id(path: str) -> bool:
    known_prefixes = (
        "meta-llama/",
        "allenai/",
        "Qwen/",
        "mistralai/",
        "google/",
    )
    if any(path.startswith(p) for p in known_prefixes):
        return True
    return "/" in path and not path.startswith("/") and not Path(path).exists()


def _resolve_model_path(path: str, *, base: Path, workspace_root: Path) -> str:
    if _is_hf_id(path):
        return path
    return str(resolve_path(path, base=base, workspace_root=workspace_root))


def _resolve_layers(layers_cfg: dict[str, Any] | list[int] | None, n_layers: int) -> list[int]:
    if layers_cfg is None:
        return list(range(n_layers))
    if isinstance(layers_cfg, list):
        return [int(x) for x in layers_cfg]
    mode = layers_cfg.get("mode", "all")
    if mode == "all":
        return list(range(n_layers))
    if mode == "range":
        return list(range(int(layers_cfg["start"]), int(layers_cfg["end"]) + 1))
    raise ValueError(f"Invalid layers config: {layers_cfg!r}")


def _resolve_steps(steps_cfg: dict[str, Any] | list[int] | None, default_n: int = 100) -> list[int]:
    if steps_cfg is None:
        return list(range(default_n))
    if isinstance(steps_cfg, list):
        return [int(x) for x in steps_cfg]
    mode = steps_cfg.get("mode", "default")
    if mode == "default":
        return list(range(default_n))
    if mode == "range":
        return list(range(int(steps_cfg["start"]), int(steps_cfg["end"]) + 1))
    raise ValueError(f"Invalid steps config: {steps_cfg!r}")


def load_extraction_config(config_path: Path) -> dict[str, Any]:
    framework_root = find_framework_root(config_path.parent)
    workspace_root = find_workspace_root(framework_root)
    raw = load_yaml(config_path)

    experiment_id = raw["experiment_id"]
    if "n_layers" not in raw or "hidden_dim" not in raw:
        raise ValueError(
            f"{config_path}: n_layers and hidden_dim are required "
            "(do not rely on Llama-3 defaults)."
        )
    n_layers = int(raw["n_layers"])
    hidden_dim = int(raw["hidden_dim"])
    default_steps = int(raw.get("default_steps", 100))

    model_id = raw.get("model_id")
    if not model_id:
        raise ValueError(f"{config_path}: model_id is required")

    checkpoint_type = raw.get("checkpoint_type", "base")
    model_path = raw.get("model_path")
    if not model_path:
        if checkpoint_type == "base":
            # Prefer HARMPROBE_BASE_MODEL override, else this YAML's model_id.
            model_path = os.environ.get("HARMPROBE_BASE_MODEL") or model_id
        else:
            model_path = DEFAULT_FT_MODEL or ""
    if model_path:
        model_path = _resolve_model_path(
            model_path, base=framework_root, workspace_root=workspace_root
        )

    # Base tokenizer for chat-template fallback on FT checkpoints.
    base_tokenizer_path = raw.get("tokenizer_path") or model_id
    base_tokenizer_path = _resolve_model_path(
        base_tokenizer_path, base=framework_root, workspace_root=workspace_root
    )

    if checkpoint_type == "base":
        tokenizer_path = base_tokenizer_path
    else:
        # Prefer explicit tokenizer_path (usually the base HF id); else FT model_path.
        tokenizer_path = raw.get("tokenizer_path") or model_path or model_id
        tokenizer_path = _resolve_model_path(
            tokenizer_path, base=framework_root, workspace_root=workspace_root
        )

    source_csv = str(
        resolve_path(raw["source_csv"], base=framework_root, workspace_root=workspace_root)
    )
    output_h5 = str(
        resolve_path(raw["output_h5"], base=framework_root, workspace_root=workspace_root)
    )
    run_dir = raw.get("run_dir", f"runs/extractions/{experiment_id}")
    run_dir = str(resolve_path(run_dir, base=framework_root, workspace_root=workspace_root))

    filter_condition = raw.get("filter_condition")
    if filter_condition is not None and not isinstance(filter_condition, dict):
        raise ValueError("filter_condition must be a dict with column and value")

    max_samples = raw.get("max_samples")
    if max_samples is None:
        raise ValueError(
            f"{config_path}: max_samples is required for Task B extraction "
            "(use 102 to match published probe sizes)."
        )

    return {
        "framework_root": str(framework_root),
        "workspace_root": str(workspace_root),
        "config_path": str(config_path.resolve()),
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "base_tokenizer_path": base_tokenizer_path,
        "checkpoint_type": checkpoint_type,
        "canonical_class": int(raw["canonical_class"]),
        "source_csv": source_csv,
        "prompt_column": raw.get("prompt_column", "adversarial_raw"),
        "prompt_id_column": raw.get("prompt_id_column"),
        "filter_condition": filter_condition,
        "output_h5": output_h5,
        "run_dir": run_dir,
        "max_samples": int(max_samples),
        "n_samples": raw.get("n_samples"),
        "layers": _resolve_layers(raw.get("layers"), n_layers),
        "steps": _resolve_steps(raw.get("steps"), default_steps),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "batch_size": int(raw.get("batch_size", 1)),
        "dtype": raw.get("dtype", "float16"),
        "device": raw.get("device", "cuda"),
        "seed": int(raw.get("seed", 42)),
        "overwrite": bool(raw.get("overwrite", False)),
        "max_input_len": int(raw.get("max_input_len", 512)),
        "max_new_tokens": int(raw.get("max_new_tokens", default_steps)),
        "temperature": float(raw.get("temperature", 0.7)),
        "top_p": float(raw.get("top_p", 0.9)),
        "hf_home": raw.get("hf_home", DEFAULT_HF_HOME),
        "local_files_only": bool(raw.get("local_files_only", False)),
    }
