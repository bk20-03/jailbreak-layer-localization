"""Load dataset pipeline YAML configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from harmprobe.extraction.config_loader import (
    DEFAULT_BASE_MODEL,
    DEFAULT_FT_MODEL,
    DEFAULT_HF_HOME,
)
from harmprobe.runners.config_loader import find_framework_root, resolve_path

DEFAULT_OUTPUTS = {
    "harmful_base_csv": "data/generated/llama3/paired_dataset1.csv",
    "harmful_ft_csv": "data/generated/llama3/paired_dataset_finetuned.csv",
    "benign_csv": "data/generated/llama3/paired_dataset_benign_llama3.csv",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_hf_id(path: str) -> bool:
    return bool(path) and (
        path.startswith("meta-llama/")
        or path.startswith("allenai/")
        or ("/" in path and not path.startswith("/") and not Path(path).exists())
    )


def _resolve_model(path: str, *, framework_root: Path) -> str:
    if not path:
        return path
    if _is_hf_id(path):
        return path
    return str(resolve_path(path, base=framework_root, workspace_root=framework_root))


def load_dataset_pipeline_config(config_path: Path) -> dict[str, Any]:
    framework_root = find_framework_root(config_path.parent)
    raw = load_yaml(config_path)

    model = raw.get("model", {})
    wildguard = raw.get("wildguard", {})
    dataset = raw.get("dataset", {})
    limits = raw.get("limits", {})
    outputs = raw.get("outputs", {})
    execution = raw.get("execution", {})

    base_model = model.get("base_model_path") or os.environ.get(
        "HARMPROBE_BASE_MODEL", DEFAULT_BASE_MODEL
    )
    ft_model = model.get("fine_tuned_model_path") or os.environ.get(
        "HARMPROBE_FT_MODEL", DEFAULT_FT_MODEL
    )
    tokenizer = model.get("tokenizer_path", base_model)

    if not ft_model:
        # Allow dry-run / benign-only; FT stage will fail clearly if executed without path.
        ft_model = ""

    resolved_outputs = {}
    for key, default in DEFAULT_OUTPUTS.items():
        rel = outputs.get(key, default)
        resolved_outputs[key] = str(
            resolve_path(rel, base=framework_root, workspace_root=framework_root)
        )

    run_dir = raw.get("run_dir", "runs/datasets/" + raw.get("pipeline_id", "llama3_dataset"))
    run_dir = str(resolve_path(run_dir, base=framework_root, workspace_root=framework_root))

    return {
        "framework_root": str(framework_root),
        "config_path": str(config_path.resolve()),
        "pipeline_id": raw.get("pipeline_id", raw.get("experiment_id", "llama3_dataset")),
        "model": {
            "name": model.get("name", "llama3_3b"),
            "base_model_path": _resolve_model(base_model, framework_root=framework_root),
            "fine_tuned_model_path": _resolve_model(ft_model, framework_root=framework_root)
            if ft_model
            else "",
            "tokenizer_path": _resolve_model(tokenizer, framework_root=framework_root),
        },
        "wildguard": {
            "model_path": wildguard.get("model_path", "allenai/wildguard"),
            "enabled": bool(wildguard.get("enabled", True)),
        },
        "dataset": {
            "source": dataset.get("source", "allenai/wildjailbreak"),
            "split": dataset.get("split", "train"),
            "harmful_type": dataset.get("harmful_type", "adversarial_harmful"),
            "benign_type": dataset.get("benign_type", "adversarial_benign"),
        },
        "limits": {
            "max_harmful_samples": int(limits.get("max_harmful_samples", 5000)),
            "max_benign_samples": int(limits.get("max_benign_samples", 5000)),
            "max_new_tokens": int(limits.get("max_new_tokens", 300)),
            "max_input_tokens": int(limits.get("max_input_tokens", 512)),
            "batch_size": int(limits.get("batch_size", 8)),
        },
        "outputs": resolved_outputs,
        "run_dir": run_dir,
        "overwrite": bool(raw.get("overwrite", False)),
        "seed": int(raw.get("seed", 42)),
        "hf_home": raw.get("hf_home", DEFAULT_HF_HOME),
        "execution": {
            "mode": str(execution.get("mode", "local")),
            "device": str(execution.get("device", "auto")),
            "use_slurm": bool(execution.get("use_slurm", False)),
            "allow_wildguard_on_cpu": bool(execution.get("allow_wildguard_on_cpu", False)),
        },
        "stages": raw.get(
            "stages",
            [
                "harmful_base_generation",
                "fine_tuned_compliance",
                "benign_generation",
            ],
        ),
    }
