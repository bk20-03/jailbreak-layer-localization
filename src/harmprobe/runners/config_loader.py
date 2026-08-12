"""Config loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FINETUNED_CHECKPOINT_ALIASES = frozenset({"finetuned", "fine_tuned"})

def is_finetuned_checkpoint(checkpoint: str) -> bool:
    return checkpoint in FINETUNED_CHECKPOINT_ALIASES

def task_h5_paths_key(checkpoint: str, task_h5_paths: dict[str, Any]) -> str:
    """Resolve experiment checkpoint name to task YAML ``h5_paths`` key."""
    if is_finetuned_checkpoint(checkpoint):
        for candidate in ("fine_tuned", "finetuned"):
            if candidate in task_h5_paths:
                return candidate
        raise KeyError(
            "Task config has no fine-tuned HDF5 block (expected 'fine_tuned' or 'finetuned')"
        )
    if checkpoint not in task_h5_paths:
        raise KeyError(f"Checkpoint '{checkpoint}' not found in task h5_paths")
    return checkpoint

def find_finetuned_checkpoint(checkpoints: list[str]) -> str | None:
    for checkpoint in checkpoints:
        if is_finetuned_checkpoint(checkpoint):
            return checkpoint
    return None

def find_framework_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__).resolve()).parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate jailbreak-layer-localization root")

def find_workspace_root(start: Path | None = None) -> Path:
    """Locate workspace root for data path resolution.

    Prefer this package root when it looks like the standalone Task B repo
    (``pyproject.toml`` + ``data/``). Otherwise fall back to a parent that
    contains ``jailbreak_llama3/`` (legacy co-located layout).
    """
    cur = (start or Path.cwd()).resolve()
    try:
        fw = find_framework_root(cur)
        if (fw / "pyproject.toml").is_file() and (fw / "data").is_dir():
            return fw
    except FileNotFoundError:
        pass
    for candidate in [cur, *cur.parents]:
        if (candidate / "jailbreak_llama3").is_dir():
            return candidate
    return find_framework_root(cur)

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_layers(exp: dict[str, Any], n_layers: int) -> list[int]:
    if "layers" not in exp:
        return list(range(n_layers))
    layers_cfg = exp["layers"]
    if isinstance(layers_cfg, list):
        return [int(layer) for layer in layers_cfg]
    if isinstance(layers_cfg, dict):
        mode = layers_cfg.get("mode", "all")
        if mode == "all":
            return list(range(n_layers))
        if mode == "range":
            start = int(layers_cfg["start"])
            end = int(layers_cfg["end"])
            return list(range(start, end + 1))
    raise ValueError(f"Invalid layers config: {layers_cfg!r}")

def _resolve_steps(exp: dict[str, Any], task: dict[str, Any]) -> list[int]:
    if "steps" in exp:
        steps_cfg = exp["steps"]
        if isinstance(steps_cfg, list):
            return [int(step) for step in steps_cfg]
        if isinstance(steps_cfg, dict):
            mode = steps_cfg.get("mode", "default")
            if mode == "default":
                return list(range(int(task["n_steps"])))
            if mode == "range":
                start = int(steps_cfg["start"])
                end = int(steps_cfg["end"])
                return list(range(start, end + 1))
    steps = list(range(int(task["n_steps"])))
    if task.get("steps") is not None:
        steps = [int(step) for step in task["steps"]]
    return steps

def resolve_path(path_str: str, *, base: Path, workspace_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if path_str.startswith("jailbreak_llama3/"):
        return workspace_root / path_str
    return (base / path_str).resolve()

def load_experiment_config(config_path: Path) -> dict[str, Any]:
    framework_root = find_framework_root(config_path.parent)
    workspace_root = find_workspace_root(framework_root)

    exp = load_yaml(config_path)
    model = load_yaml(resolve_path(exp["model"], base=framework_root, workspace_root=workspace_root))
    task = load_yaml(resolve_path(exp["task"], base=framework_root, workspace_root=workspace_root))

    probe_label_map = {int(k): int(v) for k, v in task["probe_label_map"].items()}
    canonical_classes = [int(c) for c in task["canonical_classes"]]
    load_order = task.get("load_order", canonical_classes)
    load_order = [int(c) for c in load_order]

    steps = _resolve_steps(exp, task)

    c_values = [float(c) for c in task["c_values"]]

    class_weight = task.get("class_weight", "balanced")
    if class_weight in ("null", "none", None):
        class_weight = None

    # h5_paths may be defined at the experiment level (model-specific override)
    # or fall back to the task config. Experiment-level blocks are keyed by the
    # canonical checkpoint name ("base" / "fine_tuned").
    exp_h5_paths = exp.get("h5_paths")

    def _checkpoint_h5_block(checkpoint: str) -> dict[str, str]:
        if exp_h5_paths is not None:
            if checkpoint in exp_h5_paths:
                return exp_h5_paths[checkpoint]
            if is_finetuned_checkpoint(checkpoint):
                for candidate in ("fine_tuned", "finetuned"):
                    if candidate in exp_h5_paths:
                        return exp_h5_paths[candidate]
            raise KeyError(
                f"Experiment h5_paths has no block for checkpoint '{checkpoint}'"
            )
        task_ck = task_h5_paths_key(checkpoint, task["h5_paths"])
        return task["h5_paths"][task_ck]

    h5_paths: dict[str, dict[int, str]] = {}
    for checkpoint in exp.get("checkpoints", ["base"]):
        ck_cfg = _checkpoint_h5_block(checkpoint)
        h5_paths[checkpoint] = {}
        for key, rel in ck_cfg.items():
            # key like class0_base -> extract canonical class number
            if key.startswith("class"):
                canonical = int(key.replace("class", "").split("_")[0])
            else:
                raise ValueError(f"Unexpected h5 path key: {key}")
            resolved = resolve_path(
                rel, base=framework_root, workspace_root=workspace_root
            )
            h5_paths[checkpoint][canonical] = str(resolved)

    output_dir = resolve_path(
        exp.get("output_dir", f"runs/{exp['experiment_id']}"),
        base=framework_root,
        workspace_root=workspace_root,
    )

    reference_outputs = None
    if "reference_outputs" in exp:
        reference_outputs = resolve_path(
            exp["reference_outputs"], base=workspace_root, workspace_root=workspace_root
        )

    oc_raw = task.get("overfitting_correction") or {}
    if isinstance(oc_raw, bool):
        oc_raw = {"enabled": oc_raw}
    # Experiment-level override wins when present.
    exp_oc = exp.get("overfitting_correction")
    if isinstance(exp_oc, dict):
        oc_raw = {**oc_raw, **exp_oc}
    elif isinstance(exp_oc, bool):
        oc_raw = {**oc_raw, "enabled": exp_oc}

    overfitting_correction = {
        "enabled": bool(oc_raw.get("enabled", False)),
        "pca_components": int(oc_raw.get("pca_components", 30)),
        "n_train_steps": int(oc_raw.get("n_train_steps", 12)),
        "cv_folds": int(oc_raw.get("cv_folds", task.get("cv_folds", 5))),
        "cv_repeats": int(oc_raw.get("cv_repeats", 3)),
        "c_selection": str(oc_raw.get("c_selection", "one_se")),
    }

    return {
        "framework_root": str(framework_root),
        "workspace_root": str(workspace_root),
        "experiment_id": exp["experiment_id"],
        "task_id": task["task_id"],
        "model_id": model["model_id"],
        "model_family": model.get("model_family"),
        "n_layers": int(model["n_layers"]),
        "hidden_dim": int(model["hidden_dim"]),
        "canonical_classes": canonical_classes,
        "load_order": load_order,
        "probe_label_map": probe_label_map,
        "steps": steps,
        "n_steps": len(steps),
        "c_values": c_values,
        "cv_mode": task["cv_mode"],
        "cv_folds": int(task.get("cv_folds", 5)),
        "class_weight": class_weight,
        "prompt_weighting": bool(task.get("prompt_weighting", False)),
        "split_mode": task.get("split_mode", "nested_15_15"),
        "test_size": float(task.get("test_size", 0.15)),
        "val_size": float(task.get("val_size", 0.15)),
        "seed": int(task.get("seed", exp.get("seed", 42))),
        "max_iter": int(task.get("max_iter", 3000)),
        "checkpoints": exp.get("checkpoints", ["base", "fine_tuned"]),
        "layers": _resolve_layers(exp, int(model["n_layers"])),
        "h5_paths": h5_paths,
        "output_dir": str(output_dir),
        "reference_outputs": str(reference_outputs) if reference_outputs else None,
        "task_type": task.get("task_type", "compliance_vs_refusal"),
        "overfitting_correction": overfitting_correction,
    }
