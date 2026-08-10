"""CLI runner for the dataset generation pipeline.

Supports a tiny CPU-only smoke (`--execute --device cpu`, Mission 8B) and the
original dry-run planner. This runner NEVER submits SLURM jobs; it only runs the
pipeline in-process. SLURM compatibility is preserved by keeping the runner a
plain CLI that a future sbatch wrapper can call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harmprobe.datasets.config_loader import load_dataset_pipeline_config
from harmprobe.datasets.pipeline import print_pipeline_plan, run_dataset_pipeline
from harmprobe.runners.config_loader import find_framework_root

LEGACY_FOLDERS = ("jailbreak_llama3", "jailbreaks", "Qwen", "gemma2-7b", "jailbreak_mistral")
MAX_SMOKE_SAMPLES = 50
MAX_SMOKE_NEW_TOKENS = 128


def validate_cpu_smoke(cfg: dict[str, Any], device: str) -> list[str]:
    issues: list[str] = []

    exec_device = cfg["execution"]["device"]
    if device != "cpu":
        issues.append(f"--device must be cpu for the CPU smoke (got '{device}').")
    if exec_device != "cpu":
        issues.append(f"execution.device must be cpu in the config (got '{exec_device}').")

    if cfg["execution"]["use_slurm"]:
        issues.append("execution.use_slurm must be false (this runner never submits SLURM).")

    if cfg["overwrite"]:
        issues.append("overwrite must be false to avoid clobbering existing CSVs.")

    for key in ("max_harmful_samples", "max_benign_samples"):
        val = int(cfg["limits"][key])
        if val > MAX_SMOKE_SAMPLES:
            issues.append(f"limits.{key}={val} too large for a CPU smoke (expected <= {MAX_SMOKE_SAMPLES}).")
    if int(cfg["limits"]["max_new_tokens"]) > MAX_SMOKE_NEW_TOKENS:
        issues.append(
            f"limits.max_new_tokens={cfg['limits']['max_new_tokens']} too large for a CPU smoke "
            f"(expected <= {MAX_SMOKE_NEW_TOKENS})."
        )

    for key, path in cfg["outputs"].items():
        norm = str(path).replace("\\", "/")
        if "/data/generated/" not in norm:
            issues.append(f"output '{key}' is not under data/generated/: {path}")
        for legacy in LEGACY_FOLDERS:
            if f"/{legacy}/" in norm:
                issues.append(f"output '{key}' points to legacy folder '{legacy}': {path}")

    return issues


def validate_gpu_execute(cfg: dict[str, Any], device: str) -> list[str]:
    """Pre-flight checks for GPU execution (local or via SLURM sbatch wrapper)."""
    issues: list[str] = []

    if device not in {"cuda", "auto"}:
        issues.append(f"--device must be cuda for GPU execution (got '{device}').")

    if cfg["execution"]["use_slurm"]:
        issues.append(
            "execution.use_slurm must be false in the config — the sbatch wrapper submits; "
            "the runner only executes on the compute node."
        )

    for key, path in cfg["outputs"].items():
        norm = str(path).replace("\\", "/")
        if "/data/generated/" not in norm:
            issues.append(f"output '{key}' is not under data/generated/: {path}")
        for legacy in LEGACY_FOLDERS:
            if f"/{legacy}/" in norm:
                issues.append(f"output '{key}' points to legacy folder '{legacy}': {path}")

    if not cfg["overwrite"]:
        from harmprobe.datasets.pipeline import validate_output_paths

        warnings = validate_output_paths(cfg)
        if warnings:
            for w in warnings:
                issues.append(w)

    return issues


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Llama-3 dataset pipeline")
    parser.add_argument("--config", required=True, help="Dataset pipeline YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Plan only — no model load, WildGuard, or CSV writes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the pipeline in-process (CPU smoke or GPU). Never submits SLURM.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Execution device (e.g. 'cpu'). Defaults to execution.device in the config.",
    )
    parser.add_argument("--json", action="store_true", help="Print plan as JSON")
    args = parser.parse_args(argv)

    framework_root = find_framework_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (framework_root / config_path).resolve()

    cfg = load_dataset_pipeline_config(config_path)

    dry_run = (not args.execute) or args.dry_run

    if not dry_run:
        device = args.device or cfg["execution"]["device"]

        # Hard safety: this runner never submits SLURM jobs.
        if cfg["execution"]["use_slurm"]:
            raise SystemExit(
                "ERROR: execution.use_slurm=true. This runner only executes locally and "
                "never submits SLURM jobs. Set use_slurm: false or wrap this CLI in sbatch."
            )

        if device == "cpu":
            print("CPU smoke mode enabled. No SLURM job will be submitted.")
            issues = validate_cpu_smoke(cfg, device)
        elif device in ("cuda", "auto"):
            print(f"GPU execution mode (device={device}). No SLURM job will be submitted by this runner.")
            issues = validate_gpu_execute(cfg, device if device != "auto" else "cuda")
        else:
            issues = [f"Unsupported device '{device}' — use cpu or cuda."]
        if issues:
            print("Pre-flight validation FAILED — aborting (no models loaded, no CSV writes):")
            for issue in issues:
                print(f"  - {issue}")
            raise SystemExit(1)

        cfg["execution"]["device"] = device
        print(f"Pre-flight validation passed. Executing pipeline locally on device={device}.")
        print(f"Stages: {cfg['stages']}")

    result = run_dataset_pipeline(cfg, dry_run=dry_run)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_pipeline_plan(result)


if __name__ == "__main__":
    main()
