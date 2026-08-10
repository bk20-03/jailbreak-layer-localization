"""Dataset pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harmprobe.datasets.check_compliance import run_fine_tuned_compliance
from harmprobe.datasets.generate_benign_pairs import run_benign_generation
from harmprobe.datasets.generate_harmful_pairs import run_harmful_base_generation
from harmprobe.datasets.loaders import report_row_count

STAGE_RUNNERS = {
    "harmful_base_generation": run_harmful_base_generation,
    "fine_tuned_compliance": run_fine_tuned_compliance,
    "benign_generation": run_benign_generation,
}


def validate_output_paths(cfg: dict[str, Any]) -> list[str]:
    """Return warnings about existing outputs when overwrite=false."""
    warnings: list[str] = []
    if cfg.get("overwrite"):
        return warnings
    for key, path in cfg["outputs"].items():
        p = Path(path)
        if p.is_file():
            rows = report_row_count(p)
            warnings.append(f"{key}: exists ({rows} rows) — will not overwrite without overwrite=true")
    return warnings


def validate_model_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check local model paths exist; HF IDs are reported as-is."""
    info: dict[str, Any] = {}
    for label, path in [
        ("base_model", cfg["model"]["base_model_path"]),
        ("fine_tuned_model", cfg["model"]["fine_tuned_model_path"]),
        ("tokenizer", cfg["model"]["tokenizer_path"]),
    ]:
        if not path:
            info[label] = {"path": "", "type": "unset", "exists": False}
            continue
        p = Path(path)
        if str(path).startswith("meta-llama/") or str(path).startswith("allenai/"):
            info[label] = {"path": path, "type": "hf_id", "exists": None}
        else:
            info[label] = {"path": path, "type": "local", "exists": p.exists()}
    info["wildguard"] = {
        "path": cfg["wildguard"]["model_path"],
        "enabled": cfg["wildguard"]["enabled"],
    }
    return info


def format_stage_plan(stage_result: dict[str, Any]) -> str:
    lines = [
        f"Stage: {stage_result['stage']}",
        f"  Input:  {stage_result.get('input', stage_result.get('models', '—'))}",
        f"  Output: {stage_result.get('output', '—')}",
        f"  Limit:  {stage_result.get('limit', '—')}",
        f"  Status: {stage_result.get('status', 'planned')} — {stage_result.get('message', '')}",
    ]
    if stage_result.get("canonical_class") is not None:
        lines.append(f"  Class:  {stage_result['canonical_class']}")
    if stage_result.get("processed_pairs") is not None:
        lines.append(
            f"  Run:    processed={stage_result.get('processed_pairs')} "
            f"rows_written={stage_result.get('rows_written')}"
        )
    return "\n".join(lines)


def run_dataset_pipeline(cfg: dict[str, Any], *, dry_run: bool = True) -> dict[str, Any]:
    results: dict[str, Any] = {
        "pipeline_id": cfg["pipeline_id"],
        "dry_run": dry_run,
        "overwrite": cfg["overwrite"],
        "output_warnings": validate_output_paths(cfg),
        "model_paths": validate_model_paths(cfg),
        "stages": [],
    }

    for stage_name in cfg["stages"]:
        runner = STAGE_RUNNERS.get(stage_name)
        if runner is None:
            results["stages"].append({"stage": stage_name, "error": "Unknown stage"})
            continue
        stage_result = runner(cfg, dry_run=dry_run)
        results["stages"].append(stage_result)

    return results


def print_pipeline_plan(result: dict[str, Any]) -> None:
    mode = "DRY-RUN" if result["dry_run"] else "EXECUTE"
    print("=" * 72)
    print(f"Dataset pipeline plan ({mode}): {result['pipeline_id']}")
    print("=" * 72)
    print(f"Overwrite: {result['overwrite']}")
    if result["output_warnings"]:
        print("\nOutput path warnings:")
        for w in result["output_warnings"]:
            print(f"  - {w}")
    print("\nModel paths:")
    for name, info in result["model_paths"].items():
        if isinstance(info, dict) and "path" in info:
            ex = info.get("exists")
            suffix = "" if ex is None else (" OK" if ex else " MISSING")
            print(f"  {name}: {info['path']}{suffix}")
    print("\nStages:")
    for stage in result["stages"]:
        print(format_stage_plan(stage))
        print()
    print("=" * 72)
    if result["dry_run"]:
        print("Dry-run complete — no models loaded, no WildGuard, no CSV writes.")
    print("=" * 72)
