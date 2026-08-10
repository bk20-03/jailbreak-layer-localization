"""CLI runner for hidden-state extraction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from harmprobe.extraction.config_loader import load_extraction_config
from harmprobe.extraction.hidden_states import extract_hidden_states, load_prompts_from_csv
from harmprobe.runners.status_tracker import mark_completed, mark_failed, mark_running, write_status


def _validate_before_run(cfg: dict) -> None:
    if cfg.get("checkpoint_type") != "base" and not cfg.get("model_path"):
        raise ValueError(
            "Fine-tuned extraction requires model_path in the YAML or "
            "HARMPROBE_FT_MODEL in the environment."
        )

    source = Path(cfg["source_csv"])
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source}")

    output = Path(cfg["output_h5"])
    if output.exists() and not cfg.get("overwrite"):
        raise FileExistsError(
            f"Output HDF5 already exists: {output}. Set overwrite=true to replace."
        )

    load_prompts_from_csv(cfg)  # validate prompts load


def _write_extraction_manifest(cfg: dict, output_h5: Path) -> Path:
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "job_type": "extraction",
        "config_path": cfg["config_path"],
        "output_h5": str(output_h5),
        "run_dir": str(run_dir),
        "canonical_class": cfg["canonical_class"],
        "checkpoint_type": cfg["checkpoint_type"],
        "source_csv": cfg["source_csv"],
        "layers": cfg["layers"],
        "steps": cfg["steps"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
    }
    path = run_dir / "extraction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def run_extraction(config_path: Path) -> Path:
    cfg = load_extraction_config(config_path)
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    _validate_before_run(cfg)
    write_status(
        run_dir,
        {
            "experiment_id": cfg["experiment_id"],
            "job_type": "extraction",
            "status": "running",
            "config_path": cfg["config_path"],
            "output_dir": str(run_dir),
            "output_h5": cfg["output_h5"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "launcher": "local",
            "error": None,
        },
    )

    try:
        print(f"[HarmProbe extraction] experiment_id={cfg['experiment_id']}")
        print(f"[HarmProbe extraction] source_csv={cfg['source_csv']}")
        print(f"[HarmProbe extraction] output_h5={cfg['output_h5']}")

        output_h5 = extract_hidden_states(cfg)
        manifest_path = _write_extraction_manifest(cfg, output_h5)
        mark_completed(run_dir)
        print(f"Extraction complete. HDF5: {output_h5}")
        print(f"Manifest: {manifest_path}")
        return Path(output_h5)
    except Exception as exc:
        mark_failed(run_dir, str(exc))
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract hidden states from CSV prompts")
    parser.add_argument("--config", required=True, help="Extraction YAML config path")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    run_extraction(config_path)


if __name__ == "__main__":
    main()
