"""Read/write run status.json for SLURM and local probing jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTIVE_STATUSES = frozenset({"submitted", "pending", "running"})


def status_path(output_dir: Path) -> Path:
    return Path(output_dir) / "status.json"


def read_status(output_dir: Path) -> dict[str, Any] | None:
    path = status_path(output_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_status(output_dir: Path, data: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = status_path(output_dir)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_submitted_status(
    *,
    experiment_id: str,
    config_path: Path,
    output_dir: Path,
    log_dir: Path,
    job_id: str | None = None,
    stdout_log: str | None = None,
    stderr_log: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": "submitted" if job_id else "pending",
        "job_id": job_id,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "submitted_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "log_dir": str(log_dir),
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "launcher": "slurm",
        "error": None,
    }


def mark_running(output_dir: Path) -> None:
    data = read_status(output_dir)
    if data is None:
        return
    data["status"] = "running"
    if not data.get("started_at"):
        data["started_at"] = _now_iso()
    write_status(output_dir, data)


def mark_completed(output_dir: Path) -> None:
    data = read_status(output_dir)
    if data is None:
        return
    data["status"] = "completed"
    data["finished_at"] = _now_iso()
    data["error"] = None
    write_status(output_dir, data)


def mark_failed(output_dir: Path, error: str) -> None:
    data = read_status(output_dir) or {
        "experiment_id": Path(output_dir).name,
        "output_dir": str(output_dir),
        "launcher": "slurm",
    }
    data["status"] = "failed"
    data["finished_at"] = _now_iso()
    data["error"] = error
    write_status(output_dir, data)


def is_active_submission(output_dir: Path) -> bool:
    data = read_status(output_dir)
    if not data:
        return False
    return str(data.get("status", "")).lower() in ACTIVE_STATUSES
