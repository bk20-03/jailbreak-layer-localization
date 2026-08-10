"""Dataset loading, validation, and safe CSV I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from harmprobe.datasets.schemas import CsvSchema


def load_existing_csv(path: Path | str, *, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, nrows=nrows)


def validate_csv_schema(
    path: Path | str,
    schema: CsvSchema,
    *,
    sample_rows: int = 5,
) -> dict[str, Any]:
    """Validate an existing CSV against a schema (read-only)."""
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "schema": schema.name,
        "exists": path.is_file(),
        "valid": False,
        "row_count": None,
        "missing_columns": [],
        "extra_columns": [],
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("File does not exist")
        return result

    try:
        df = pd.read_csv(path, nrows=max(sample_rows, 1))
        result["row_count"] = len(pd.read_csv(path, usecols=[0]))
    except (OSError, pd.errors.EmptyDataError, ValueError) as exc:
        result["errors"].append(f"Could not read CSV: {exc}")
        return result

    cols = set(df.columns)
    required = set(schema.required_columns)
    missing = sorted(required - cols)
    result["missing_columns"] = missing
    result["extra_columns"] = sorted(cols - required - set(schema.optional_columns))
    if missing:
        result["errors"].append(f"Missing columns: {missing}")
    else:
        result["valid"] = True
    return result


def sample_rows(df: pd.DataFrame, n: int, *, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def save_csv_safe(
    df: pd.DataFrame,
    path: Path | str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write CSV only when not dry-run and overwrite rules allow."""
    path = Path(path)
    info: dict[str, Any] = {
        "path": str(path),
        "rows": len(df),
        "written": False,
        "dry_run": dry_run,
    }
    if dry_run:
        info["message"] = "Dry-run: CSV not written"
        return info
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    info["written"] = True
    info["message"] = f"Wrote {len(df)} rows"
    return info


def load_wildjailbreak_split(
    source: str = "allenai/wildjailbreak",
    split: str = "train",
    *,
    dry_run: bool = False,
    data_type: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame | None:
    """
    Load WildJailbreak TSV split.

    In dry-run mode returns None without downloading.
    """
    if dry_run:
        return None
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets package required for WildJailbreak loading") from exc

    ds = load_dataset(source, split=split)
    df = ds.to_pandas()
    if data_type and "data_type" in df.columns:
        df = df[df["data_type"] == data_type].copy()
    if max_rows is not None:
        df = df.head(int(max_rows))
    return df


def find_wildjailbreak_tsv(hf_home: str | Path) -> Path | None:
    """Locate the cached WildJailbreak ``train.tsv`` under an HF cache home (offline)."""
    hf_home = Path(hf_home)
    candidates = list(
        hf_home.glob("hub/datasets--allenai--wildjailbreak/snapshots/*/train/train.tsv")
    )
    candidates += list(
        hf_home.glob("datasets--allenai--wildjailbreak/snapshots/*/train/train.tsv")
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_wildjailbreak_pairs(
    hf_home: str | Path | None,
    *,
    data_type: str,
    max_pairs: int,
    min_vanilla_len: int = 5,
    min_adversarial_len: int = 20,
    chunksize: int = 20000,
) -> pd.DataFrame:
    """Read up to ``max_pairs`` (vanilla, adversarial) pairs of one data_type from the
    locally cached WildJailbreak TSV — offline, chunked, no network/datasets package.

    Raises FileNotFoundError with a clear message if the cache is missing.
    """
    if hf_home is None:
        raise FileNotFoundError(
            "HF cache home is unset; set HF_HOME / HARMPROBE_HF_HOME or use "
            "load_wildjailbreak_pairs_online()."
        )
    tsv = find_wildjailbreak_tsv(hf_home)
    if tsv is None:
        raise FileNotFoundError(
            "WildJailbreak train.tsv not found in HF cache "
            f"({hf_home}/hub/datasets--allenai--wildjailbreak/...). "
            "The dataset is gated; it must be cached locally for offline runs."
        )
    collected: list[dict[str, Any]] = []
    for chunk in pd.read_csv(
        tsv, sep="\t", on_bad_lines="skip", usecols=["vanilla", "adversarial", "data_type"], chunksize=chunksize
    ):
        sub = chunk[chunk["data_type"] == data_type]
        sub = sub[sub["vanilla"].notna() & sub["adversarial"].notna()]
        for _, row in sub.iterrows():
            van = str(row["vanilla"]).strip()
            adv = str(row["adversarial"]).strip()
            if len(van) > min_vanilla_len and len(adv) > min_adversarial_len:
                collected.append({"vanilla": van, "adversarial": adv})
            if len(collected) >= max_pairs:
                break
        if len(collected) >= max_pairs:
            break
    return pd.DataFrame(collected, columns=["vanilla", "adversarial"])


def load_wildjailbreak_pairs_online(
    hf_home: str | Path | None,
    *,
    data_type: str,
    max_pairs: int,
    source: str = "allenai/wildjailbreak",
    min_vanilla_len: int = 5,
    min_adversarial_len: int = 20,
) -> pd.DataFrame:
    """Load WildJailbreak pairs from local cache if present, otherwise download via HF Hub."""
    if hf_home is not None:
        try:
            return load_wildjailbreak_pairs(
                hf_home,
                data_type=data_type,
                max_pairs=max_pairs,
                min_vanilla_len=min_vanilla_len,
                min_adversarial_len=min_adversarial_len,
            )
        except FileNotFoundError:
            pass

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download WildJailbreak") from exc

    filepath = hf_hub_download(
        repo_id=source,
        filename="train/train.tsv",
        repo_type="dataset",
    )
    collected: list[dict[str, Any]] = []
    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        on_bad_lines="skip",
        usecols=["vanilla", "adversarial", "data_type"],
        chunksize=20000,
    ):
        sub = chunk[chunk["data_type"] == data_type]
        sub = sub[sub["vanilla"].notna() & sub["adversarial"].notna()]
        for _, row in sub.iterrows():
            van = str(row["vanilla"]).strip()
            adv = str(row["adversarial"]).strip()
            if len(van) > min_vanilla_len and len(adv) > min_adversarial_len:
                collected.append({"vanilla": van, "adversarial": adv})
            if len(collected) >= max_pairs:
                break
        if len(collected) >= max_pairs:
            break
    return pd.DataFrame(collected, columns=["vanilla", "adversarial"])


def report_row_count(path: Path | str) -> int | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return len(pd.read_csv(path, usecols=[0]))
    except (OSError, pd.errors.EmptyDataError, ValueError):
        return None
