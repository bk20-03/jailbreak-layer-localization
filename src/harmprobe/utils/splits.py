"""Prompt-level train/val/test splits."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def make_prompt_splits_nested(
    y_probe: np.ndarray,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Task A split (probe.ipynb cells 4 + 13).

    Uses the same ``train_test_split`` call pattern as the notebook
    (including shuffle order via a dummy X array) so prompt IDs match exactly.
    """
    sample_ids = np.arange(len(y_probe))
    placeholder = np.zeros(len(y_probe), dtype=np.float32)

    (
        _,
        _,
        _,
        _,
        ids_trainval,
        ids_test,
    ) = train_test_split(
        placeholder,
        y_probe,
        sample_ids,
        test_size=test_size,
        random_state=seed,
        stratify=y_probe,
    )

    val_fraction_of_trainval = val_size / (1.0 - test_size)

    _, _, _, _, ids_train, ids_val = train_test_split(
        placeholder[ids_trainval],
        y_probe[ids_trainval],
        ids_trainval,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        stratify=y_probe[ids_trainval],
    )

    return np.asarray(ids_train, dtype=np.int64), np.asarray(ids_val, dtype=np.int64), np.asarray(ids_test, dtype=np.int64)


def make_prompt_splits_holdout_30_50(
    y_probe: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Task B split (probess_hard.ipynb cell 3):
    70% train, 15% val, 15% test via 30% holdout then 50/50 split.
    """
    all_ids = np.arange(len(y_probe))

    ids_train, ids_temp, _, y_temp = train_test_split(
        all_ids,
        y_probe,
        test_size=0.30,
        random_state=seed,
        stratify=y_probe,
    )

    ids_val, ids_test, _, _ = train_test_split(
        ids_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    return ids_train, ids_val, ids_test


def make_prompt_splits(
    y_probe: np.ndarray,
    *,
    split_mode: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_mode == "nested_15_15":
        return make_prompt_splits_nested(
            y_probe, test_size=test_size, val_size=val_size, seed=seed
        )
    if split_mode == "holdout_30_then_50":
        return make_prompt_splits_holdout_30_50(y_probe, seed=seed)
    raise ValueError(f"Unknown split_mode: {split_mode}")
