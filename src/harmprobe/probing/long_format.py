"""Long-format expansion: one row per prompt × generation step."""

from __future__ import annotations

import numpy as np


def _valid_step_mask(x_sub: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(x_sub).all(axis=2)
    nonzero_mask = np.linalg.norm(x_sub, axis=2) > 1e-8
    return finite_mask & nonzero_mask


def build_long_split(
    x_steps: np.ndarray,
    y_probe: np.ndarray,
    ids_split: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Task A long format (probe.ipynb cell 77).

    Returns X_long, y_long, prompt_ids, step_ids.
    """
    x_sub = x_steps[ids_split]
    y_sub = y_probe[ids_split]
    n_prompts, n_steps, _ = x_sub.shape
    valid_mask = _valid_step_mask(x_sub)

    rows = []
    labels = []
    prompt_ids = []
    step_ids = []

    for local_i, global_id in enumerate(ids_split):
        for step in range(n_steps):
            if valid_mask[local_i, step]:
                rows.append(x_sub[local_i, step])
                labels.append(y_sub[local_i])
                prompt_ids.append(global_id)
                step_ids.append(step)

    if not rows:
        raise ValueError("No valid long-format rows for split")

    x_long = np.vstack(rows).astype(np.float32)
    y_long = np.array(labels, dtype=np.int64)
    prompt_long = np.array(prompt_ids, dtype=np.int64)
    step_long = np.array(step_ids, dtype=np.int64)
    return x_long, y_long, prompt_long, step_long


def build_long_split_with_prompt_weights(
    x_steps: np.ndarray,
    y_probe: np.ndarray,
    ids_split: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Task B long format (probess_hard.ipynb cell 4).

    Each prompt receives total weight ≈ 1 across its valid steps.
    """
    x_sub = x_steps[ids_split]
    y_sub = y_probe[ids_split]
    n_prompts, n_steps, _ = x_sub.shape
    valid_mask = _valid_step_mask(x_sub)
    valid_steps_per_prompt = valid_mask.sum(axis=1)

    rows = []
    labels = []
    prompt_ids = []
    step_ids = []
    weights = []

    for local_i, global_id in enumerate(ids_split):
        n_valid = int(valid_steps_per_prompt[local_i])
        if n_valid == 0:
            continue
        row_weight = 1.0 / n_valid

        for step in range(n_steps):
            if valid_mask[local_i, step]:
                rows.append(x_sub[local_i, step])
                labels.append(y_sub[local_i])
                prompt_ids.append(global_id)
                step_ids.append(step)
                weights.append(row_weight)

    if not rows:
        raise ValueError("No valid long-format rows for split")

    x_long = np.vstack(rows).astype(np.float32)
    y_long = np.array(labels, dtype=np.int64)
    prompt_long = np.array(prompt_ids, dtype=np.int64)
    step_long = np.array(step_ids, dtype=np.int64)
    w_long = np.array(weights, dtype=np.float64)
    return x_long, y_long, prompt_long, step_long, w_long
