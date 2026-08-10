"""Stepwise evaluation and matrix export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from harmprobe.probing.logreg_layerwise import LayerProbeResult
from harmprobe.probing.metrics import evaluate_probe_metrics


def class_balanced_weights(y_true: np.ndarray) -> np.ndarray:
    """Task B step-level class-balanced weights (probess_hard.ipynb cell 11)."""
    y_true = np.asarray(y_true)
    weights = np.zeros(len(y_true), dtype=np.float64)
    for c in (0, 1):
        mask = y_true == c
        n_c = mask.sum()
        if n_c > 0:
            weights[mask] = 1.0 / n_c
    return weights / weights.mean()


def evaluate_probes_by_step_task_a(
    *,
    model_name: str,
    load_layer_fn,
    layer_results: dict[int, LayerProbeResult],
    ids_train: np.ndarray,
    ids_val: np.ndarray,
    ids_test: np.ndarray,
    steps: list[int],
) -> pd.DataFrame:
    """Task A stepwise eval (probe.ipynb cell 83)."""
    records = []
    split_ids = {"train": ids_train, "val": ids_val, "test": ids_test}

    for layer, result in sorted(layer_results.items()):
        x_steps, y_probe = load_layer_fn(layer)
        probe = result.probe
        scaler = result.scaler

        for step in steps:
            x_step = x_steps[:, step, :]
            finite_mask = np.isfinite(x_step).all(axis=1)
            nonzero_mask = np.linalg.norm(x_step, axis=1) > 1e-8
            valid_mask = finite_mask & nonzero_mask

            for split_name, ids_split in split_ids.items():
                split_valid = valid_mask[ids_split]
                x_split = x_step[ids_split][split_valid]
                y_split = y_probe[ids_split][split_valid]

                if len(np.unique(y_split)) < 2:
                    records.append(
                        {
                            "model": model_name,
                            "layer": layer,
                            "step": step,
                            "split": split_name,
                            "status": "skip_one_class",
                            "n": len(y_split),
                        }
                    )
                    continue

                x_split_s = scaler.transform(x_split)
                probs = probe.predict_proba(x_split_s)[:, 1]
                logits = probe.decision_function(x_split_s)

                m = evaluate_probe_metrics(y_split, probs, weighted_metrics=False)
                m.update(
                    {
                        "model": model_name,
                        "layer": layer,
                        "step": step,
                        "split": split_name,
                        "status": "ok",
                        "training_mode": "long_steps",
                        "class0_mean_prob": float(probs[y_split == 0].mean()),
                        "class1_mean_prob": float(probs[y_split == 1].mean()),
                        "class0_mean_logit": float(logits[y_split == 0].mean()),
                        "class1_mean_logit": float(logits[y_split == 1].mean()),
                    }
                )
                m["prob_gap_refused_minus_complied"] = (
                    m["class1_mean_prob"] - m["class0_mean_prob"]
                )
                m["logit_gap_refused_minus_complied"] = (
                    m["class1_mean_logit"] - m["class0_mean_logit"]
                )
                records.append(m)

    return pd.DataFrame(records)


def evaluate_probes_by_step_task_b(
    *,
    model_name: str,
    load_layer_fn,
    layer_results: dict[int, LayerProbeResult],
    ids_train: np.ndarray,
    ids_val: np.ndarray,
    ids_test: np.ndarray,
    steps: list[int],
) -> pd.DataFrame:
    """Task B stepwise eval (probess_hard.ipynb cell 11)."""
    records = []
    split_ids = {"train": ids_train, "val": ids_val, "test": ids_test}

    for layer, result in sorted(layer_results.items()):
        x_steps, y_probe = load_layer_fn(layer)
        probe = result.probe
        scaler = result.scaler

        for step in steps:
            x_step = x_steps[:, step, :]
            finite_mask = np.isfinite(x_step).all(axis=1)
            nonzero_mask = np.linalg.norm(x_step, axis=1) > 1e-8
            valid_mask = finite_mask & nonzero_mask

            for split_name, ids_split in split_ids.items():
                split_valid = valid_mask[ids_split]
                x_split = x_step[ids_split][split_valid]
                y_split = y_probe[ids_split][split_valid]

                if len(y_split) == 0 or len(np.unique(y_split)) < 2:
                    records.append(
                        {
                            "model": model_name,
                            "layer": layer,
                            "step": step,
                            "split": split_name,
                            "status": "skip_one_class",
                            "n": len(y_split),
                            "vinfo_bits": np.nan,
                        }
                    )
                    continue

                x_split_s = scaler.transform(x_split)
                probs = probe.predict_proba(x_split_s)[:, 1]
                w_step = class_balanced_weights(y_split)

                m = evaluate_probe_metrics(
                    y_split, probs, sample_weight=w_step, weighted_metrics=True
                )
                m.update(
                    {
                        "model": model_name,
                        "layer": layer,
                        "step": step,
                        "split": split_name,
                        "status": "ok",
                        "evaluation_mode": "stepwise_class_balanced",
                    }
                )
                records.append(m)

    return pd.DataFrame(records)


def save_task_a_matrices(
    df: pd.DataFrame,
    *,
    model_name: str,
    output_dir: Path,
    split: str = "test",
    n_layers: int = 28,
    steps: list[int],
) -> dict[str, Path]:
    """Export Task A layer × step matrices (probe.ipynb cell 97)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sub = df[
        (df["split"] == split)
        & (df["status"] == "ok")
        & (df["step"].isin(steps))
    ].copy()

    metrics = [
        "vinfo_bits",
        "prob_gap_refused_minus_complied",
        "logit_gap_refused_minus_complied",
    ]
    saved: dict[str, Path] = {}

    for metric in metrics:
        matrix = sub.pivot(index="layer", columns="step", values=metric)
        matrix = matrix.reindex(index=range(n_layers), columns=steps)
        path = output_dir / f"{model_name}_{split}_{metric}_matrix.csv"
        matrix.to_csv(path)
        saved[metric] = path

    compact_cols = [
        c
        for c in [
            "model",
            "split",
            "layer",
            "step",
            "n",
            "vinfo_bits",
            "prob_gap_refused_minus_complied",
            "logit_gap_refused_minus_complied",
        ]
        if c in sub.columns
    ]
    compact_path = output_dir / f"{model_name}_{split}_core_long_results.csv"
    sub[compact_cols].to_csv(compact_path, index=False)
    saved["core_long_table"] = compact_path
    return saved


def save_task_b_checkpoint_matrix(
    by_step_df: pd.DataFrame,
    *,
    checkpoint_name: str,
    output_dir: Path,
    split: str = "test",
    n_layers: int = 28,
    steps: list[int],
) -> Path:
    """Export a single checkpoint's Task B V-info matrix (no FT-base diff).

    ``checkpoint_name`` must be the canonical name ("base" or "fine_tuned").
    Written as ``{checkpoint_name}_{split}_vinfo_matrix.csv`` so the dashboard
    can discover per-checkpoint matrices for any model/shape.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sub = by_step_df[
        (by_step_df["split"] == split)
        & (by_step_df["status"] == "ok")
        & (by_step_df["step"].isin(steps))
    ]
    matrix = sub.pivot(index="layer", columns="step", values="vinfo_bits")
    matrix = matrix.reindex(index=range(n_layers), columns=steps)
    path = output_dir / f"{checkpoint_name}_{split}_vinfo_matrix.csv"
    matrix.to_csv(path)
    return path


def save_task_b_matrices(
    base_df: pd.DataFrame,
    ft_df: pd.DataFrame,
    *,
    output_dir: Path,
    split: str = "test",
    n_layers: int = 28,
    steps: list[int],
) -> dict[str, Path]:
    """Export Task B V-info matrices and FT − base diff."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    def _vinfo_matrix(df: pd.DataFrame, name: str) -> pd.DataFrame:
        sub = df[
            (df["split"] == split)
            & (df["status"] == "ok")
            & (df["step"].isin(steps))
        ]
        matrix = sub.pivot(index="layer", columns="step", values="vinfo_bits")
        return matrix.reindex(index=range(n_layers), columns=steps)

    base_mat = _vinfo_matrix(base_df, "base")
    ft_mat = _vinfo_matrix(ft_df, "fine_tuned")
    diff_mat = ft_mat - base_mat

    paths = {
        "base_test_vinfo_matrix": output_dir / "base_test_vinfo_matrix.csv",
        "fine_tuned_test_vinfo_matrix": output_dir
        / "fine_tuned_test_vinfo_matrix.csv",
        "diff_fine_tuned_minus_base_test_vinfo_matrix": output_dir
        / "diff_fine_tuned_minus_base_test_vinfo_matrix.csv",
    }
    base_mat.to_csv(paths["base_test_vinfo_matrix"])
    ft_mat.to_csv(paths["fine_tuned_test_vinfo_matrix"])
    diff_mat.to_csv(paths["diff_fine_tuned_minus_base_test_vinfo_matrix"])
    saved.update(paths)
    return saved
