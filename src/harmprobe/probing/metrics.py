"""Probe metrics: CE bits, V-info, AUC, accuracy, F1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def entropy_bits(y_true: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Class entropy H(Y) in bits."""
    y_true = np.asarray(y_true)

    if sample_weight is None:
        _, counts = np.unique(y_true, return_counts=True)
        p = counts / counts.sum()
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        weighted_counts = np.array(
            [sample_weight[y_true == c].sum() for c in (0, 1)],
            dtype=float,
        )
        p = weighted_counts / weighted_counts.sum()

    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_ce_bits(
    y_true: np.ndarray,
    probs: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Cross-entropy in bits."""
    return float(
        log_loss(
            y_true,
            probs,
            labels=[0, 1],
            sample_weight=sample_weight,
        )
        / np.log(2)
    )


def compute_vinfo_bits(
    y_true: np.ndarray,
    probs: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """V-usable information: H(Y) - CE(Y|probe) in bits."""
    h_y = entropy_bits(y_true, sample_weight=sample_weight)
    ce = compute_ce_bits(y_true, probs, sample_weight=sample_weight)
    return float(h_y - ce)


def evaluate_probe_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    weighted_metrics: bool = True,
) -> dict:
    """
    Evaluate binary probe predictions.

    ``probs`` is P(probe label 1).
    V-info = H(Y) - CE, so better probes yield higher vinfo_bits.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)

    ce_bits = compute_ce_bits(y_true, probs, sample_weight=sample_weight)
    h_y = entropy_bits(y_true, sample_weight=sample_weight)
    vinfo_bits = h_y - ce_bits

    try:
        auc = float(
            roc_auc_score(y_true, probs, sample_weight=sample_weight)
        )
    except ValueError:
        auc = float("nan")

    result = {
        "n": int(len(y_true)),
        "n_class0": int((y_true == 0).sum()),
        "n_class1": int((y_true == 1).sum()),
        "ce_bits": ce_bits,
        "entropy_bits": h_y,
        "vinfo_bits": vinfo_bits,
        "auc": auc,
    }

    if weighted_metrics and sample_weight is not None:
        result.update(
            {
                "acc": float(
                    accuracy_score(y_true, preds, sample_weight=sample_weight)
                ),
                "balanced_acc": float(
                    balanced_accuracy_score(
                        y_true, preds, sample_weight=sample_weight
                    )
                ),
                "f1": float(
                    f1_score(y_true, preds, sample_weight=sample_weight)
                ),
                "acc_weighted": float(
                    accuracy_score(y_true, preds, sample_weight=sample_weight)
                ),
                "f1_weighted": float(
                    f1_score(y_true, preds, sample_weight=sample_weight)
                ),
            }
        )
    else:
        result.update(
            {
                "acc": float(accuracy_score(y_true, preds)),
                "f1": float(f1_score(y_true, preds)),
                "acc_unweighted": float(accuracy_score(y_true, preds)),
                "f1_unweighted": float(f1_score(y_true, preds)),
            }
        )
        if sample_weight is not None:
            result["balanced_acc"] = float(
                balanced_accuracy_score(y_true, preds, sample_weight=sample_weight)
            )

    return result
