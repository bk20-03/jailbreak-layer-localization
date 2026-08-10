"""Per-layer L2 logistic regression probes with best-C selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from harmprobe.probing.long_format import (
    build_long_split,
    build_long_split_with_prompt_weights,
)
from harmprobe.probing.metrics import evaluate_probe_metrics


@dataclass
class LayerProbeResult:
    layer: int
    best_c: float
    probe: LogisticRegression
    scaler: StandardScaler
    sweep_df: pd.DataFrame | None = None
    final_metrics_df: pd.DataFrame | None = None


def _make_logreg(
    c: float,
    *,
    class_weight,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=c,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )


def group_grid_search_c(
    x_train: np.ndarray,
    y_train: np.ndarray,
    prompt_train: np.ndarray,
    w_train: np.ndarray | None,
    *,
    c_values: list[float],
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 3000,
    class_weight=None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Task B C selection (probess_hard.ipynb cell 6).

    StratifiedGroupKFold runs on train long-format rows only.
    """
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        has_stratified = True
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        has_stratified = False

    n_groups = len(np.unique(prompt_train))
    n_splits = min(n_splits, n_groups)

    rows = []

    for c in c_values:
        if has_stratified:
            split_iter = splitter.split(x_train, y_train, groups=prompt_train)
        else:
            split_iter = splitter.split(x_train, y_train, groups=prompt_train)

        for fold, (idx_tr, idx_va) in enumerate(split_iter):
            x_tr = x_train[idx_tr]
            y_tr = y_train[idx_tr]
            w_tr = w_train[idx_tr] if w_train is not None else None

            x_va = x_train[idx_va]
            y_va = y_train[idx_va]
            w_va = w_train[idx_va] if w_train is not None else None

            scaler = StandardScaler()
            x_tr_s = scaler.fit_transform(x_tr)
            x_va_s = scaler.transform(x_va)

            probe = _make_logreg(
                c, class_weight=class_weight, max_iter=max_iter, seed=seed
            )
            if w_tr is not None:
                probe.fit(x_tr_s, y_tr, sample_weight=w_tr)
            else:
                probe.fit(x_tr_s, y_tr)

            probs_va = probe.predict_proba(x_va_s)[:, 1]
            m = evaluate_probe_metrics(
                y_va, probs_va, sample_weight=w_va, weighted_metrics=w_va is not None
            )
            m.update({"C": c, "fold": fold})
            rows.append(m)

    cv_fold_df = pd.DataFrame(rows)
    cv_summary_df = (
        cv_fold_df.groupby("C")
        .agg(
            mean_ce_bits=("ce_bits", "mean"),
            std_ce_bits=("ce_bits", "std"),
            mean_vinfo_bits=("vinfo_bits", "mean"),
            std_vinfo_bits=("vinfo_bits", "std"),
            mean_acc=("acc", "mean"),
            std_acc=("acc", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
        )
        .reset_index()
        .sort_values(["mean_ce_bits", "mean_auc"], ascending=[True, False])
    )

    best_c = float(cv_summary_df.iloc[0]["C"])
    return best_c, cv_summary_df, cv_fold_df


def train_one_layer_val_ce(
    *,
    layer: int,
    model_name: str,
    x_steps: np.ndarray,
    y_probe: np.ndarray,
    ids_train: np.ndarray,
    ids_val: np.ndarray,
    ids_test: np.ndarray,
    c_values: list[float],
    max_iter: int = 3000,
    seed: int = 42,
    class_weight="balanced",
) -> LayerProbeResult:
    """Task A training (probe.ipynb cell 78)."""
    x_train, y_train, _, _ = build_long_split(x_steps, y_probe, ids_train)
    x_val, y_val, _, _ = build_long_split(x_steps, y_probe, ids_val)
    x_test, y_test, _, _ = build_long_split(x_steps, y_probe, ids_test)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    sweep_rows = []
    candidates: dict[float, LogisticRegression] = {}

    for c in c_values:
        probe = _make_logreg(
            c, class_weight=class_weight, max_iter=max_iter, seed=seed
        )
        probe.fit(x_train_s, y_train)

        for split_name, xs, ys in [
            ("train", x_train_s, y_train),
            ("val", x_val_s, y_val),
            ("test", x_test_s, y_test),
        ]:
            probs = probe.predict_proba(xs)[:, 1]
            m = evaluate_probe_metrics(ys, probs, weighted_metrics=False)
            m.update(
                {
                    "model": model_name,
                    "layer": layer,
                    "split": split_name,
                    "C": c,
                    "training_mode": "long_steps",
                }
            )
            sweep_rows.append(m)
        candidates[c] = probe

    sweep_df = pd.DataFrame(sweep_rows)
    val_sweep = sweep_df[sweep_df["split"] == "val"].copy()
    val_sorted = val_sweep.sort_values(
        ["ce_bits", "auc", "f1"], ascending=[True, False, False]
    )
    best_c = float(val_sorted.iloc[0]["C"])
    best_probe = candidates[best_c]

    final_rows = []
    split_data = {
        "train": (x_train_s, y_train),
        "val": (x_val_s, y_val),
        "test": (x_test_s, y_test),
    }

    for split_name, (xs, ys) in split_data.items():
        probs = best_probe.predict_proba(xs)[:, 1]
        logits = best_probe.decision_function(xs)
        m = evaluate_probe_metrics(ys, probs, weighted_metrics=False)
        m.update(
            {
                "model": model_name,
                "layer": layer,
                "split": split_name,
                "best_C": best_c,
                "training_mode": "long_steps",
                "class0_mean_prob": float(probs[ys == 0].mean()),
                "class1_mean_prob": float(probs[ys == 1].mean()),
                "class0_mean_logit": float(logits[ys == 0].mean()),
                "class1_mean_logit": float(logits[ys == 1].mean()),
            }
        )
        m["prob_gap_refused_minus_complied"] = (
            m["class1_mean_prob"] - m["class0_mean_prob"]
        )
        m["logit_gap_refused_minus_complied"] = (
            m["class1_mean_logit"] - m["class0_mean_logit"]
        )
        final_rows.append(m)

    return LayerProbeResult(
        layer=layer,
        best_c=best_c,
        probe=best_probe,
        scaler=scaler,
        sweep_df=sweep_df,
        final_metrics_df=pd.DataFrame(final_rows),
    )


def train_one_layer_group_kfold(
    *,
    layer: int,
    model_name: str,
    x_steps: np.ndarray,
    y_probe: np.ndarray,
    ids_train: np.ndarray,
    ids_val: np.ndarray,
    ids_test: np.ndarray,
    c_values: list[float],
    max_iter: int = 3000,
    seed: int = 42,
    class_weight=None,
    cv_folds: int = 5,
) -> LayerProbeResult:
    """Task B training (probess_hard.ipynb cells 6–7)."""
    x_train, y_train, prompt_train, _, w_train = build_long_split_with_prompt_weights(
        x_steps, y_probe, ids_train
    )
    x_val, y_val, _, _, w_val = build_long_split_with_prompt_weights(
        x_steps, y_probe, ids_val
    )
    x_test, y_test, _, _, w_test = build_long_split_with_prompt_weights(
        x_steps, y_probe, ids_test
    )

    best_c, cv_summary_df, cv_fold_df = group_grid_search_c(
        x_train,
        y_train,
        prompt_train,
        w_train,
        c_values=c_values,
        n_splits=cv_folds,
        seed=seed,
        max_iter=max_iter,
        class_weight=class_weight,
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    probe = _make_logreg(
        best_c, class_weight=class_weight, max_iter=max_iter, seed=seed
    )
    probe.fit(x_train_s, y_train, sample_weight=w_train)

    final_rows = []
    for split_name, xs, ys, ws in [
        ("train", x_train_s, y_train, w_train),
        ("val", x_val_s, y_val, w_val),
        ("test", x_test_s, y_test, w_test),
    ]:
        probs = probe.predict_proba(xs)[:, 1]
        m = evaluate_probe_metrics(ys, probs, sample_weight=ws, weighted_metrics=True)
        m.update(
            {
                "model": model_name,
                "layer": layer,
                "split": split_name,
                "best_C": best_c,
                "training_mode": "group_gridsearch_prompt_weighted",
            }
        )
        final_rows.append(m)

    sweep_df = cv_fold_df.copy()
    sweep_df["cv_summary"] = True

    return LayerProbeResult(
        layer=layer,
        best_c=best_c,
        probe=probe,
        scaler=scaler,
        sweep_df=sweep_df,
        final_metrics_df=pd.DataFrame(final_rows),
    )


def run_layerwise_training(
    *,
    cv_mode: str,
    layers: list[int],
    model_name: str,
    load_layer_fn,
    ids_train: np.ndarray,
    ids_val: np.ndarray,
    ids_test: np.ndarray,
    c_values: list[float],
    max_iter: int = 3000,
    seed: int = 42,
    class_weight="balanced",
    cv_folds: int = 5,
    prompt_weighting: bool = False,
) -> tuple[dict[int, LayerProbeResult], pd.DataFrame]:
    results: dict[int, LayerProbeResult] = {}
    best_c_rows = []

    for layer in layers:
        x_steps, y_probe = load_layer_fn(layer)

        if cv_mode == "val_ce":
            result = train_one_layer_val_ce(
                layer=layer,
                model_name=model_name,
                x_steps=x_steps,
                y_probe=y_probe,
                ids_train=ids_train,
                ids_val=ids_val,
                ids_test=ids_test,
                c_values=c_values,
                max_iter=max_iter,
                seed=seed,
                class_weight=class_weight,
            )
        elif cv_mode == "group_kfold":
            result = train_one_layer_group_kfold(
                layer=layer,
                model_name=model_name,
                x_steps=x_steps,
                y_probe=y_probe,
                ids_train=ids_train,
                ids_val=ids_val,
                ids_test=ids_test,
                c_values=c_values,
                max_iter=max_iter,
                seed=seed,
                class_weight=class_weight,
                cv_folds=cv_folds,
            )
        else:
            raise ValueError(f"Unknown cv_mode: {cv_mode}")

        results[layer] = result
        best_c_rows.append({"layer": layer, "best_C": result.best_c})

    return results, pd.DataFrame(best_c_rows)
