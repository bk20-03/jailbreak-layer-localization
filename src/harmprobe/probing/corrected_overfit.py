"""Anti-overfitting Task B probing (PCA + 12-step train + repeated CV + 1-SE).

Faithful port of the recipe used for the published example figures
(``probess_hard.ipynb`` / framework ``run_corrected_task_b.py``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:  # pragma: no cover
    HAS_STRATIFIED_GROUP_KFOLD = False

from harmprobe.probing.metrics import evaluate_probe_metrics
from harmprobe.probing.stepwise_eval import class_balanced_weights

LoadLayerFn = Callable[[int], tuple[np.ndarray, np.ndarray]]


def select_c_one_se(cv_fold_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Pick the most regularized C within one SE of the best mean CE."""
    summary = (
        cv_fold_df.groupby("C")
        .agg(
            mean_ce_bits=("ce_bits", "mean"),
            std_ce_bits=("ce_bits", "std"),
            mean_auc=("auc", "mean"),
            n_folds=("ce_bits", "count"),
        )
        .reset_index()
        .sort_values("C")
    )
    best_idx = summary["mean_ce_bits"].idxmin()
    best_mean = summary.loc[best_idx, "mean_ce_bits"]
    best_std = summary.loc[best_idx, "std_ce_bits"]
    n_folds = summary.loc[best_idx, "n_folds"]
    se = best_std / np.sqrt(max(n_folds, 1))
    if not np.isfinite(se):
        se = 0.0
    eligible = summary[summary["mean_ce_bits"] <= best_mean + se]
    best_c = float(eligible.iloc[0]["C"])
    return best_c, summary


def build_outer_folds(
    y_probe: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    all_ids = np.arange(len(y_probe))
    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    folds = []
    for fold_global, (tr_idx, te_idx) in enumerate(outer_cv.split(all_ids, y_probe)):
        folds.append(
            {
                "repeat": fold_global // n_splits,
                "fold": fold_global % n_splits,
                "train_ids": all_ids[tr_idx],
                "test_ids": all_ids[te_idx],
            }
        )
    return folds


def build_long_split_n_steps(
    x_steps: np.ndarray,
    y_probe: np.ndarray,
    ids_split: np.ndarray,
    *,
    n_steps_keep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Long format with optional evenly spaced step subsample per prompt."""
    x_sub = x_steps[ids_split]
    y_sub = y_probe[ids_split]
    finite_mask = np.isfinite(x_sub).all(axis=2)
    nonzero_mask = np.linalg.norm(x_sub, axis=2) > 1e-8
    valid_mask = finite_mask & nonzero_mask

    rows, labels, prompt_ids, step_ids, weights = [], [], [], [], []
    for local_i, global_id in enumerate(ids_split):
        valid_steps = np.where(valid_mask[local_i])[0]
        if valid_steps.size == 0:
            continue
        if n_steps_keep is not None and valid_steps.size > n_steps_keep:
            sel = np.linspace(0, valid_steps.size - 1, n_steps_keep)
            sel = np.unique(np.round(sel).astype(int))
            kept_steps = valid_steps[sel]
        else:
            kept_steps = valid_steps
        row_weight = 1.0 / len(kept_steps)
        for step in kept_steps:
            rows.append(x_sub[local_i, step])
            labels.append(y_sub[local_i])
            prompt_ids.append(global_id)
            step_ids.append(int(step))
            weights.append(row_weight)

    if not rows:
        raise ValueError("No valid long-format rows for split")

    x_long = np.vstack(rows).astype(np.float32)
    y_long = np.array(labels, dtype=np.int64)
    prompt_long = np.array(prompt_ids, dtype=np.int64)
    step_long = np.array(step_ids, dtype=np.int64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.mean()
    return x_long, y_long, prompt_long, step_long, weights_arr


def fit_scaler_pca(
    x: np.ndarray, *, pca_components: int, seed: int
) -> tuple[Any, Any, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    n_comp = min(pca_components, xs.shape[1], max(1, xs.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    xp = pca.fit_transform(xs)
    return scaler, pca, xp


def apply_scaler_pca(scaler, pca, x: np.ndarray) -> np.ndarray:
    return pca.transform(scaler.transform(x))


def group_grid_search_c_corrected(
    x_train: np.ndarray,
    y_train: np.ndarray,
    prompt_train: np.ndarray,
    w_train: np.ndarray,
    *,
    c_values: list[float],
    n_splits: int,
    seed: int,
    max_iter: int,
    pca_components: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    n_groups = len(np.unique(prompt_train))
    n_splits = min(n_splits, n_groups)
    if HAS_STRATIFIED_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)

    rows = []
    for fold, (idx_tr, idx_va) in enumerate(
        splitter.split(x_train, y_train, groups=prompt_train)
    ):
        scaler, pca, x_tr_p = fit_scaler_pca(
            x_train[idx_tr], pca_components=pca_components, seed=seed
        )
        x_va_p = apply_scaler_pca(scaler, pca, x_train[idx_va])
        y_tr = y_train[idx_tr]
        w_tr = w_train[idx_tr]
        y_va = y_train[idx_va]
        w_va = w_train[idx_va]
        for c in c_values:
            clf = LogisticRegression(
                penalty="l2",
                C=c,
                class_weight=None,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=seed,
            )
            clf.fit(x_tr_p, y_tr, sample_weight=w_tr)
            probs_va = clf.predict_proba(x_va_p)[:, 1]
            m = evaluate_probe_metrics(
                y_va, probs_va, sample_weight=w_va, weighted_metrics=True
            )
            m.update({"C": c, "fold": fold})
            rows.append(m)

    cv_fold_df = pd.DataFrame(rows)
    best_c, cv_summary_df = select_c_one_se(cv_fold_df)
    return best_c, cv_summary_df, cv_fold_df


def run_one_layer_corrected(
    *,
    model_name: str,
    layer: int,
    x_steps: np.ndarray,
    y_probe: np.ndarray,
    outer_folds: list[dict[str, Any]],
    c_values: list[float],
    n_train_steps: int,
    pca_components: int,
    cv_folds: int,
    max_iter: int,
    seed: int,
    steps: list[int],
) -> dict[str, Any]:
    n_prompts = x_steps.shape[0]
    n_steps = len(steps)
    oof_prob_sum = np.zeros((n_prompts, n_steps), dtype=np.float64)
    oof_prob_cnt = np.zeros((n_prompts, n_steps), dtype=np.int64)
    fold_rows: list[dict] = []
    best_cs: list[float] = []

    for spec in outer_folds:
        train_ids = spec["train_ids"]
        test_ids = spec["test_ids"]
        x_tr, y_tr, prompt_tr, _, w_tr = build_long_split_n_steps(
            x_steps, y_probe, train_ids, n_steps_keep=n_train_steps
        )
        x_te, y_te, _, _, w_te = build_long_split_n_steps(
            x_steps, y_probe, test_ids, n_steps_keep=n_train_steps
        )
        best_c, _, _ = group_grid_search_c_corrected(
            x_tr,
            y_tr,
            prompt_tr,
            w_tr,
            c_values=c_values,
            n_splits=cv_folds,
            seed=seed,
            max_iter=max_iter,
            pca_components=pca_components,
        )
        best_cs.append(best_c)

        scaler, pca, x_tr_p = fit_scaler_pca(
            x_tr, pca_components=pca_components, seed=seed
        )
        clf = LogisticRegression(
            penalty="l2",
            C=best_c,
            class_weight=None,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
        clf.fit(x_tr_p, y_tr, sample_weight=w_tr)

        m_tr = evaluate_probe_metrics(
            y_tr, clf.predict_proba(x_tr_p)[:, 1], sample_weight=w_tr, weighted_metrics=True
        )
        x_te_p = apply_scaler_pca(scaler, pca, x_te)
        m_te = evaluate_probe_metrics(
            y_te, clf.predict_proba(x_te_p)[:, 1], sample_weight=w_te, weighted_metrics=True
        )
        for split_name, m in [("train", m_tr), ("test", m_te)]:
            m.update(
                {
                    "model": model_name,
                    "layer": layer,
                    "split": split_name,
                    "repeat": spec["repeat"],
                    "fold": spec["fold"],
                    "best_C": best_c,
                }
            )
            fold_rows.append(m)

        x_test_steps = x_steps[test_ids]
        for step_i, step in enumerate(steps):
            x_step = x_test_steps[:, step, :]
            finite = np.isfinite(x_step).all(axis=1)
            nonzero = np.linalg.norm(x_step, axis=1) > 1e-8
            valid = finite & nonzero
            if not valid.any():
                continue
            probs = clf.predict_proba(
                apply_scaler_pca(scaler, pca, x_step[valid])
            )[:, 1]
            valid_global = test_ids[valid]
            oof_prob_sum[valid_global, step_i] += probs
            oof_prob_cnt[valid_global, step_i] += 1

    fold_metrics_df = pd.DataFrame(fold_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        oof_prob = np.where(oof_prob_cnt > 0, oof_prob_sum / oof_prob_cnt, np.nan)

    step_rows = []
    for step_i, step in enumerate(steps):
        has_pred = oof_prob_cnt[:, step_i] > 0
        y_step = y_probe[has_pred]
        p_step = oof_prob[has_pred, step_i]
        if len(y_step) == 0 or len(np.unique(y_step)) < 2:
            step_rows.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "step": step,
                    "split": "test",
                    "status": "skip_one_class",
                    "n": int(has_pred.sum()),
                    "vinfo_bits": np.nan,
                    "ce_bits": np.nan,
                    "auc": np.nan,
                }
            )
            continue
        w_step = class_balanced_weights(y_step)
        m = evaluate_probe_metrics(
            y_step, p_step, sample_weight=w_step, weighted_metrics=True
        )
        m.update(
            {
                "model": model_name,
                "layer": layer,
                "step": step,
                "split": "test",
                "status": "ok",
                "evaluation_mode": "oof_stepwise_class_balanced",
            }
        )
        step_rows.append(m)

    return {
        "layer": layer,
        "fold_metrics_df": fold_metrics_df,
        "by_step_df": pd.DataFrame(step_rows),
        "best_cs": best_cs,
    }


def run_corrected_checkpoint(
    *,
    model_name: str,
    load_layer_fn: LoadLayerFn,
    layers: list[int],
    steps: list[int],
    y_probe: np.ndarray,
    c_values: list[float],
    pca_components: int = 30,
    n_train_steps: int = 12,
    cv_folds: int = 5,
    cv_repeats: int = 3,
    max_iter: int = 3000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run corrected probing for one checkpoint. Returns fold, by_step, best_c dfs."""
    outer_folds = build_outer_folds(
        y_probe, n_splits=cv_folds, n_repeats=cv_repeats, seed=seed
    )
    all_fold_metrics = []
    all_by_step = []
    best_c_records = []

    for layer in layers:
        print(
            f"{model_name} | layer {layer:02d} | {len(outer_folds)} outer folds",
            flush=True,
        )
        x_steps, y = load_layer_fn(layer)
        if len(y) != len(y_probe) or not np.array_equal(y, y_probe):
            # Labels come from H5 class order; keep the layer's y for this run.
            y_use = y
            folds = build_outer_folds(
                y_use, n_splits=cv_folds, n_repeats=cv_repeats, seed=seed
            )
        else:
            y_use = y_probe
            folds = outer_folds

        result = run_one_layer_corrected(
            model_name=model_name,
            layer=layer,
            x_steps=x_steps,
            y_probe=y_use,
            outer_folds=folds,
            c_values=c_values,
            n_train_steps=n_train_steps,
            pca_components=pca_components,
            cv_folds=cv_folds,
            max_iter=max_iter,
            seed=seed,
            steps=steps,
        )
        all_fold_metrics.append(result["fold_metrics_df"])
        all_by_step.append(result["by_step_df"])
        best_c_records.append(
            {"layer": layer, "best_C": float(np.median(result["best_cs"]))}
        )

    return (
        pd.concat(all_fold_metrics, ignore_index=True),
        pd.concat(all_by_step, ignore_index=True),
        pd.DataFrame(best_c_records),
    )
