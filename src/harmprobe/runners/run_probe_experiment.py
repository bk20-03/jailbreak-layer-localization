"""CLI runner for probing experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from harmprobe.probing.corrected_overfit import run_corrected_checkpoint
from harmprobe.probing.h5_loader import get_n_samples, load_layer_steps
from harmprobe.probing.logreg_layerwise import run_layerwise_training
from harmprobe.probing.stepwise_eval import (
    evaluate_probes_by_step_task_a,
    evaluate_probes_by_step_task_b,
    save_task_a_matrices,
    save_task_b_checkpoint_matrix,
)
from harmprobe.runners.config_loader import (
    load_experiment_config,
)
from harmprobe.runners.status_tracker import mark_completed, mark_failed, mark_running
from harmprobe.utils.splits import make_prompt_splits
from harmprobe.visualization.heatmaps import plot_layer_profiles_grid, plot_layer_step_heatmap
from harmprobe.visualization.html_dashboard import build_comparison_dashboard

def _canonical_checkpoint_name(checkpoint: str) -> str:
    """Map any checkpoint alias to the canonical run/output name."""
    return "base" if checkpoint == "base" else "fine_tuned"

def _build_y_probe(n_per_class: int, probe_label_map: dict[int, int], load_order: list[int]) -> np.ndarray:
    labels = []
    for cid in load_order:
        labels.append(np.full(n_per_class, probe_label_map[cid], dtype=np.int64))
    return np.concatenate(labels)

def run_experiment(config_path: Path) -> Path:
    cfg = load_experiment_config(config_path)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_running(output_dir)

    try:
        return _run_experiment_body(config_path, cfg, output_dir)
    except Exception as exc:
        mark_failed(output_dir, str(exc))
        raise

def _run_experiment_body(config_path: Path, cfg: dict, output_dir: Path) -> Path:
    task_type = cfg["task_type"]
    steps = cfg["steps"]
    layers = [int(l) for l in cfg["layers"]]
    seed = cfg["seed"]
    oc = cfg.get("overfitting_correction") or {}
    use_corrected = bool(oc.get("enabled")) and task_type == "benign_vs_harmful"

    # Build probe labels for split (one class file is enough for n_samples)
    first_ckpt = cfg["checkpoints"][0]
    first_class = cfg["load_order"][0]
    n_per_class = get_n_samples(cfg["h5_paths"][first_ckpt][first_class])
    y_probe = _build_y_probe(n_per_class, cfg["probe_label_map"], cfg["load_order"])

    checkpoint_store: dict[str, dict] = {}
    matrix_dir = output_dir / "html_vinfo_step_dashboard"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    if use_corrected:
        ids_train = ids_val = ids_test = np.array([], dtype=np.int64)
        for checkpoint in cfg["checkpoints"]:
            ck_name = _canonical_checkpoint_name(checkpoint)
            h5_by_class = cfg["h5_paths"][checkpoint]
            ck_out = output_dir / ck_name
            ck_out.mkdir(parents=True, exist_ok=True)

            def load_layer_fn(layer: int, _h5=h5_by_class):
                return load_layer_steps(
                    h5_files_by_class=_h5,
                    load_order=cfg["load_order"],
                    probe_label_map=cfg["probe_label_map"],
                    layer=layer,
                    steps=steps,
                )

            fold_df, by_step, best_c_df = run_corrected_checkpoint(
                model_name=ck_name,
                load_layer_fn=load_layer_fn,
                layers=layers,
                steps=steps,
                y_probe=y_probe,
                c_values=cfg["c_values"],
                pca_components=int(oc.get("pca_components", 30)),
                n_train_steps=int(oc.get("n_train_steps", 12)),
                cv_folds=int(oc.get("cv_folds", 5)),
                cv_repeats=int(oc.get("cv_repeats", 3)),
                max_iter=cfg["max_iter"],
                seed=seed,
            )
            best_c_df.to_csv(ck_out / "best_c.csv", index=False)
            fold_df.to_csv(ck_out / f"{ck_name}_fold_metrics.csv", index=False)
            by_step.to_csv(ck_out / f"{ck_name}_by_step.csv", index=False)
            matrix_path = save_task_b_checkpoint_matrix(
                by_step,
                checkpoint_name=ck_name,
                output_dir=matrix_dir,
                split="test",
                n_layers=cfg["n_layers"],
                steps=steps,
            )
            vinfo_mat = pd.read_csv(matrix_path, index_col=0)
            plot_layer_step_heatmap(
                vinfo_mat,
                title=f"{ck_name} test V-info (bits) [corrected]",
                output_path=matrix_dir / f"{ck_name}_test_vinfo_heatmap.png",
            )
            checkpoint_store[ck_name] = {
                "best_c_df": best_c_df,
                "by_step": by_step,
                "matrices": {"vinfo_bits": vinfo_mat},
                "matrix_path": matrix_path,
                "fold_metrics": fold_df,
            }
    else:
        ids_train, ids_val, ids_test = make_prompt_splits(
            y_probe,
            split_mode=cfg["split_mode"],
            test_size=cfg["test_size"],
            val_size=cfg["val_size"],
            seed=seed,
        )

        for checkpoint in cfg["checkpoints"]:
            ck_name = _canonical_checkpoint_name(checkpoint)
            h5_by_class = cfg["h5_paths"][checkpoint]
            ck_out = output_dir / ck_name
            ck_out.mkdir(parents=True, exist_ok=True)

            def load_layer_fn(layer: int, _h5=h5_by_class):
                return load_layer_steps(
                    h5_files_by_class=_h5,
                    load_order=cfg["load_order"],
                    probe_label_map=cfg["probe_label_map"],
                    layer=layer,
                    steps=steps,
                )

            layer_results, best_c_df = run_layerwise_training(
                cv_mode=cfg["cv_mode"],
                layers=layers,
                model_name=ck_name,
                load_layer_fn=load_layer_fn,
                ids_train=ids_train,
                ids_val=ids_val,
                ids_test=ids_test,
                c_values=cfg["c_values"],
                max_iter=cfg["max_iter"],
                seed=seed,
                class_weight=cfg["class_weight"],
                cv_folds=cfg["cv_folds"],
                prompt_weighting=cfg["prompt_weighting"],
            )

            best_c_df.to_csv(ck_out / "best_c.csv", index=False)
            ck_store: dict = {
                "layer_results": layer_results,
                "best_c_df": best_c_df,
            }

            if task_type == "compliance_vs_refusal":
                by_step = evaluate_probes_by_step_task_a(
                    model_name=ck_name,
                    load_layer_fn=load_layer_fn,
                    layer_results=layer_results,
                    ids_train=ids_train,
                    ids_val=ids_val,
                    ids_test=ids_test,
                    steps=steps,
                )
                saved = save_task_a_matrices(
                    by_step,
                    model_name=ck_name,
                    output_dir=ck_out,
                    split="test",
                    n_layers=cfg["n_layers"],
                    steps=steps,
                )
                vinfo_mat = pd.read_csv(saved["vinfo_bits"], index_col=0)
                plot_layer_step_heatmap(
                    vinfo_mat,
                    title=f"{ck_name} test V-info (bits)",
                    output_path=ck_out / f"{ck_name}_test_vinfo_heatmap.png",
                )
                ck_store["by_step"] = by_step
                ck_store["matrices"] = {"vinfo_bits": vinfo_mat}
            else:
                by_step = evaluate_probes_by_step_task_b(
                    model_name=ck_name,
                    load_layer_fn=load_layer_fn,
                    layer_results=layer_results,
                    ids_train=ids_train,
                    ids_val=ids_val,
                    ids_test=ids_test,
                    steps=steps,
                )
                by_step.to_csv(ck_out / f"{ck_name}_by_step.csv", index=False)
                matrix_path = save_task_b_checkpoint_matrix(
                    by_step,
                    checkpoint_name=ck_name,
                    output_dir=matrix_dir,
                    split="test",
                    n_layers=cfg["n_layers"],
                    steps=steps,
                )
                vinfo_mat = pd.read_csv(matrix_path, index_col=0)
                plot_layer_step_heatmap(
                    vinfo_mat,
                    title=f"{ck_name} test V-info (bits)",
                    output_path=matrix_dir / f"{ck_name}_test_vinfo_heatmap.png",
                )
                ck_store["by_step"] = by_step
                ck_store["matrices"] = {"vinfo_bits": vinfo_mat}
                ck_store["matrix_path"] = matrix_path

            checkpoint_store[ck_name] = ck_store

    present_checkpoints = list(checkpoint_store.keys())
    h5_sources = {
        _canonical_checkpoint_name(ck): {
            int(cid): path for cid, path in cfg["h5_paths"][ck].items()
        }
        for ck in cfg["checkpoints"]
    }

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "task_id": cfg["task_id"],
        "task_type": task_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "model_id": cfg.get("model_id"),
        "model_family": cfg.get("model_family"),
        "n_layers": cfg["n_layers"],
        "hidden_dim": cfg.get("hidden_dim"),
        "steps": list(steps),
        "n_steps": cfg["n_steps"],
        "checkpoints": present_checkpoints,
        "canonical_classes": cfg["canonical_classes"],
        "probe_label_map": cfg["probe_label_map"],
        "h5_sources": h5_sources,
        "split_sizes": {
            "train": int(len(ids_train)),
            "val": int(len(ids_val)),
            "test": int(len(ids_test)),
        },
        "matrix_shape": [cfg["n_layers"], cfg["n_steps"]],
        "overfitting_correction": oc if use_corrected else {"enabled": False},
    }

    # Visual comparison only: base and fine_tuned heatmaps side by side.
    # No FT - base difference matrix is generated (probes/scalers differ).
    has_base = "base" in checkpoint_store
    has_ft = "fine_tuned" in checkpoint_store

    if task_type == "compliance_vs_refusal":
        if has_base and has_ft:
            build_comparison_dashboard(
                {
                    "base": checkpoint_store["base"]["matrices"]["vinfo_bits"],
                    "fine_tuned": checkpoint_store["fine_tuned"]["matrices"]["vinfo_bits"],
                },
                output_path=output_dir / "layer_step_probe_dashboard.html",
                title="Class 0 vs 2 — layer × step V-info",
                subtitle="Test split. Base and fine-tuned shown separately (no subtraction).",
            )

    elif task_type == "benign_vs_harmful":
        matrix_paths = {
            f"{ck}_test_vinfo_matrix": str(store["matrix_path"])
            for ck, store in checkpoint_store.items()
        }
        if has_base and has_ft:
            profile_path = plot_layer_profiles_grid(
                checkpoint_store["base"]["matrices"]["vinfo_bits"],
                checkpoint_store["fine_tuned"]["matrices"]["vinfo_bits"],
                output_path=matrix_dir / "harmful_vs_benign_vinfo_layer_profiles.png",
            )
            matrix_paths["harmful_vs_benign_vinfo_layer_profiles"] = str(profile_path)
            build_comparison_dashboard(
                {
                    "base": checkpoint_store["base"]["matrices"]["vinfo_bits"],
                    "fine_tuned": checkpoint_store["fine_tuned"]["matrices"]["vinfo_bits"],
                },
                output_path=matrix_dir / "harmful_vs_benign_vinfo_step_dashboard.html",
                title="Class 1 vs 0 — harmful-vs-benign V-info",
                subtitle="Test split. Base and fine-tuned shown separately (no subtraction).",
            )
        manifest["matrix_paths"] = matrix_paths

    config_blob = json.dumps(cfg, sort_keys=True, default=str)
    manifest["config_hash"] = hashlib.sha256(config_blob.encode()).hexdigest()[:16]

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mark_completed(output_dir)
    print(f"Experiment complete. Outputs: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return output_dir

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run harmprobe experiment from YAML config")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML (e.g. configs/experiments/llama3_task_b_base_ft.yaml)",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    run_experiment(config_path)

if __name__ == "__main__":
    main()
