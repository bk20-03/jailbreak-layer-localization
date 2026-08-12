"""Unit tests for Task B probing core logic and extraction config hygiene."""

from pathlib import Path

import numpy as np
import yaml

import pandas as pd

from harmprobe.extraction.config_loader import load_extraction_config
from harmprobe.probing.corrected_overfit import select_c_one_se
from harmprobe.probing.metrics import compute_vinfo_bits, entropy_bits
from harmprobe.runners.config_loader import find_framework_root, load_experiment_config
from harmprobe.utils.splits import make_prompt_splits

def test_vinfo_increases_as_probe_improves():
    y = np.array([0, 0, 1, 1])
    random_probs = np.array([0.52, 0.48, 0.51, 0.49])
    good_probs = np.array([0.05, 0.1, 0.9, 0.95])
    v_random = compute_vinfo_bits(y, random_probs)
    v_good = compute_vinfo_bits(y, good_probs)
    assert v_good > v_random
    h = entropy_bits(y)
    ce_good = h - v_good
    assert v_good == h - ce_good

def test_task_b_split_sizes():
    y = np.concatenate([np.zeros(102, int), np.ones(102, int)])
    ids_train, ids_val, ids_test = make_prompt_splits(y, split_mode="holdout_30_then_50", seed=42)
    assert len(ids_train) == 142
    assert len(ids_val) == 31
    assert len(ids_test) == 31
    assert len(set(ids_train) & set(ids_val)) == 0
    assert len(set(ids_train) & set(ids_test)) == 0

def test_extraction_configs_have_matching_max_samples():
    root = find_framework_root()
    paths = sorted((root / "configs" / "extraction").glob("*_class*_*.yaml"))
    assert paths, "expected extraction configs"
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "max_samples" in raw, f"{path.name} missing max_samples"
        assert int(raw["max_samples"]) == 102, f"{path.name} max_samples != 102"
        assert "n_layers" in raw and "hidden_dim" in raw, f"{path.name} missing dims"
        assert "model_id" in raw, f"{path.name} missing model_id"
        cfg = load_extraction_config(path)
        assert cfg["max_samples"] == 102
        assert cfg["n_layers"] == int(raw["n_layers"])
        assert cfg["hidden_dim"] == int(raw["hidden_dim"])

def test_prompt_layout_per_model():
    root = find_framework_root()
    for family in ("llama3", "llama2", "qwen"):
        d = root / "data" / "prompts" / family
        assert (d / "paired_dataset_finetuned.csv").is_file(), family
        assert (d / "paired_dataset_benign.csv").is_file(), family

def test_select_c_one_se_picks_more_regularized():
    # Smaller C is more regularized. Best mean CE at C=1e-4; C=1e-5 within 1-SE.
    rows = []
    for c, means in [
        (1e-5, [0.495, 0.500, 0.505]),
        (1e-4, [0.490, 0.500, 0.510]),
        (1e-3, [0.600, 0.610, 0.590]),
    ]:
        for fold, ce in enumerate(means):
            rows.append({"C": c, "fold": fold, "ce_bits": ce, "auc": 0.7})
    best_c, summary = select_c_one_se(pd.DataFrame(rows))
    assert best_c == 1e-5
    assert not summary.empty

def test_corrected_experiment_config_enables_overfit_fix():
    root = find_framework_root()
    cfg = load_experiment_config(root / "configs" / "experiments" / "llama3_task_b_corrected.yaml")
    assert cfg["overfitting_correction"]["enabled"] is True
    assert cfg["overfitting_correction"]["pca_components"] == 30
    assert cfg["overfitting_correction"]["n_train_steps"] == 12
    assert min(cfg["c_values"]) <= 1e-7
