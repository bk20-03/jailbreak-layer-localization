"""Unit tests for Task B probing core logic."""

import numpy as np

from harmprobe.probing.metrics import compute_vinfo_bits, entropy_bits
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
