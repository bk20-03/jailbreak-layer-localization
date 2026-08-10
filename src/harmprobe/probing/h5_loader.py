"""HDF5 hidden-state loader with explicit canonical → probe label remapping."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _layer_step_key(layer: int, step: int) -> str:
    return f"layer{layer:02d}_step{step:02d}"


def validate_h5_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HDF5 file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Expected file, got: {p}")
    return p


def _load_one_class(h5_path: Path, layer: int, steps: list[int]) -> np.ndarray:
    step_arrays = []
    with h5py.File(h5_path, "r") as f:
        for step in steps:
            key = _layer_step_key(layer, step)
            if key not in f:
                raise KeyError(f"Missing key {key} in {h5_path}")
            arr = f[key][:].astype(np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D array for {key} in {h5_path}, got shape {arr.shape}"
                )
            step_arrays.append(arr)
    return np.stack(step_arrays, axis=1)


def load_layer_steps(
    *,
    h5_files_by_class: dict[int, str | Path],
    load_order: list[int],
    probe_label_map: dict[int, int],
    layer: int,
    steps: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one layer across selected steps from multiple class HDF5 files.

    Parameters
    ----------
    h5_files_by_class
        Mapping canonical class id → HDF5 path.
    load_order
        Order in which classes are stacked (e.g. [0, 2] or [1, 0]).
    probe_label_map
        Mapping canonical class id → sklearn probe binary label.
    """
    for cid in load_order:
        if cid not in h5_files_by_class:
            raise KeyError(f"Missing HDF5 path for canonical class {cid}")
        if cid not in probe_label_map:
            raise KeyError(f"Missing probe_label_map entry for canonical class {cid}")

    arrays = []
    labels = []

    n_samples_ref: int | None = None
    hidden_dim_ref: int | None = None

    for cid in load_order:
        path = validate_h5_path(h5_files_by_class[cid])
        x = _load_one_class(path, layer, steps)
        n_samples, n_steps, hidden_dim = x.shape

        if n_steps != len(steps):
            raise ValueError(
                f"Expected {len(steps)} steps from {path}, got {n_steps}"
            )
        if n_samples_ref is None:
            n_samples_ref = n_samples
            hidden_dim_ref = hidden_dim
        elif (n_samples, hidden_dim) != (n_samples_ref, hidden_dim_ref):
            raise ValueError(
                f"Shape mismatch for class {cid} in {path}: "
                f"{x.shape} vs reference ({n_samples_ref}, {len(steps)}, {hidden_dim_ref})"
            )

        arrays.append(x)
        labels.append(
            np.full(n_samples, probe_label_map[cid], dtype=np.int64)
        )

    x_steps = np.concatenate(arrays, axis=0)
    y_probe = np.concatenate(labels, axis=0)
    return x_steps, y_probe


def get_n_samples(h5_path: str | Path) -> int:
    path = validate_h5_path(h5_path)
    with h5py.File(path, "r") as f:
        key = _layer_step_key(0, 0)
        if key not in f:
            raise KeyError(f"Missing {key} in {path}")
        return int(f[key].shape[0])
