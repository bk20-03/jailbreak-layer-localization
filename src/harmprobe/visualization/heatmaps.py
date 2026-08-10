"""Layer × step heatmap plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_layer_step_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())

    step_cols = [int(c) for c in matrix.columns]
    tick_positions = list(range(0, len(step_cols), max(1, len(step_cols) // 10)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([step_cols[i] for i in tick_positions])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_diff_heatmap(
    diff_matrix: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> Path:
    vmax = np.nanmax(np.abs(diff_matrix.to_numpy(dtype=float)))
    return plot_layer_step_heatmap(
        diff_matrix,
        title=title,
        output_path=output_path,
        cmap="RdBu_r",
        vmin=-vmax if np.isfinite(vmax) else None,
        vmax=vmax if np.isfinite(vmax) else None,
    )
