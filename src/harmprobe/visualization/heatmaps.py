"""Layer × step heatmap and profile-grid plots."""

from __future__ import annotations

import math
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


def plot_layer_profiles_grid(
    base_matrix: pd.DataFrame,
    ft_matrix: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Harmful-vs-benign V-info across generation steps for all layers",
    ncols: int = 4,
) -> Path:
    """Multi-panel figure: one subplot per layer, base vs fine-tuned V-info vs step.

    Matches the ``output1.png`` style used in the analysis write-up.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = base_matrix.copy()
    ft = ft_matrix.copy()
    base.index = base.index.astype(int)
    ft.index = ft.index.astype(int)
    base.columns = base.columns.astype(int)
    ft.columns = ft.columns.astype(int)
    layers = sorted(set(base.index).intersection(ft.index))
    steps = sorted(set(base.columns).intersection(ft.columns))
    if not layers or not steps:
        raise ValueError("base and fine-tuned matrices have no overlapping layers/steps")

    n_layers = len(layers)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_layers / ncols))

    all_vals = np.concatenate(
        [
            base.loc[layers, steps].to_numpy(dtype=float).ravel(),
            ft.loc[layers, steps].to_numpy(dtype=float).ravel(),
        ]
    )
    finite = all_vals[np.isfinite(all_vals)]
    ymin = float(min(0.0, np.nanmin(finite))) if finite.size else -0.05
    ymax = float(np.nanmax(finite)) if finite.size else 0.6
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ymin -= pad
    ymax += pad

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, y=0.995)

    base_line = None
    ft_line = None
    for i, layer in enumerate(layers):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        y_base = base.loc[layer, steps].to_numpy(dtype=float)
        y_ft = ft.loc[layer, steps].to_numpy(dtype=float)
        (base_line,) = ax.plot(steps, y_base, color="#1f77b4", linewidth=1.0, label="Base")
        (ft_line,) = ax.plot(steps, y_ft, color="#ff7f0e", linewidth=1.0, label="Fine-tuned")
        ax.axhline(0.0, color="#9ecae1", linestyle="--", linewidth=0.8)
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_ylim(ymin, ymax)
        if c == 0:
            ax.set_ylabel("V-info, bits")
        if r == nrows - 1:
            ax.set_xlabel("Generation step")

    for j in range(n_layers, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if base_line is not None and ft_line is not None:
        fig.legend(
            [base_line, ft_line],
            ["Base", "Fine-tuned"],
            loc="upper center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.5, 0.98),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
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
