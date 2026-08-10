"""Simple HTML dashboards from exported CSV matrices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def matrix_to_json_safe(df: pd.DataFrame) -> list:
    arr = df.to_numpy(dtype=float)
    out = []
    for row in arr:
        out.append(
            [None if (v != v) else float(v) for v in row]  # NaN check
        )
    return out


def build_comparison_dashboard(
    matrices: dict[str, pd.DataFrame],
    *,
    output_path: Path,
    title: str,
    subtitle: str,
) -> Path:
    """
    Build a lightweight HTML dashboard with embedded matrix data.

    Expected keys in ``matrices``: at minimum ``base`` and ``fine_tuned``.
    Optional ``diff`` matrix.
    """
    base = matrices["base"]
    dashboard_data = {
        "title": title,
        "subtitle": subtitle,
        "layers": [int(x) for x in base.index.tolist()],
        "steps": [int(x) for x in base.columns.tolist()],
        "base": matrix_to_json_safe(base),
        "fine_tuned": matrix_to_json_safe(matrices["fine_tuned"]),
    }
    if "diff" in matrices:
        dashboard_data["diff"] = matrix_to_json_safe(matrices["diff"])

    dashboard_json = json.dumps(dashboard_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; }}
h1 {{ margin-bottom: 6px; }}
.subtitle {{ color: #555; margin-bottom: 20px; }}
.plot-card {{ background: white; padding: 16px; border-radius: 12px; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
<div class="plot-card"><div id="baseHeatmap" style="height:620px"></div></div>
<div class="plot-card"><div id="ftHeatmap" style="height:620px"></div></div>
<div class="plot-card"><div id="diffHeatmap" style="height:620px"></div></div>
<script>
const DATA = {dashboard_json};
function plotHeatmap(divId, z, plotTitle) {{
  Plotly.newPlot(divId, [{{
    z: z, x: DATA.steps, y: DATA.layers, type: 'heatmap',
    colorbar: {{ title: 'Value' }}
  }}], {{
    title: plotTitle,
    xaxis: {{ title: 'Generation step' }},
    yaxis: {{ title: 'Layer', dtick: 1 }},
    margin: {{ t: 50, l: 70, r: 20, b: 60 }}
  }});
}}
plotHeatmap('baseHeatmap', DATA.base, 'Base model');
plotHeatmap('ftHeatmap', DATA.fine_tuned, 'Fine-tuned model');
if (DATA.diff) plotHeatmap('diffHeatmap', DATA.diff, 'Fine-tuned minus base');
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
