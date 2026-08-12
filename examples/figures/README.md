Reference Task B figures (``output1.png`` style):

- ``harmful_vs_benign_vinfo_layer_profiles.png`` — one subplot per layer;
  Base (blue) vs Fine-tuned (orange) V-info across generation steps.

``llama3/``, ``llama2/``, and ``qwen/`` are from anti-overfitting corrected
probe runs (PCA + repeated grouped CV + 1-SE). FT−base difference plots are
intentionally omitted.

Heatmaps may still be produced by the probe runner as secondary artifacts;
these multi-panel profiles are the primary examples.
