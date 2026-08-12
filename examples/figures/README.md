Reference Task B figures (anti-overfitting corrected runs):

- ``harmful_vs_benign_vinfo_layer_profiles.png`` — one subplot per layer;
  Base (blue) vs Fine-tuned (orange) V-info across generation steps.

**Claim these figures support:** adversarial harmful-complied vs adversarial
benign-complied hidden states are linearly separable, with **highest V-info in
middle layers** (early ≈ 0; late informative but usually below the mid peak).

``llama3/``, ``llama2/``, and ``qwen/`` all use corrected probes (PCA + repeated
grouped CV + 1-SE). FT−base difference plots are intentionally omitted.
