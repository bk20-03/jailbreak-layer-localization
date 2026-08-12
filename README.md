# Jailbreak Layer Localization

**Main claim:** among *complied* generations, **adversarial harmful** vs **adversarial benign**
hidden states are **linearly separable**, and that linear signal is **strongest in middle layers**
(highest per-layer V-usable information under an L2 logistic probe).

**Main objective:** localize which transformer layers (and generation steps) encode
this harmfulness distinction — i.e. where hidden states most strongly separate
*adversarial harmful-complied* vs *adversarial benign-complied* responses.

This repo reproduces the full pipeline end-to-end:

1. **Dataset generation** — WildJailbreak `adversarial_harmful` and `adversarial_benign` pairs  
2. **Hidden-state extraction** — all layers × 100 generation steps → HDF5  
3. **Linear probing** — per-layer L2 logistic regression + **V-usable information**  

**Secondary comparison:** the same probes are run on a **base** checkpoint and a
**deep safety alignment** fine-tuned checkpoint, only to see whether fine-tuning
changes the internal harmfulness signal. Fine-tuning itself is **not** the goal of
this repository; see
[Unispac/shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment)
for that method. Do **not** subtract FT − base matrices (probes are trained independently).

Primary example figures are **per-layer profile grids** (Base vs Fine-tuned V-info
across steps for every layer), from anti-overfitting corrected probe runs.

---

## Finding: linear separability peaks in middle layers

Across **Llama-3.2-3B**, **Llama-2-7B-chat**, and **Qwen2.5-7B-Instruct** (corrected
probes: PCA + repeated grouped CV + 1-SE):

- **Early layers** (~first third): V-info near zero — little linearly recoverable
  harmful-vs-benign signal.
- **Middle layers**: **highest** mean V-info — adversarial harmful-complied vs
  adversarial benign-complied states are easiest to separate with a linear probe.
  Peak base layers ≈ **15** (Llama-3), **16** (Llama-2), **19** (Qwen).
- **Late layers**: still informative, but usually below the mid-layer peak.

Evidence figures (one subplot per layer; Base blue, Fine-tuned orange):

| Model | Figure |
|-------|--------|
| Llama-3 | [`examples/figures/llama3/harmful_vs_benign_vinfo_layer_profiles.png`](examples/figures/llama3/harmful_vs_benign_vinfo_layer_profiles.png) |
| Llama-2 | [`examples/figures/llama2/harmful_vs_benign_vinfo_layer_profiles.png`](examples/figures/llama2/harmful_vs_benign_vinfo_layer_profiles.png) |
| Qwen | [`examples/figures/qwen/harmful_vs_benign_vinfo_layer_profiles.png`](examples/figures/qwen/harmful_vs_benign_vinfo_layer_profiles.png) |

**Scope:** this is a **linear probe / V-info** localization result (where the distinction
is most recoverable), not a causal proof that middle layers alone “cause” jailbreaks.
Both classes are **complied** responses; this study does **not** contrast compliance vs refusal.

---

## Supported models

| Model | Config | Layers | Hidden dim | V-info matrix |
|-------|--------|--------|------------|---------------|
| Llama-3.2-3B-Instruct | `configs/models/llama3_3b.yaml` | 28 | 3072 | 28 × 100 |
| Llama-2-7B-chat | `configs/models/llama2_7b.yaml` | 32 | 4096 | 32 × 100 |
| Qwen2.5-7B-Instruct | `configs/models/qwen2_5_7b.yaml` | 28 | 3584 | 28 × 100 |

Each model has **base** and **fine-tuned** (DSA) checkpoints. Fine-tuned weights are
**not** redistributed — set `HARMPROBE_FT_MODEL` or YAML `model_path` /
`fine_tuned_model_path`.

---

## Requirements

- Python 3.10+
- CUDA GPU (dataset WildGuard stages + extraction)
- Hugging Face access to the chosen base model(s), `allenai/wildjailbreak`, and `allenai/wildguard`
- Fine-tuned checkpoint path for FT arms

---

## Install

```bash
git clone https://github.com/bk20-03/jailbreak-layer-localization.git
cd jailbreak-layer-localization
pip install -e ".[all]"
```

---

## Reproduce (as done)

### 0) Environment

```bash
export HF_TOKEN=...   # if models/datasets are gated
# Fine-tuned checkpoint for the model you are running:
export HARMPROBE_FT_MODEL=/path/to/your-finetuned-checkpoint
```

### 1) Generate datasets (GPU)

Build adversarial harmful / adversarial benign complied pairs:

| Model | Config |
|-------|--------|
| Llama-3 | `configs/datasets/llama3_task_b_dataset.yaml` |
| Llama-2 | `configs/datasets/llama2_task_b_dataset.yaml` |
| Qwen | `configs/datasets/qwen_task_b_dataset.yaml` |

```bash
python -m harmprobe.runners.run_dataset_pipeline \
  --config configs/datasets/llama3_task_b_dataset.yaml --dry-run

python -m harmprobe.runners.run_dataset_pipeline \
  --config configs/datasets/llama3_task_b_dataset.yaml --execute --device cuda
```

After generation, copy CSVs into the matching `data/prompts/` folder used by extraction (or keep using the committed production CSVs below).

**Skip generation:** production prompt CSVs are shipped under:

- `data/prompts/llama3/` — Llama-3  
- `data/prompts/llama2/` — Llama-2  
- `data/prompts/qwen/` — Qwen  

Extraction uses `max_samples: 102` per class (first 102 rows after filtering) to match published probe sizes. Shipped CSVs may be larger; only the capped subset is extracted.

### 2) Extract hidden states (GPU)

Example for Llama-3 (same pattern for `llama2_*` / `qwen_*` configs):

```bash
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class0_base.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class1_base.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class0_finetuned.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class1_finetuned.yaml
```

Llama-2 / Qwen:

```bash
python -m harmprobe.runners.run_extraction --config configs/extraction/llama2_class0_base.yaml
# ... class1_base, class0_finetuned, class1_finetuned

python -m harmprobe.runners.run_extraction --config configs/extraction/qwen_class0_base.yaml
# ... class1_base, class0_finetuned, class1_finetuned
```

HDF5 outputs (gitignored):

- `data/hiddens/llama3/`  
- `data/hiddens/llama2/`  
- `data/hiddens/qwen/`  

### 3) Run linear probes

Train per-layer probes and write V-info matrices / profile figures:

```bash
python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/llama3_task_b_base_ft.yaml

python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/llama2_task_b_base_ft.yaml

python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/qwen_task_b_base_ft.yaml
```

Primary figures from validated **corrected** runs (one subplot per layer; Base vs Fine-tuned):

[`examples/figures/`](examples/figures/) → `*/harmful_vs_benign_vinfo_layer_profiles.png`

These figures are the main evidence that middle-layer V-info is highest for
adversarial harmful-complied vs adversarial benign-complied separation.

The probe runner also writes this figure to
`runs/<experiment>/html_vinfo_step_dashboard/harmful_vs_benign_vinfo_layer_profiles.png`.

---

## Class / label semantics

| Canonical ID | Meaning |
|--------------|---------|
| **0** | Adversarial harmful prompt, model **complied** |
| **1** | Adversarial benign prompt, model **complied** |

Probe label map: class `1 → 0`, class `0 → 1` (see [`configs/tasks/task_b_benign_vs_harmful.yaml`](configs/tasks/task_b_benign_vs_harmful.yaml)).

---

## Reading the figures

Primary figure (`harmful_vs_benign_vinfo_layer_profiles.png`):

- One subplot **per layer** (all layers in one figure)  
- X-axis = generation step; Y-axis = V-info (bits)  
- Blue = base, orange = fine-tuned (DSA)  
- Higher curve = stronger **linear** recoverability of adversarial harmful vs
  adversarial benign (both complied)  
- **What to look for:** early layers near zero; **middle layers** with the tallest
  Base curves (peak linear separability); late layers still elevated but usually
  below the mid peak  
- Compare Base vs Fine-tuned visually; do not treat FT − base as a primary result  

---

## Repository layout

```text
configs/models/         llama3_3b, llama2_7b, qwen2_5_7b
configs/datasets/       WildJailbreak harmful/benign pipelines (per model)
configs/extraction/     class0/class1 × base/finetuned (per model)
configs/experiments/    probing runs (per model)
data/prompts/{llama3,llama2,qwen}/   committed prompt CSVs
data/hiddens/{llama3,llama2,qwen}/   extracted HDF5 (gitignored)
examples/figures/{llama3,llama2,qwen}/  reference layer-profile figures
src/harmprobe/          datasets + extraction + probing
```

---

## Tests

```bash
pip install -e .
pytest tests/ -q
```

---

## Notes

- Harmful + FT stages need **WildGuard on GPU**; benign uses a keyword refusal judge.  
- Extraction is GPU-heavy (full layer × 100 activations) and caps at **102 samples/class**.  
- Probing does not need a GPU once HDF5 files exist.  
- Fine-tuned checkpoints are **not** redistributed; set `HARMPROBE_FT_MODEL` or YAML `model_path` per model.  
- Published Qwen benign CSV is smaller (~265 rows) than the Llama dumps (~5k); extraction still takes the first 102.  
- Compliance vs refusal probing is intentionally out of scope.  
- Do **not** treat FT − base V-info differences as primary results (probes are trained independently).
