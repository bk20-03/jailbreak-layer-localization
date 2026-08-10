# Jailbreak Layer Localization (Task B)

Localize which transformer layers (and generation steps) encode **jailbreak-related harmfulness** in LLMs.

This repo reproduces **Task B** end-to-end as run in the HarmProbe / Llama-3 experiments:

1. **Dataset generation** — WildJailbreak `adversarial_harmful` and `adversarial_benign` pairs  
2. **Hidden-state extraction** — all layers × 100 generation steps → HDF5  
3. **Linear probing** — per-layer L2 logistic regression + **V-usable information** heatmaps  

**Question answered:** where in the network does the hidden state most strongly separate *harmful complied* vs *benign complied* responses?

Compare **base vs fine-tuned** models visually (side-by-side heatmaps). Do **not** subtract FT − base matrices — probes are trained independently.

---

## Requirements

- Python 3.10+
- CUDA GPU (dataset WildGuard stages + extraction)
- Hugging Face access:
  - `meta-llama/Llama-3.2-3B-Instruct` (gated)
  - `allenai/wildjailbreak` (gated)
  - `allenai/wildguard`
- Fine-tuned checkpoint path for FT arms (`HARMPROBE_FT_MODEL` or YAML `model_path` / `fine_tuned_model_path`)

Fine-tuned weights are **not** redistributed here.

---

## Install

```bash
git clone <your-repo-url> jailbreak-layer-localization
cd jailbreak-layer-localization
pip install -e ".[all]"
```

---

## Reproduce (as done)

### 0) Environment

```bash
export HF_TOKEN=...   # if models/datasets are gated
export HARMPROBE_FT_MODEL=/path/to/Llama-3.2-3B-Instruct__seed-42   # required for FT stages
```

### 1) Generate Task B datasets (GPU)

Builds:

| Stage | WildJailbreak type | Output |
|-------|--------------------|--------|
| `harmful_base_generation` | `adversarial_harmful` | `data/generated/llama3/paired_dataset1.csv` |
| `fine_tuned_compliance` | (from previous) | `data/generated/llama3/paired_dataset_finetuned.csv` (class 0) |
| `benign_generation` | `adversarial_benign` | `data/generated/llama3/paired_dataset_benign_llama3.csv` (class 1) |

```bash
python -m harmprobe.runners.run_dataset_pipeline \
  --config configs/datasets/llama3_task_b_dataset.yaml --dry-run

python -m harmprobe.runners.run_dataset_pipeline \
  --config configs/datasets/llama3_task_b_dataset.yaml --execute --device cuda
```

After generation, copy (or symlink) CSVs into the prompt paths used by extraction:

```bash
mkdir -p data/prompts
cp data/generated/llama3/paired_dataset_finetuned.csv data/prompts/
cp data/generated/llama3/paired_dataset_benign_llama3.csv data/prompts/
```

**Skip generation:** this repo already ships the production prompt CSVs under [`data/prompts/`](data/prompts/) from the validated Llama-3 Task B run. You can go straight to extraction.

### 2) Extract hidden states (GPU)

All layers, 100 steps, class 0 + class 1, base + fine-tuned:

```bash
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class0_base.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class1_base.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class0_finetuned.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class1_finetuned.yaml
```

Outputs land in `data/hiddens/*.h5` (~GB scale; gitignored).

### 3) Probe Task B (CPU/sklearn once H5 exists)

```bash
python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/llama3_task_b_base_ft.yaml
```

Results: `runs/llama3_task_b_base_ft/` — V-info matrices, heatmaps, HTML comparison.

Example heatmaps from the validated run are in [`examples/figures/`](examples/figures/).

---

## Class / label semantics

| Canonical ID | Meaning |
|--------------|---------|
| **0** | Adversarial harmful prompt, model **complied** |
| **1** | Adversarial benign prompt, model **complied** |

Task B probe labels: class `1 → 0`, class `0 → 1` (see [`configs/tasks/task_b_benign_vs_harmful.yaml`](configs/tasks/task_b_benign_vs_harmful.yaml)).

---

## Reading the heatmaps

- Rows = transformer layers, columns = generation steps  
- Cell value = **V-usable information** (bits) about harmful vs benign  
- Brighter / higher = stronger recoverable jailbreak-related signal at that layer×step  
- Inspect base and fine-tuned heatmaps separately; do not treat their difference as a primary result  

---

## Repository layout

```text
configs/datasets/     Task B WildJailbreak pipeline
configs/extraction/   class0/class1 × base/finetuned
configs/experiments/  probing run
configs/tasks/        Task B label map + H5 paths
data/prompts/         committed prompt CSVs (paper run)
data/generated/       pipeline CSV outputs (gitignored)
data/hiddens/         extracted HDF5 (gitignored)
examples/figures/     reference base / FT heatmaps
src/harmprobe/        datasets + extraction + probing
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
- Extraction is GPU-heavy (full 28×100 activations).  
- Probing does not need a GPU once HDF5 files exist.  
- Task A (compliance vs refusal / class 2) is intentionally out of scope.
