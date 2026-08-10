# Jailbreak Layer Localization (Task B)

Localize which transformer layers (and generation steps) encode **jailbreak-related harmfulness** in LLMs.

This repo reproduces **Task B** end-to-end:

1. **Dataset generation** — WildJailbreak `adversarial_harmful` and `adversarial_benign` pairs  
2. **Hidden-state extraction** — all layers × 100 generation steps → HDF5  
3. **Linear probing** — per-layer L2 logistic regression + **V-usable information** heatmaps  

**Question answered:** where in the network does the hidden state most strongly separate *harmful complied* vs *benign complied* responses?

Compare **base vs fine-tuned** models visually (side-by-side heatmaps). Do **not** subtract FT − base matrices — probes are trained independently.

---

## Supported models

| Model | Config | Layers | Hidden dim | Task B matrix |
|-------|--------|--------|------------|---------------|
| Llama-3.2-3B-Instruct | `configs/models/llama3_3b.yaml` | 28 | 3072 | 28 × 100 |
| Llama-2-7B-chat | `configs/models/llama2_7b.yaml` | 32 | 4096 | 32 × 100 |
| Qwen2.5-7B-Instruct | `configs/models/qwen2_5_7b.yaml` | 28 | 3584 | 28 × 100 |

Each model has **base** and **fine-tuned** checkpoints. Fine-tuned weights are **not** redistributed — set `HARMPROBE_FT_MODEL` or `model_path` / `fine_tuned_model_path` in YAML.

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

### 1) Generate Task B datasets (GPU)

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

### 3) Probe Task B

```bash
python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/llama3_task_b_base_ft.yaml

python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/llama2_task_b_base_ft.yaml

python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/qwen_task_b_base_ft.yaml
```

Example heatmaps from validated runs: [`examples/figures/`](examples/figures/) (`llama3/`, `llama2/`, `qwen/`).

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
configs/models/         llama3_3b, llama2_7b, qwen2_5_7b
configs/datasets/       Task B WildJailbreak pipelines (per model)
configs/extraction/     class0/class1 × base/finetuned (per model)
configs/experiments/    probing runs (per model)
data/prompts/{llama3,llama2,qwen}/   committed prompt CSVs
data/hiddens/{llama3,llama2,qwen}/   extracted HDF5 (gitignored)
examples/figures/{llama3,llama2,qwen}/  reference base / FT heatmaps
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
- Task A (compliance vs refusal / class 2) is intentionally out of scope.  
- Do **not** treat FT − base V-info differences as primary results (probes are trained independently).
