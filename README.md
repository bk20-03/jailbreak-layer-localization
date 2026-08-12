# Jailbreak Layer Localization

Find **which layers** of an LLM make **adversarial harmful** vs **adversarial benign**
answers easiest to tell apart with a simple linear probe.

**Short answer from the results:** early layers barely show the difference;
**middle layers** show it most clearly; late layers still show it, but usually less
than the middle. Example plots for three models live in [`examples/figures/`](examples/figures/).

---

## What this repo does

You can reproduce a full pipeline:

1. Build (or reuse) two sets of prompts where the model **complied**:
   - adversarial **harmful**
   - adversarial **benign**
2. Save hidden states for every layer across 100 generation steps.
3. Train a linear probe per layer and plot **V-info** (how separable the two
   classes are) over layers and steps.

Optional: run the same probes on a **base** model and a **safety fine-tuned**
model and compare the curves (blue vs orange in the figures).

---

## Look at the results first

Open these images (one panel per layer; blue = base, orange = fine-tuned):

| Model | Figure |
|-------|--------|
| Llama-3.2-3B | [examples/figures/llama3/…](examples/figures/llama3/harmful_vs_benign_vinfo_layer_profiles.png) |
| Llama-2-7B-chat | [examples/figures/llama2/…](examples/figures/llama2/harmful_vs_benign_vinfo_layer_profiles.png) |
| Qwen2.5-7B-Instruct | [examples/figures/qwen/…](examples/figures/qwen/harmful_vs_benign_vinfo_layer_profiles.png) |

**How to read a plot**

- X-axis = generation step (0 → 100)  
- Y-axis = V-info in bits (higher = easier for a linear probe to separate harmful vs benign)  
- Early layers ≈ flat near 0  
- Middle layers = highest curves (peak around layer **15** / **16** / **19** for Llama-3 / Llama-2 / Qwen)  
- Late layers = still up, usually below the mid peak  

Both classes are **complied** answers. This is about linear separability in
representations — not a proof that middle layers “cause” jailbreaks.

---

## Setup

**You need**

- Python 3.10+  
- A GPU for dataset generation and hidden-state extraction  
- Hugging Face access to your base model(s), `allenai/wildjailbreak`, and `allenai/wildguard`  
- (For orange curves) path to a deep-safety-aligned fine-tuned checkpoint  

**Install**

```bash
git clone https://github.com/bk20-03/jailbreak-layer-localization.git
cd jailbreak-layer-localization
pip install -e ".[all]"
```

```bash
export HF_TOKEN=...   # if needed
export HARMPROBE_FT_MODEL=/path/to/your-finetuned-checkpoint
```

Quick check (no GPU):

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

---

## Tutorial: run the pipeline

### Step 1 — Get prompts (or skip)

Shipped CSVs are already in:

- `data/prompts/llama3/`
- `data/prompts/llama2/`
- `data/prompts/qwen/`

Extraction uses the first **102** samples per class.

To **regenerate** with the GPU pipeline (optional):

```bash
python -m harmprobe.runners.run_dataset_pipeline \
  --config configs/datasets/llama3_task_b_dataset.yaml --dry-run

python -m harmprobe.runners.run_dataset_pipeline \
  --config configs/datasets/llama3_task_b_dataset.yaml --execute --device cuda
```

Same idea for `llama2_task_b_dataset.yaml` and `qwen_task_b_dataset.yaml`.

### Step 2 — Extract hidden states (GPU)

You need four runs per model: harmful/benign × base/finetuned.

```bash
# Llama-3
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class0_base.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class1_base.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class0_finetuned.yaml
python -m harmprobe.runners.run_extraction --config configs/extraction/llama3_class1_finetuned.yaml
```

Swap `llama3` → `llama2` or `qwen` in the config names for the other models.

HDF5 files go to `data/hiddens/{llama3,llama2,qwen}/` (not committed).

**Labels in files**

| File class | Meaning |
|------------|---------|
| class0 | Adversarial harmful, complied |
| class1 | Adversarial benign, complied |

### Step 3 — Probe and plot

Once the HDF5 files exist, probing can run on CPU:

```bash
python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/llama3_task_b_base_ft.yaml

python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/llama2_task_b_base_ft.yaml

python -m harmprobe.runners.run_probe_experiment \
  --config configs/experiments/qwen_task_b_base_ft.yaml
```

Look under `runs/<experiment>/html_vinfo_step_dashboard/` for:

- `*_test_vinfo_matrix.csv`
- `harmful_vs_benign_vinfo_layer_profiles.png`

The published example figures under `examples/figures/` come from stronger
anti-overfitting probe settings (PCA + cross-validation) matching the analysis plots.

---

## Models

| Model | Layers | Hidden size | Config |
|-------|--------|-------------|--------|
| Llama-3.2-3B-Instruct | 28 | 3072 | `configs/models/llama3_3b.yaml` |
| Llama-2-7B-chat | 32 | 4096 | `configs/models/llama2_7b.yaml` |
| Qwen2.5-7B-Instruct | 28 | 3584 | `configs/models/qwen2_5_7b.yaml` |

Fine-tuned weights are **not** included — set `HARMPROBE_FT_MODEL` yourself.
For how those FT checkpoints are trained, see
[shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment).

---

## Project layout

```text
configs/          YAML for datasets, extraction, probing
data/prompts/     Prompt CSVs (committed)
data/hiddens/     HDF5 activations (local)
examples/figures/ Ready-made result plots
src/harmprobe/    Code
tests/            Small unit tests
```

---

## Tips

- Dataset + extraction need GPU; probing does not (once HDF5 exists).  
- Compare base and fine-tuned curves separately — don’t subtract FT − base matrices.  
- This pipeline does **not** study refuse vs comply; only harmful-complied vs benign-complied.  
- Config filenames may still contain `task_b`; that is only a legacy name in the path.
