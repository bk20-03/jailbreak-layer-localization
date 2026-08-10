"""Stage 2: fine-tuned compliance check (class 0 CSV for Task B).

Port of jailbreak_llama3/scripts/check_compliance_ft.py.
Re-generates responses with the fine-tuned model on the harmful base pairs and
keeps rows where vanilla is refused and adversarial is harmful (WildGuard).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from harmprobe.datasets.loaders import load_existing_csv, save_csv_safe, validate_csv_schema
from harmprobe.datasets.schemas import PAIRED_HARMFUL_FT_SCHEMA, PAIRED_HARMFUL_SCHEMA
from harmprobe.utils.text_generation import GenerationConfig, generate_responses, load_model_tokenizer
from harmprobe.utils.wildguard import WildGuardConfig, classify_batch, load_wildguard

STAGE_NAME = "fine_tuned_compliance"
CHECKPOINT_EVERY = 500


def plan_fine_tuned_compliance(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "input": cfg["outputs"]["harmful_base_csv"],
        "output": cfg["outputs"]["harmful_ft_csv"],
        "limit": cfg["limits"]["max_harmful_samples"],
        "model": cfg["model"]["fine_tuned_model_path"],
        "wildguard": cfg["wildguard"]["enabled"],
        "schema": PAIRED_HARMFUL_FT_SCHEMA.name,
        "canonical_class": PAIRED_HARMFUL_FT_SCHEMA.canonical_class,
        "class_filter": "adversarial_harmful == 1 for class 0 extraction",
        "status": "planned",
    }


def run_fine_tuned_compliance(cfg: dict[str, Any], *, dry_run: bool = True) -> dict[str, Any]:
    plan = plan_fine_tuned_compliance(cfg)
    in_path = Path(cfg["outputs"]["harmful_base_csv"])
    out_path = Path(cfg["outputs"]["harmful_ft_csv"])

    if in_path.is_file():
        plan["input_validation"] = validate_csv_schema(in_path, PAIRED_HARMFUL_SCHEMA)
    else:
        plan["input_validation"] = {"exists": False, "valid": False}

    if out_path.is_file():
        plan["output_validation"] = validate_csv_schema(out_path, PAIRED_HARMFUL_FT_SCHEMA)

    if dry_run:
        plan["message"] = "Dry-run: no FT generation or WildGuard calls"
        return plan

    execution = cfg.get("execution", {})
    device = execution.get("device", "auto")
    if cfg["wildguard"]["enabled"] and device == "cpu" and not execution.get(
        "allow_wildguard_on_cpu"
    ):
        plan["status"] = "blocked"
        plan["message"] = (
            "Blocked on CPU: fine-tuned compliance requires WildGuard on GPU. "
            "Re-run with --device cuda."
        )
        return plan

    ft_path = cfg["model"].get("fine_tuned_model_path") or ""
    if not ft_path:
        plan["status"] = "blocked"
        plan["message"] = (
            "fine_tuned_model_path is empty. Set it in the YAML or export HARMPROBE_FT_MODEL."
        )
        return plan

    if not in_path.is_file():
        plan["status"] = "blocked"
        plan["message"] = f"Missing harmful base CSV: {in_path}"
        return plan

    seed = int(cfg.get("seed", 42))
    limits = cfg["limits"]
    n_target = int(limits["max_harmful_samples"])
    batch_size = int(limits["batch_size"])

    df_in = load_existing_csv(in_path)
    vanilla_prompts = df_in["vanilla_raw"].astype(str).str.strip().tolist()
    adversarial_prompts = df_in["adversarial_raw"].astype(str).str.strip().tolist()
    n = len(vanilla_prompts)
    print(f"  [{STAGE_NAME}] loaded {n} base pairs from {in_path}")

    print(f"  [{STAGE_NAME}] loading fine-tuned model: {ft_path}")
    model, tokenizer = load_model_tokenizer(ft_path, dtype="bfloat16", local_files_only=False)

    # Ensure chat template when FT tokenizer lacks one
    if tokenizer.chat_template is None:
        base_path = cfg["model"]["base_model_path"]
        from transformers import AutoTokenizer

        base_tok = AutoTokenizer.from_pretrained(base_path)
        if base_tok.chat_template:
            tokenizer.chat_template = base_tok.chat_template
            print(f"  [{STAGE_NAME}] copied chat template from {base_path}")

    wg_cfg = WildGuardConfig(
        model_path=cfg["wildguard"]["model_path"],
        enabled=True,
        max_new_tokens=32,
        use_legacy_template=True,
    )
    print(f"  [{STAGE_NAME}] loading WildGuard: {wg_cfg.model_path}")
    wg_model, wg_tokenizer = load_wildguard(wg_cfg)

    gen_cfg = GenerationConfig(
        max_new_tokens=int(limits["max_new_tokens"]),
        max_input_tokens=int(limits["max_input_tokens"]),
        batch_size=batch_size,
        seed=seed,
    )

    records: list[dict[str, Any]] = []
    already_seen: set[int] = set()
    if out_path.is_file() and not cfg.get("overwrite", False):
        existing = pd.read_csv(out_path)
        if len(existing) > 0 and "pair_id" in existing.columns:
            records = existing.to_dict("records")
            already_seen = set(int(x) for x in existing["pair_id"].tolist())
            print(f"  [{STAGE_NAME}] resuming with {len(records)} existing pairs")

    n_processed = len(already_seen)
    for i in range(0, n, batch_size):
        if len(records) >= n_target:
            print(f"  [{STAGE_NAME}] reached target {n_target} clean pairs")
            break

        batch_van = vanilla_prompts[i : i + batch_size]
        batch_adv = adversarial_prompts[i : i + batch_size]
        actual = len(batch_van)
        batch_ids = list(range(i, i + actual))
        if all(pid in already_seen for pid in batch_ids):
            continue

        pair_seed = seed + i
        van_resp = generate_responses(model, tokenizer, batch_van, gen_cfg)
        adv_resp = generate_responses(model, tokenizer, batch_adv, gen_cfg)
        van_labels = classify_batch(
            wg_model, wg_tokenizer, list(zip(batch_van, van_resp)), wg_cfg
        )
        adv_labels = classify_batch(
            wg_model, wg_tokenizer, list(zip(batch_adv, adv_resp)), wg_cfg
        )

        for b in range(actual):
            pair_id = i + b
            if pair_id in already_seen:
                continue
            n_processed += 1
            van_refusal, _ = van_labels[b]
            _, adv_harmful = adv_labels[b]
            if van_refusal and adv_harmful:
                src = df_in.iloc[pair_id]
                records.append(
                    {
                        "pair_id": int(src.get("pair_id", pair_id)),
                        "pair_seed": pair_seed,
                        "vanilla_raw": batch_van[b],
                        "adversarial_raw": batch_adv[b],
                        "vanilla_response": van_resp[b],
                        "adversarial_response": adv_resp[b],
                        "vanilla_tokens": int(src.get("vanilla_tokens", 0)),
                        "adversarial_tokens": int(src.get("adversarial_tokens", 0)),
                        "vanilla_refused": 1,
                        "adversarial_harmful": 1,
                    }
                )
                already_seen.add(pair_id)

        if n_processed % CHECKPOINT_EVERY == 0 and records:
            pd.DataFrame(records).to_csv(out_path, index=False)
            print(f"  [{STAGE_NAME}] checkpoint {len(records)} pairs → {out_path}")

    df_out = pd.DataFrame(records, columns=list(PAIRED_HARMFUL_FT_SCHEMA.required_columns))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    plan["status"] = "completed"
    plan["processed_pairs"] = n_processed
    plan["compliant_pairs"] = len(df_out)
    plan["rows_written"] = len(df_out)
    plan["written"] = True
    plan["message"] = (
        f"FT compliance: processed {n_processed}, kept {len(df_out)} pairs → {out_path}"
    )
    plan["output_validation"] = validate_csv_schema(out_path, PAIRED_HARMFUL_FT_SCHEMA)
    return plan
