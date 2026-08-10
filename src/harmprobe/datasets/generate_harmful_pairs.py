"""Stage 1: generate harmful base pairs (adversarial_harmful → class 0 feed).

Port of jailbreak_llama3/scripts/generate_filter1.py into the config-driven pipeline.
A clean pair keeps vanilla refused + adversarial harmful (WildGuard).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from harmprobe.datasets.loaders import (
    load_wildjailbreak_pairs_online,
    save_csv_safe,
    validate_csv_schema,
)
from harmprobe.datasets.schemas import PAIRED_HARMFUL_SCHEMA
from harmprobe.utils.text_generation import GenerationConfig, generate_responses, load_model_tokenizer
from harmprobe.utils.wildguard import WildGuardConfig, classify_batch, load_wildguard

STAGE_NAME = "harmful_base_generation"
CHECKPOINT_EVERY = 500


def plan_harmful_base_generation(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "input": f"{cfg['dataset']['source']} ({cfg['dataset']['harmful_type']})",
        "output": cfg["outputs"]["harmful_base_csv"],
        "limit": cfg["limits"]["max_harmful_samples"],
        "model": cfg["model"]["base_model_path"],
        "wildguard": cfg["wildguard"]["enabled"],
        "schema": PAIRED_HARMFUL_SCHEMA.name,
        "canonical_class": PAIRED_HARMFUL_SCHEMA.canonical_class,
        "status": "planned",
    }


def run_harmful_base_generation(cfg: dict[str, Any], *, dry_run: bool = True) -> dict[str, Any]:
    plan = plan_harmful_base_generation(cfg)
    out_path = Path(cfg["outputs"]["harmful_base_csv"])

    if out_path.is_file():
        plan["existing_validation"] = validate_csv_schema(out_path, PAIRED_HARMFUL_SCHEMA)

    if dry_run:
        plan["message"] = "Dry-run: no generation or WildGuard calls"
        return plan

    execution = cfg.get("execution", {})
    device = execution.get("device", "auto")
    if cfg["wildguard"]["enabled"] and device == "cpu" and not execution.get(
        "allow_wildguard_on_cpu"
    ):
        plan["status"] = "blocked"
        plan["message"] = (
            "Blocked on CPU: harmful base generation requires WildGuard on GPU. "
            "Re-run with --device cuda."
        )
        return plan

    if not cfg["wildguard"]["enabled"]:
        plan["status"] = "blocked"
        plan["message"] = "wildguard.enabled must be true for harmful base generation"
        return plan

    seed = int(cfg.get("seed", 42))
    limits = cfg["limits"]
    n_target = int(limits["max_harmful_samples"])
    batch_size = int(limits["batch_size"])
    max_input = int(limits["max_input_tokens"])

    print(f"  [{STAGE_NAME}] loading WildJailbreak adversarial_harmful pairs...")
    pairs_df = load_wildjailbreak_pairs_online(
        cfg.get("hf_home"),
        data_type=cfg["dataset"]["harmful_type"],
        max_pairs=max(n_target * 8, 40000),
        source=cfg["dataset"]["source"],
    )
    if pairs_df.empty:
        plan["status"] = "blocked"
        plan["message"] = "No harmful pairs loaded from WildJailbreak"
        return plan

    print(f"  [{STAGE_NAME}] loading base model: {cfg['model']['base_model_path']}")
    model, tokenizer = load_model_tokenizer(
        cfg["model"]["base_model_path"],
        dtype="bfloat16",
        local_files_only=False,
    )

    # Filter long prompts
    van_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in pairs_df["vanilla"]]
    adv_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in pairs_df["adversarial"]]
    mask = [(v <= max_input and a <= max_input) for v, a in zip(van_lens, adv_lens)]
    pairs_df = pairs_df.loc[mask].reset_index(drop=True)
    pairs_df["vanilla_tokens"] = [van_lens[i] for i, keep in enumerate(mask) if keep]
    pairs_df["adversarial_tokens"] = [adv_lens[i] for i, keep in enumerate(mask) if keep]
    print(f"  [{STAGE_NAME}] {len(pairs_df)} pairs after token-length filter")

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
        max_input_tokens=max_input,
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
            print(f"  [{STAGE_NAME}] resuming with {len(records)} existing clean pairs")

    vanilla_prompts = pairs_df["vanilla"].str.strip().tolist()
    adversarial_prompts = pairs_df["adversarial"].str.strip().tolist()
    n = len(vanilla_prompts)
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
                records.append(
                    {
                        "pair_id": pair_id,
                        "pair_seed": pair_seed,
                        "vanilla_raw": batch_van[b],
                        "adversarial_raw": batch_adv[b],
                        "vanilla_response": van_resp[b],
                        "adversarial_response": adv_resp[b],
                        "vanilla_tokens": int(pairs_df.iloc[pair_id]["vanilla_tokens"]),
                        "adversarial_tokens": int(pairs_df.iloc[pair_id]["adversarial_tokens"]),
                        "vanilla_refused": 1,
                        "adversarial_harmful": 1,
                    }
                )
                already_seen.add(pair_id)

        if n_processed % CHECKPOINT_EVERY == 0 and records:
            pd.DataFrame(records).to_csv(out_path, index=False)
            print(f"  [{STAGE_NAME}] checkpoint {len(records)} pairs → {out_path}")

    df_out = pd.DataFrame(records, columns=list(PAIRED_HARMFUL_SCHEMA.required_columns))
    if out_path.exists() and not cfg.get("overwrite", False) and already_seen:
        # Resume path already wrote checkpoints; force final write
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        written = True
    else:
        save_info = save_csv_safe(
            df_out, out_path, overwrite=bool(cfg.get("overwrite", False)), dry_run=False
        )
        written = save_info["written"]

    plan["status"] = "completed"
    plan["processed_pairs"] = n_processed
    plan["compliant_pairs"] = len(df_out)
    plan["rows_written"] = len(df_out)
    plan["written"] = written
    plan["message"] = (
        f"Harmful base: processed {n_processed}, kept {len(df_out)} clean pairs → {out_path}"
    )
    plan["output_validation"] = validate_csv_schema(out_path, PAIRED_HARMFUL_SCHEMA)
    return plan


def validate_harmful_base_output(path: Path | str) -> dict[str, Any]:
    return validate_csv_schema(path, PAIRED_HARMFUL_SCHEMA)
