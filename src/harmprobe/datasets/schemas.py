"""CSV schema definitions for Llama-3 dataset pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CANONICAL_CLASS_HARMFUL_COMPLIED = 0
CANONICAL_CLASS_BENIGN_COMPLIED = 1
CANONICAL_CLASS_HARMFUL_REFUSED = 2


@dataclass(frozen=True)
class CsvSchema:
    name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    prompt_columns: tuple[str, ...]
    response_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    canonical_class: int | None
    description: str


PAIRED_HARMFUL_SCHEMA = CsvSchema(
    name="paired_dataset1",
    required_columns=(
        "pair_id",
        "pair_seed",
        "vanilla_raw",
        "adversarial_raw",
        "vanilla_response",
        "adversarial_response",
        "vanilla_tokens",
        "adversarial_tokens",
        "vanilla_refused",
        "adversarial_harmful",
    ),
    optional_columns=(),
    prompt_columns=("vanilla_raw", "adversarial_raw"),
    response_columns=("vanilla_response", "adversarial_response"),
    label_columns=("vanilla_refused", "adversarial_harmful"),
    canonical_class=CANONICAL_CLASS_HARMFUL_COMPLIED,
    description=(
        "Harmful pairs: vanilla refused, adversarial harmful (base). "
        "Class 0 when FT also complies (see paired_dataset_finetuned)."
    ),
)

PAIRED_HARMFUL_FT_SCHEMA = CsvSchema(
    name="paired_dataset_finetuned",
    required_columns=PAIRED_HARMFUL_SCHEMA.required_columns,
    optional_columns=(),
    prompt_columns=PAIRED_HARMFUL_SCHEMA.prompt_columns,
    response_columns=PAIRED_HARMFUL_SCHEMA.response_columns,
    label_columns=PAIRED_HARMFUL_SCHEMA.label_columns,
    canonical_class=CANONICAL_CLASS_HARMFUL_COMPLIED,
    description="Harmful pairs regenerated with fine-tuned model; class 0 rows have adversarial_harmful==1.",
)

PAIRED_BENIGN_SCHEMA = CsvSchema(
    name="paired_dataset_benign_llama3",
    required_columns=(
        "pair_id",
        "pair_seed",
        "vanilla_raw",
        "adversarial_raw",
        "vanilla_response",
        "adversarial_response",
        "vanilla_tokens",
        "adversarial_tokens",
        "vanilla_refused",
        "adversarial_refused",
    ),
    optional_columns=(),
    prompt_columns=("vanilla_raw", "adversarial_raw"),
    response_columns=("vanilla_response", "adversarial_response"),
    label_columns=("vanilla_refused", "adversarial_refused"),
    canonical_class=CANONICAL_CLASS_BENIGN_COMPLIED,
    description="Benign pairs where both prompts received non-refusal responses (class 1).",
)

REFUSAL_SCHEMA = CsvSchema(
    name="adversarial_refused_by_base_and_ft",
    required_columns=(
        "prompt_id",
        "source",
        "data_type",
        "prompt_type",
        "prompt",
        "base_response",
        "ft_response",
        "base_refused",
        "ft_refused",
        "both_refused",
        "base_harmful",
        "ft_harmful",
        "prompt_tokens",
    ),
    optional_columns=(),
    prompt_columns=("prompt",),
    response_columns=("base_response", "ft_response"),
    label_columns=("base_refused", "ft_refused", "both_refused", "base_harmful", "ft_harmful"),
    canonical_class=CANONICAL_CLASS_HARMFUL_REFUSED,
    description="Adversarial harmful prompts refused by both base and fine-tuned models (class 2).",
)

SCHEMA_BY_NAME: dict[str, CsvSchema] = {
    "paired_dataset1": PAIRED_HARMFUL_SCHEMA,
    "paired_dataset_finetuned": PAIRED_HARMFUL_FT_SCHEMA,
    "paired_dataset_benign_llama3": PAIRED_BENIGN_SCHEMA,
    "adversarial_refused_by_base_and_ft": REFUSAL_SCHEMA,
}


def get_schema(name: str) -> CsvSchema:
    if name not in SCHEMA_BY_NAME:
        raise KeyError(f"Unknown schema: {name}. Known: {list(SCHEMA_BY_NAME)}")
    return SCHEMA_BY_NAME[name]


def schema_summary(schema: CsvSchema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "canonical_class": schema.canonical_class,
        "required_columns": list(schema.required_columns),
        "prompt_columns": list(schema.prompt_columns),
        "response_columns": list(schema.response_columns),
        "label_columns": list(schema.label_columns),
        "description": schema.description,
    }
