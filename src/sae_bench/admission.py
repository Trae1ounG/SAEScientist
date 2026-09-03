from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sources import require_official_source


@dataclass(frozen=True)
class AdmissionThresholds:
    min_heldout_prompts: int = 20
    min_target_delta_over_baseline: float = 0.20
    min_target_delta_over_random: float = 0.20
    min_success_rate: float = 0.70
    min_usable_target_rate: float = 0.50
    min_nondegenerate_rate: float = 0.90
    min_rerun_agreement: float = 0.80


@dataclass(frozen=True)
class ActivationThresholds:
    min_auroc: float = 0.95
    min_positive_active_rate: float = 0.80
    max_hard_negative_ratio: float = 0.30


def activation_failures(
    summary: dict[str, Any], thresholds: ActivationThresholds = ActivationThresholds()
) -> list[str]:
    failures = []
    if float(summary.get("auroc", 0.0)) < thresholds.min_auroc:
        failures.append("activation AUROC is below 0.95")
    if float(summary.get("positive_active_rate", 0.0)) < thresholds.min_positive_active_rate:
        failures.append("positive activation rate is below 0.80")
    if (
        float(summary.get("hard_negative_to_positive_ratio", 1.0))
        > thresholds.max_hard_negative_ratio
    ):
        failures.append("hard-negative activation ratio exceeds 0.30")
    return failures


def admission_failures(
    result: dict[str, Any], thresholds: AdmissionThresholds = AdmissionThresholds()
) -> list[str]:
    failures: list[str] = []
    source = result.get("source", {})
    try:
        official = require_official_source(source.get("repo", ""))
    except ValueError as error:
        failures.append(str(error))
        official = None

    if official is not None:
        if source.get("publisher") != official.publisher:
            failures.append("publisher does not match the official source")
        if source.get("base_model") != official.base_model:
            failures.append("base model does not match the official SAE")
        if not source.get("resolved_revision"):
            failures.append("official checkpoint revision was not resolved")

    evaluation = result.get("evaluation", {})
    heldout = int(evaluation.get("heldout_prompts", 0))
    baseline = float(evaluation.get("baseline_target_score", 0.0))
    feature = float(evaluation.get("feature_target_score", 0.0))
    random = float(evaluation.get("random_target_score", 0.0))

    if heldout < thresholds.min_heldout_prompts:
        failures.append("too few held-out prompts")
    if feature - baseline < thresholds.min_target_delta_over_baseline:
        failures.append("target effect over baseline is too small")
    if feature - random < thresholds.min_target_delta_over_random:
        failures.append("target effect over matched-random control is too small")
    if float(evaluation.get("feature_success_rate", 0.0)) < thresholds.min_success_rate:
        failures.append("feature success rate is too low")
    if (
        "usable_target_rate" in evaluation
        and float(evaluation["usable_target_rate"]) < thresholds.min_usable_target_rate
    ):
        failures.append("usable target rate is too low")
    if float(evaluation.get("nondegenerate_rate", 0.0)) < thresholds.min_nondegenerate_rate:
        failures.append("non-degenerate generation rate is too low")
    if float(evaluation.get("rerun_agreement", 0.0)) < thresholds.min_rerun_agreement:
        failures.append("rerun agreement is too low")
    if not evaluation.get("scorer"):
        failures.append("target scorer is missing")

    return failures


def is_admitted(result: dict[str, Any]) -> bool:
    return not admission_failures(result)

