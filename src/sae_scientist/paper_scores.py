"""Current paper metrics from a completed activation and judge evaluation."""
from __future__ import annotations

import math
from typing import Any


SCORING_VERSION = "rank-auroc-steering-v1"


def finite_number(value: Any, name: str, low: float, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < low or (high is not None and number > high):
        raise ValueError(f"{name} is outside its valid range")
    return number


def paper_scores(activation: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    """Score one feature from activation.json and judgment_summary.json.

    Completion checks verify summary counts, not the underlying judge transcript.
    """
    counts = [judgment[k] for k in ("expected_rows", "valid_rows", "error_rows")]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in counts):
        raise ValueError("judge row counts must be integers")
    expected, valid, errors = counts
    if expected <= 0 or valid != expected or errors != 0:
        raise ValueError("judge evaluation must be complete with no errors")
    candidate = activation["activation_rank"]
    expert = activation["expert_activation_rank"]
    rank = finite_number(candidate["positive"]["mean_rank"], "candidate rank", 1)
    expert_rank = finite_number(expert["positive"]["mean_rank"], "Expert rank", 1)
    auroc = finite_number(candidate["activation_auroc"], "AUROC", 0, 1)
    target = {
        condition: finite_number(
            judgment["conditions"][condition]["target_relevance"],
            f"{condition} target relevance", 0, 4,
        )
        for condition in ("feature", "baseline", "random")
    }
    effect = (target["feature"] - max(target["baseline"], target["random"])) / 4
    metrics = {
        "Rank": 200 * expert_rank / (rank + expert_rank),
        "Activation": 100 * max(0, 2 * auroc - 1),
        "Steering": 100 * max(0, effect),
    }
    metrics["Overall"] = sum(metrics.values()) / 3
    return {
        "scoring_version": SCORING_VERSION,
        "scores": metrics,
        "measurements": {
            "positive_mean_rank": rank,
            "expert_positive_mean_rank": expert_rank,
            "activation_auroc": auroc,
            "target_relevance": target,
            "target_effect": effect,
        },
    }
