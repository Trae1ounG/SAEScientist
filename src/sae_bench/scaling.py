from __future__ import annotations

from typing import Any, Iterable


def merge_discovery_batches(
    batches: Iterable[dict[str, Any]], excluded_ids: set[str] | None = None
) -> dict[str, Any]:
    """Validate and merge independently authored discovery concept batches."""

    excluded_ids = excluded_ids or set()
    checkpoint = None
    concepts: list[dict[str, Any]] = []
    seen_ids = set(excluded_ids)

    for batch in batches:
        if batch.get("schema") != 1:
            raise ValueError("each discovery batch must use schema 1")
        batch_checkpoint = batch.get("checkpoint")
        if not isinstance(batch_checkpoint, str) or not batch_checkpoint:
            raise ValueError("each discovery batch must name its checkpoint")
        if checkpoint is None:
            checkpoint = batch_checkpoint
        elif checkpoint != batch_checkpoint:
            raise ValueError("discovery batches use different checkpoints")

        for concept in batch.get("concepts", []):
            concept_id = concept.get("id")
            if not isinstance(concept_id, str) or not concept_id:
                raise ValueError("every concept needs a non-empty id")
            if concept_id in seen_ids:
                raise ValueError(f"duplicate or excluded concept id: {concept_id}")
            if not isinstance(concept.get("target"), str) or not concept["target"].strip():
                raise ValueError(f"concept {concept_id} needs a target")

            positive = concept.get("positive")
            negative = concept.get("negative")
            if not isinstance(positive, list) or len(positive) != 6:
                raise ValueError(f"concept {concept_id} needs exactly 6 positives")
            if not isinstance(negative, list) or len(negative) != 6:
                raise ValueError(f"concept {concept_id} needs exactly 6 negatives")
            examples = positive + negative
            if not all(isinstance(text, str) and text.strip() for text in examples):
                raise ValueError(f"concept {concept_id} contains an empty example")
            if len(set(examples)) != len(examples):
                raise ValueError(f"concept {concept_id} contains duplicate examples")

            seen_ids.add(concept_id)
            concepts.append(concept)

    if checkpoint is None or not concepts:
        raise ValueError("no discovery concepts were provided")
    return {"schema": 1, "checkpoint": checkpoint, "concepts": concepts}


def merge_validation_batches(
    batches: Iterable[dict[str, Any]], expected_ids: set[str]
) -> dict[str, Any]:
    """Validate and merge held-out activation suites for one discovery round."""

    checkpoint = None
    calibration = None
    steering_prompts = None
    concepts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_case_ids: set[str] = set()

    for batch in batches:
        if batch.get("schema") != 1:
            raise ValueError("each validation batch must use schema 1")
        shared = (batch.get("checkpoint"), batch.get("steering_prompts"))
        if checkpoint is None:
            checkpoint, steering_prompts = shared
            calibration = batch.get("steering_calibration")
        elif shared != (checkpoint, steering_prompts):
            raise ValueError("validation batches use different shared settings")

        for concept in batch.get("concepts", []):
            concept_id = concept.get("id")
            if concept_id in seen_ids:
                raise ValueError(f"duplicate validation concept id: {concept_id}")
            if concept_id not in expected_ids:
                raise ValueError(f"unexpected validation concept id: {concept_id}")
            cases = concept.get("activation_cases", [])
            counts = {
                label: sum(case.get("label") == label for case in cases)
                for label in ("positive", "hard_negative", "neutral")
            }
            if counts != {"positive": 8, "hard_negative": 8, "neutral": 4}:
                raise ValueError(f"concept {concept_id} has invalid activation case counts")
            case_ids = [case.get("id") for case in cases]
            texts = [case.get("text") for case in cases]
            if not all(isinstance(value, str) and value.strip() for value in case_ids + texts):
                raise ValueError(f"concept {concept_id} contains an empty activation case")
            if len(set(case_ids)) != len(case_ids) or seen_case_ids.intersection(case_ids):
                raise ValueError(f"concept {concept_id} contains duplicate case ids")
            if len(set(texts)) != len(texts):
                raise ValueError(f"concept {concept_id} contains duplicate activation texts")
            target = concept.get("steering_target", {})
            required = {"concept", "cues", "strong_evidence", "insufficient_evidence"}
            if not required.issubset(target) or not target.get("cues"):
                raise ValueError(f"concept {concept_id} has an incomplete steering target")
            seen_ids.add(concept_id)
            seen_case_ids.update(case_ids)
            concepts.append(concept)

    missing = expected_ids - seen_ids
    if missing:
        raise ValueError(f"validation concepts are missing: {', '.join(sorted(missing))}")
    if checkpoint is None or not isinstance(calibration, list) or len(calibration) != 5:
        raise ValueError("validation suite needs five shared calibration prompts")
    if not isinstance(steering_prompts, str) or not steering_prompts:
        raise ValueError("validation suite needs a steering prompt path")
    from .suites import target_score

    contaminated = [
        concept["id"]
        for concept in concepts
        if any(target_score(row["prompt"], concept) > 0 for row in calibration)
    ]
    if contaminated:
        raise ValueError(
            "shared calibration prompts contain target cues for: "
            + ", ".join(contaminated)
        )
    return {
        "schema": 1,
        "checkpoint": checkpoint,
        "steering_calibration": calibration,
        "steering_prompts": steering_prompts,
        "concepts": concepts,
    }


def select_unique_activation_candidates(
    concepts: Iterable[dict[str, Any]],
    excluded_feature_ids: set[int] | None = None,
    max_per_concept: int = 1,
) -> list[dict[str, Any]]:
    """Select the best activation-qualified feature per concept without ID reuse."""

    if max_per_concept < 1:
        raise ValueError("max_per_concept must be positive")
    used = set(excluded_feature_ids or set())
    selected = []
    for concept in concepts:
        kept = 0
        for candidate in concept.get("candidates", []):
            feature_id = int(candidate["feature_id"])
            if not candidate.get("activation_stable") or feature_id in used:
                continue
            used.add(feature_id)
            selected.append(
                {
                    "concept_id": concept["id"],
                    "feature_id": feature_id,
                    "activation_auroc": candidate["activation_auroc"],
                    "positive_active_rate": candidate["positive_active_rate"],
                    "hard_negative_to_positive_ratio": candidate[
                        "hard_negative_to_positive_ratio"
                    ],
                    "positive_mean_rank": candidate["positive_mean_rank"],
                }
            )
            kept += 1
            if kept == max_per_concept:
                break
    return selected


def rank_steering_screen(
    results: Iterable[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rank steering-screen results using a frozen, deliberately lenient gate."""

    ranked = []
    for result in results:
        summary = result["steering"]["summary"]
        feature = float(summary["feature_target_score"])
        baseline_delta = feature - float(summary["baseline_target_score"])
        random_delta = feature - float(summary["random_target_score"])
        failures = []
        if baseline_delta < protocol["min_target_delta_over_baseline"]:
            failures.append("target effect over baseline is too small")
        if random_delta < protocol["min_target_delta_over_random"]:
            failures.append("target effect over random control is too small")
        if float(summary["feature_success_rate"]) < protocol["min_target_success_rate"]:
            failures.append("target success rate is too low")
        if float(summary["nondegenerate_rate"]) < protocol["min_nondegenerate_rate"]:
            failures.append("non-degenerate rate is too low")
        if float(summary["rerun_agreement"]) < protocol["min_rerun_agreement"]:
            failures.append("rerun agreement is too low")
        ranked.append(
            {
                "concept_id": result["suite"].get("concept_id")
                or result.get("_concept_id")
                or result["suite"]["id"].removesuffix("_v1"),
                "feature_id": int(result["feature"]["feature_id"]),
                "selected_alpha": result["steering"]["selected_alpha"],
                "target_delta_over_baseline": baseline_delta,
                "target_delta_over_random": random_delta,
                "feature_success_rate": summary["feature_success_rate"],
                "nondegenerate_rate": summary["nondegenerate_rate"],
                "rerun_agreement": summary["rerun_agreement"],
                "screen_pass": not failures,
                "screen_failures": failures,
            }
        )
    ranked.sort(
        key=lambda row: (
            not row["screen_pass"],
            -min(
                row["target_delta_over_baseline"], row["target_delta_over_random"]
            ),
            -row["feature_success_rate"],
            -row["nondegenerate_rate"],
            -row["rerun_agreement"],
            row["concept_id"],
        )
    )
    return ranked

