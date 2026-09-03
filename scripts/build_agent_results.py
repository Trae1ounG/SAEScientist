#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL_PE_STAGE = "formal_steering"
FORMAL_PE_REPEATS = 2


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clip_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def benchmark_scores(
    activation: dict[str, Any],
    steering: dict[str, Any],
    expert_steering: dict[str, Any],
) -> dict[str, float]:
    """Return three expert-normalized blocks and their equal-weight mean."""
    fallback = float(activation["mean_score"])
    rank_score = clip_unit(activation.get("positive_rank_recovery", fallback))
    activation_parts = [
        activation[key]
        for key in (
            "auroc_recovery",
            "activation_contrast_recovery",
            "expert_pattern_spearman",
        )
        if key in activation
    ]
    activation_score = (
        mean([clip_unit(value) for value in activation_parts])
        if activation_parts
        else clip_unit(fallback)
    )
    expert_effect = float(expert_steering["target_effect"])
    if expert_effect <= 0:
        raise ValueError("expert steering target effect must be positive")
    effect_recovery = clip_unit(
        max(float(steering["target_effect"]), 0.0) / expert_effect
    )
    pattern_recovery = clip_unit(
        steering.get("pattern_correlation_to_expert") or 0.0
    )
    steering_score = mean([effect_recovery, pattern_recovery])
    overall_score = mean([rank_score, activation_score, steering_score])
    return {
        "rank_score": rank_score,
        "activation_score": activation_score,
        "steering_effect_recovery": effect_recovery,
        "steering_pattern_recovery": pattern_recovery,
        "steering_score": steering_score,
        "overall_score": overall_score,
    }


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            output[position] = rank
        index = end
    return output


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks, right_ranks = ranks(left), ranks(right)
    left_mean, right_mean = mean(left_ranks), mean(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left_ranks))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right_ranks))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def case_effects(path: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "ratings" not in row:
            continue
        ratings = row["ratings"]
        feature = float(ratings["feature"]["target_relevance"]) / 4
        control = max(
            float(ratings["baseline"]["target_relevance"]),
            float(ratings["random"]["target_relevance"]),
        ) / 4
        grouped.setdefault(row["case_id"], []).append(feature - control)
    return {case_id: mean(values) for case_id, values in grouped.items()}


def pattern_correlation(candidate: Path, expert: Path) -> float | None:
    candidate_effects = case_effects(candidate)
    expert_effects = case_effects(expert)
    case_ids = sorted(set(candidate_effects) & set(expert_effects))
    return spearman(
        [candidate_effects[case_id] for case_id in case_ids],
        [expert_effects[case_id] for case_id in case_ids],
    )


def index_files(directories: list[Path], suffix: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.rglob(f"*{suffix}")):
            if path.name in {"summary.json", "batch_summary.json"}:
                continue
            key = path.name[: -len(suffix)]
            if key in output:
                raise ValueError(f"duplicate result for {key}: {output[key]} and {path}")
            output[key] = path
    return output


def retry_attempt(run_id: str) -> int:
    match = re.search(r"-retry-(\d+)$", run_id)
    return int(match.group(1)) if match else 0


def score_reasoning_effort(score: dict[str, Any], runs_root: Path) -> str | None:
    if score.get("reasoning_effort"):
        return str(score["reasoning_effort"])
    manifest_path = runs_root / score["run_id"] / "run.json"
    if not manifest_path.exists():
        return None
    value = read_json(manifest_path).get("reasoning_effort")
    return str(value) if value else None


def activation_score_paths(summary_path: Path, activation_dir: Path) -> list[Path]:
    summary = read_json(summary_path)
    rows = summary.get("runs")
    if not isinstance(rows, list) or int(summary.get("eligible_runs", -1)) != len(rows):
        raise ValueError("activation score summary has invalid run coverage")
    output: list[Path] = []
    seen: set[str] = set()
    expected_dir = activation_dir.resolve()
    for row in rows:
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen:
            raise ValueError("activation score summary has invalid run IDs")
        if row.get("status") not in {"scored", "skipped"}:
            raise ValueError(f"activation score summary is incomplete for {run_id}")
        path_value = row.get("output")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"activation score summary has no output for {run_id}")
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if path.parent != expected_dir or not path.is_file():
            raise ValueError(f"activation score output is invalid for {run_id}")
        if read_json(path).get("run_id") != run_id:
            raise ValueError(f"activation score output has wrong run ID for {run_id}")
        seen.add(run_id)
        output.append(path)
    return output


def suite_matches(actual: Any, expected: str) -> bool:
    if isinstance(actual, dict):
        actual = actual.get("path")
    if not isinstance(actual, str):
        return False
    actual_path = Path(actual).as_posix()
    expected_path = Path(expected).as_posix()
    return actual_path == expected_path or actual_path.endswith(f"/{expected_path}")


def judge_rows(path: Path, repeats: int) -> set[tuple[str, int]]:
    coverage: set[tuple[str, int]] = set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if "ratings" not in row:
            raise ValueError(f"incomplete PE rows in {path.name}")
        case_id = row.get("case_id")
        repeat = row.get("repeat")
        if (
            not isinstance(case_id, str)
            or not case_id
            or isinstance(repeat, bool)
            or not isinstance(repeat, int)
        ):
            raise ValueError(f"invalid PE coverage in {path.name}")
        key = (case_id, repeat)
        if key in coverage or not 0 <= key[1] < repeats:
            raise ValueError(f"invalid PE coverage in {path.name}")
        coverage.add(key)
    return coverage


def validate_pe_pair(
    *,
    candidate_id: str,
    expert_id: str,
    expected_suite: str,
    candidate_summary_path: Path,
    expert_summary_path: Path,
    candidate_rows_path: Path,
    expert_rows_path: Path,
    candidate_result_path: Path,
    expert_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_summary = read_json(candidate_summary_path)
    expert_summary = read_json(expert_summary_path)
    candidate_result = read_json(candidate_result_path)
    expert_result = read_json(expert_result_path)

    for expected_id, summary, summary_path, result, result_path in (
        (
            candidate_id,
            candidate_summary,
            candidate_summary_path,
            candidate_result,
            candidate_result_path,
        ),
        (
            expert_id,
            expert_summary,
            expert_summary_path,
            expert_result,
            expert_result_path,
        ),
    ):
        if result_path.stem != expected_id:
            raise ValueError(f"wrong PE result ID for {expected_id}")
        named_result = summary.get("result")
        if not isinstance(named_result, str) or Path(named_result).stem != expected_id:
            raise ValueError(f"wrong PE summary result ID for {expected_id}")
        if not suite_matches(result.get("suite"), expected_suite):
            raise ValueError(f"wrong PE suite for {expected_id}")
        expected_rows = int(summary.get("expected_rows", -1))
        if (
            expected_rows < 1
            or int(summary.get("valid_rows", -1)) != expected_rows
            or int(summary.get("error_rows", -1)) != 0
        ):
            raise ValueError(f"incomplete PE summary for {expected_id}")
        if summary_path.name != f"{expected_id}_summary.json":
            raise ValueError(f"wrong PE summary ID for {expected_id}")

    if not candidate_summary.get("judge_model") or (
        candidate_summary.get("judge_model") != expert_summary.get("judge_model")
    ):
        raise ValueError("candidate and expert use different PE judges")
    if candidate_summary.get("judge_provider") != expert_summary.get("judge_provider"):
        raise ValueError("candidate and expert use different PE judge providers")
    if (
        candidate_summary.get("stage") != FORMAL_PE_STAGE
        or expert_summary.get("stage") != FORMAL_PE_STAGE
    ):
        raise ValueError(f"PE stage must be {FORMAL_PE_STAGE}")
    candidate_repeats = int(candidate_summary.get("repeats", 0))
    expert_repeats = int(expert_summary.get("repeats", 0))
    if (
        candidate_repeats != FORMAL_PE_REPEATS
        or expert_repeats != FORMAL_PE_REPEATS
    ):
        raise ValueError(f"formal PE repeats must be {FORMAL_PE_REPEATS}")

    candidate_coverage = judge_rows(candidate_rows_path, candidate_repeats)
    expert_coverage = judge_rows(expert_rows_path, expert_repeats)
    if len(candidate_coverage) != int(candidate_summary["expected_rows"]):
        raise ValueError("candidate PE rows do not match its summary")
    if len(expert_coverage) != int(expert_summary["expected_rows"]):
        raise ValueError("expert PE rows do not match its summary")
    if candidate_coverage != expert_coverage:
        raise ValueError("candidate and expert PE coverage differs")
    return candidate_summary, candidate_result


def validate_expert_pe(
    *,
    expert_id: str,
    expected_suite: str,
    summary_path: Path,
    rows_path: Path,
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = read_json(summary_path)
    result = read_json(result_path)
    if result_path.stem != expert_id:
        raise ValueError(f"wrong PE result ID for {expert_id}")
    named_result = summary.get("result")
    if not isinstance(named_result, str) or Path(named_result).stem != expert_id:
        raise ValueError(f"wrong PE summary result ID for {expert_id}")
    if not suite_matches(result.get("suite"), expected_suite):
        raise ValueError(f"wrong PE suite for {expert_id}")
    expected_rows = int(summary.get("expected_rows", -1))
    if (
        expected_rows < 1
        or int(summary.get("valid_rows", -1)) != expected_rows
        or int(summary.get("error_rows", -1)) != 0
    ):
        raise ValueError(f"incomplete PE summary for {expert_id}")
    if summary.get("stage") != FORMAL_PE_STAGE:
        raise ValueError(f"PE stage must be {FORMAL_PE_STAGE}")
    repeats = int(summary.get("repeats", 0))
    if repeats != FORMAL_PE_REPEATS:
        raise ValueError(f"formal PE repeats must be {FORMAL_PE_REPEATS}")
    if len(judge_rows(rows_path, repeats)) != expected_rows:
        raise ValueError("expert PE rows do not match its summary")
    return summary, result


def steering_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    evaluation = summary["admission_evaluation"]
    conditions = summary["conditions"]
    target_effect = float(evaluation["feature_target_score"]) - max(
        float(evaluation["baseline_target_score"]),
        float(evaluation["random_target_score"]),
    )
    return {
        "pe_target_relevance": conditions["feature"]["target_relevance"],
        "pe_target_success_rate": evaluation["feature_success_rate"],
        "pe_task_preservation": conditions["feature"]["task_preservation"],
        "pe_usable_target_rate": evaluation["usable_target_rate"],
        "pe_degenerate_rate": conditions["feature"]["degenerate_rate"],
        "target_effect": target_effect,
        "causal_stable": summary["quality"]["causal_stable"],
        "usable_steering": summary["quality"]["usable_steering"],
    }


def frozen_expert_steering(reference: dict[str, Any]) -> dict[str, Any]:
    conditions = reference["steering"]
    feature = conditions["feature"]
    baseline = conditions["baseline"]
    random = conditions["random"]
    target_effect = float(feature["target_relevance"]) / 4 - max(
        float(baseline["target_relevance"]),
        float(random["target_relevance"]),
    ) / 4
    return {
        "selected_alpha": reference["protocol"]["alpha"],
        "pe_target_relevance": feature["target_relevance"],
        "pe_target_success_rate": feature["target_success_rate"],
        "pe_task_preservation": feature["task_preservation"],
        "pe_usable_target_rate": feature.get("usable_target_rate"),
        "pe_degenerate_rate": feature["degenerate_rate"],
        "target_effect": target_effect,
        "pattern_correlation_to_expert": 1.0,
        "causal_stable": True,
        "usable_steering": reference.get("admission_tier", "usable") != "causal",
        "source": "frozen expert reference reused because the submitted feature is identical",
    }


def compact_activation(score: dict[str, Any]) -> dict[str, Any]:
    activation = score["activation_rank"]
    return {
        "positive_mean_rank": activation["positive"]["mean_rank"],
        "positive_mean_percentile": activation["positive"]["mean_percentile"],
        "hard_negative_mean_rank": activation["hard_negative"]["mean_rank"],
        "neutral_mean_rank": activation["neutral"]["mean_rank"],
        "auroc": activation["activation_auroc"],
        "expert_pattern_spearman": score["expert_activation_spearman"],
    }


def build_results(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = read_json(args.benchmark)
    task_rows = {row["task"]: row for row in benchmark["tasks"]}
    eligible = {
        row["run_id"] for row in read_json(args.audit)["runs"] if row["eligible"]
    }
    summaries = index_files(args.candidate_judge_dir, "_summary.json")
    expert_summaries = index_files(args.expert_judge_dir, "_summary.json")
    candidate_jsonl = index_files(args.candidate_judge_dir, ".jsonl")
    expert_jsonl = index_files(args.expert_judge_dir, ".jsonl")
    raw_results = index_files(args.candidate_result_dir, ".json")

    agents_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in task_rows}
    expert_metrics_by_task: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    judge_models: set[str] = set()
    judge_providers: set[str] = set()
    selected_scores: dict[tuple[str, str, str, str | None], tuple[Path, dict[str, Any]]] = {}
    for path in activation_score_paths(args.activation_summary, args.activation_dir):
        score = read_json(path)
        run_id = score["run_id"]
        task_name = score["task"]
        if run_id not in eligible or task_name not in task_rows:
            skipped[run_id] = "ineligible trace or task"
            continue
        score["reasoning_effort"] = score_reasoning_effort(score, args.runs_root)
        key = (
            task_name,
            score["harness"],
            score["model"],
            score.get("reasoning_effort"),
        )
        previous = selected_scores.get(key)
        if previous is None or retry_attempt(run_id) < retry_attempt(previous[1]["run_id"]):
            if previous is not None:
                skipped[previous[1]["run_id"]] = "superseded by earlier eligible attempt"
            selected_scores[key] = (path, score)
        else:
            skipped[run_id] = "superseded by earlier eligible attempt"

    for path, score in selected_scores.values():
        run_id = score["run_id"]
        task_name = score["task"]
        exact = int(score["feature_id"]) == int(task_rows[task_name]["expert_feature_id"])
        reference = read_json(ROOT / task_rows[task_name]["reference"])
        task = read_json(ROOT / task_name)
        if exact:
            expert_id = f"{task['task_id']}__expert_anchor"
            if getattr(args, "live_expert_judge", False):
                required = {
                    "expert summary": expert_summaries.get(expert_id),
                    "expert judge rows": expert_jsonl.get(expert_id),
                    "expert raw result": raw_results.get(expert_id),
                }
                missing = [name for name, value in required.items() if value is None]
                if missing:
                    skipped[run_id] = f"missing {', '.join(missing)}"
                    continue
                try:
                    expert_summary, expert_raw = validate_expert_pe(
                        expert_id=expert_id,
                        expected_suite=reference["suite"],
                        summary_path=required["expert summary"],
                        rows_path=required["expert judge rows"],
                        result_path=required["expert raw result"],
                    )
                except ValueError as error:
                    skipped[run_id] = str(error)
                    continue
                steering = steering_from_summary(expert_summary)
                steering["selected_alpha"] = expert_raw["steering"]["selected_alpha"]
                steering["pattern_correlation_to_expert"] = 1.0
                steering["source"] = "expert feature freshly scored by the selected judge"
                judge_models.add(str(expert_summary["judge_model"]))
                judge_providers.add(str(expert_summary.get("judge_provider", "unknown")))
            else:
                steering = frozen_expert_steering(reference)
            expert_steering = steering
        else:
            candidate_id = f"{task['task_id']}__candidate_{score['feature_id']}"
            expert_id = f"{task['task_id']}__expert_anchor"
            required = {
                "candidate summary": summaries.get(candidate_id),
                "expert summary": expert_summaries.get(expert_id),
                "candidate judge rows": candidate_jsonl.get(candidate_id),
                "expert judge rows": expert_jsonl.get(expert_id),
                "candidate raw result": raw_results.get(candidate_id),
                "expert raw result": raw_results.get(expert_id),
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                skipped[run_id] = f"missing {', '.join(missing)}"
                continue
            try:
                candidate_summary, raw = validate_pe_pair(
                    candidate_id=candidate_id,
                    expert_id=expert_id,
                    expected_suite=reference["suite"],
                    candidate_summary_path=required["candidate summary"],
                    expert_summary_path=required["expert summary"],
                    candidate_rows_path=required["candidate judge rows"],
                    expert_rows_path=required["expert judge rows"],
                    candidate_result_path=required["candidate raw result"],
                    expert_result_path=required["expert raw result"],
                )
            except ValueError as error:
                skipped[run_id] = str(error)
                continue
            steering = steering_from_summary(candidate_summary)
            expert_steering = steering_from_summary(read_json(required["expert summary"]))
            judge_models.add(str(candidate_summary["judge_model"]))
            judge_providers.add(str(candidate_summary.get("judge_provider", "unknown")))
            steering["selected_alpha"] = raw["steering"]["selected_alpha"]
            steering["pattern_correlation_to_expert"] = pattern_correlation(
                required["candidate judge rows"], required["expert judge rows"]
            )

        scores = benchmark_scores(score["gt_normalized"], steering, expert_steering)
        steering["expert_target_effect"] = expert_steering["target_effect"]
        expert_metrics_by_task[task_name] = {
            "target_effect": expert_steering["target_effect"],
            "target_relevance": expert_steering["pe_target_relevance"],
            "task_preservation": expert_steering["pe_task_preservation"],
            "causal_stable": expert_steering["causal_stable"],
            "usable_steering": expert_steering["usable_steering"],
        }

        agents_by_task[task_name].append(
            {
                "status": "complete",
                "run_id": run_id,
                "harness": score["harness"],
                "model": score["model"],
                "reasoning_effort": score.get("reasoning_effort"),
                "source_commit": score.get("source_commit"),
                "elapsed_seconds": score.get("elapsed_seconds"),
                "feature_id": score["feature_id"],
                "exact_match": exact,
                "gt_normalized": score["gt_normalized"],
                "scores": scores,
                "activation": compact_activation(score),
                "steering": steering,
            }
        )

    if len(judge_models) > 1:
        raise ValueError("formal PE results use different judges across tasks")
    if len(judge_providers) > 1:
        raise ValueError("formal PE results use different judge providers across tasks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for task_name, agents in agents_by_task.items():
        task = read_json(ROOT / task_name)
        row = task_rows[task_name]
        output = args.output_dir / f"{task['task_id']}_v2.json"
        payload = {
            "schema": 2,
            "task": task_name,
            "suite": row["suite"],
            "expert": {
                "feature_id": row["expert_feature_id"],
                "reference": row["reference"],
                "score_baseline": {
                    "rank_score": 1.0,
                    "activation_score": 1.0,
                    "steering_score": 1.0,
                    "overall_score": 1.0,
                },
                "steering": expert_metrics_by_task.get(task_name),
            },
            "agents": sorted(agents, key=lambda item: item["model"]),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        files.append(str(output))
    return {
        "tasks": len(files),
        "complete_runs": sum(len(rows) for rows in agents_by_task.values()),
        "judge_model": next(iter(judge_models), None),
        "judge_provider": next(iter(judge_providers), None),
        "skipped_runs": len(skipped),
        "skipped": skipped,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge activation and steering scores into compact agent results.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--activation-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--candidate-result-dir", type=Path, action="append", required=True)
    parser.add_argument("--candidate-judge-dir", type=Path, action="append", required=True)
    parser.add_argument("--expert-judge-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--live-expert-judge",
        action="store_true",
        help="Score exact-match runs from the selected expert judge artifacts instead of frozen reference values.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_results(args)))


if __name__ == "__main__":
    main()
