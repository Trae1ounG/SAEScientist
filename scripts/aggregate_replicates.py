#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


METRICS = (
    "mean_overall_score",
    "total_overall_score",
    "mean_rank_score",
    "mean_activation_score",
    "mean_steering_score",
    "macro_gt_normalized_activation",
    "exact_match_rate",
    "causal_steering_rate",
    "usable_steering_rate",
    "median_elapsed_seconds",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_replicate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("replicates must use LABEL=PATH")
    return label, Path(path)


def configuration_key(row: dict[str, Any]) -> tuple[str, str, str | None]:
    return row["harness"], row["model"], row.get("reasoning_effort")


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": fmean(values), "std": pstdev(values)}


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    left_ranks, right_ranks = rankdata(left), rankdata(right)
    left_mean, right_mean = fmean(left_ranks), fmean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_norm = sum((value - left_mean) ** 2 for value in left_ranks) ** 0.5
    right_norm = sum((value - right_mean) ** 2 for value in right_ranks) ** 0.5
    return numerator / (left_norm * right_norm)


def aggregate(
    replicates: list[tuple[str, dict[str, Any]]],
    expected_replicates: int | None = None,
) -> dict[str, Any]:
    labels = list(dict.fromkeys(label for label, _ in replicates))
    if len(labels) < 2:
        raise ValueError("at least two replicates are required")
    expected_replicates = expected_replicates or len(labels)
    if not 2 <= expected_replicates <= len(labels):
        raise ValueError("expected replicates must be between 2 and the number of inputs")

    benchmark_tasks = {
        int(payload["task_coverage"]["stable_benchmark_tasks"])
        for _, payload in replicates
    }
    if len(benchmark_tasks) != 1:
        raise ValueError("replicates use different benchmark task counts")
    task_count = benchmark_tasks.pop()

    by_config: dict[tuple[str, str, str | None], dict[str, dict[str, Any]]] = defaultdict(dict)
    runs_by_config: dict[
        tuple[str, str, str | None], dict[str, dict[str, dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for label, payload in replicates:
        for row in payload["configurations"]:
            key = configuration_key(row)
            if label in by_config[key]:
                raise ValueError(f"duplicate configuration in {label}: {key}")
            if int(row["completed_tasks"]) != task_count:
                raise ValueError(f"incomplete configuration in {label}: {key}")
            by_config[key][label] = row
        for row in payload["runs"]:
            key = (row["harness"], row["model"], row.get("reasoning_effort"))
            task = row["task"]
            if task in runs_by_config[key][label]:
                raise ValueError(f"duplicate task in {label}: {key} {task}")
            runs_by_config[key][label][task] = row

    rows = []
    for key, by_label in by_config.items():
        config_labels = [label for label in labels if label in by_label]
        if len(config_labels) != expected_replicates:
            raise ValueError(
                f"configuration has {len(config_labels)} replicates, expected "
                f"{expected_replicates}: {key}"
            )
        run_sets = runs_by_config[key]
        if set(run_sets) != set(config_labels):
            raise ValueError(f"configuration has missing run data: {key}")
        task_sets = [set(run_sets[label]) for label in config_labels]
        if any(tasks != task_sets[0] for tasks in task_sets[1:]) or len(task_sets[0]) != task_count:
            raise ValueError(f"configuration has inconsistent task coverage: {key}")

        metrics = {
            metric: mean_std([float(by_label[label][metric]) for label in config_labels])
            for metric in METRICS
        }
        pe_target = []
        pe_preservation = []
        pairwise_matches = []
        all_same = []
        for label in config_labels:
            runs = run_sets[label].values()
            pe_target.append(fmean(float(row["pe_target_relevance"]) for row in runs))
            pe_preservation.append(fmean(float(row["pe_task_preservation"]) for row in runs))
        for task in sorted(task_sets[0]):
            feature_ids = [
                int(run_sets[label][task]["selected_feature_id"])
                for label in config_labels
            ]
            matches = [
                left == right
                for index, left in enumerate(feature_ids)
                for right in feature_ids[index + 1 :]
            ]
            pairwise_matches.extend(matches)
            all_same.append(len(set(feature_ids)) == 1)

        metrics["macro_pe_target_relevance"] = mean_std(pe_target)
        metrics["macro_pe_task_preservation"] = mean_std(pe_preservation)
        rows.append(
            {
                "configuration": by_label[config_labels[0]]["configuration"],
                "harness": key[0],
                "model": key[1],
                "reasoning_effort": key[2],
                "replicates": len(config_labels),
                "replicate_labels": config_labels,
                "benchmark_tasks": task_count,
                "metrics": metrics,
                "feature_id_pairwise_agreement": fmean(pairwise_matches),
                "feature_id_all_replicates_same_rate": fmean(all_same),
            }
        )

    rows.sort(
        key=lambda row: row["metrics"]["mean_overall_score"]["mean"],
        reverse=True,
    )
    all_runs = [run for labels in runs_by_config.values() for tasks in labels.values() for run in tasks.values()]
    analysis_fields = {
        "exact_match",
        "causal_stable",
        "usable_steering",
        "gt_normalized_activation",
        "steering_effect",
        "expert_feature_id",
    }
    analysis = None
    if all(analysis_fields <= run.keys() for run in all_runs):
        alternative_runs = [run for run in all_runs if not run["exact_match"]]
        task_feature_pairs = {
            (run["task"], int(run["selected_feature_id"])) for run in all_runs
        } | {
            (run["task"], int(run["expert_feature_id"])) for run in all_runs
        }
        analysis = {
            "exact_runs": sum(bool(run["exact_match"]) for run in all_runs),
            "causal_runs": sum(bool(run["causal_stable"]) for run in all_runs),
            "usable_runs": sum(bool(run["usable_steering"]) for run in all_runs),
            "alternative_runs": len(alternative_runs),
            "alternative_causal_runs": sum(
                bool(run["causal_stable"]) for run in alternative_runs
            ),
            "alternative_usable_runs": sum(
                bool(run["usable_steering"]) for run in alternative_runs
            ),
            "activation_steering_spearman": correlation(
                [float(run["gt_normalized_activation"]) for run in all_runs],
                [float(run["steering_effect"]) for run in all_runs],
            ),
            "alternative_activation_steering_spearman": correlation(
                [float(run["gt_normalized_activation"]) for run in alternative_runs],
                [float(run["steering_effect"]) for run in alternative_runs],
            ),
            "exact_mean_steering_effect": fmean(
                float(run["steering_effect"]) for run in all_runs if run["exact_match"]
            ),
            "alternative_mean_steering_effect": fmean(
                float(run["steering_effect"]) for run in alternative_runs
            ),
            "evaluated_task_feature_pairs": len(task_feature_pairs),
        }
    expert_baselines = [
        payload["expert_baseline"]
        for _, payload in replicates
        if "expert_baseline" in payload
    ]
    return {
        "schema": 1,
        "replicate_sources": labels,
        "replicates": expected_replicates,
        "benchmark_tasks": task_count,
        "discovery_runs": len(all_runs),
        "expert_baseline": expert_baselines[0] if expert_baselines else None,
        "analysis": analysis,
        "configurations": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent SAEScientist-Bench replicates.")
    parser.add_argument("--replicate", action="append", type=parse_replicate, required=True)
    parser.add_argument(
        "--expected-replicates",
        type=int,
        help="Required replicate count per configuration; defaults to all inputs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(
        [(label, load_json(path)) for label, path in args.replicate],
        args.expected_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"replicates": payload["replicates"], "configurations": len(payload["configurations"])}))


if __name__ == "__main__":
    main()
