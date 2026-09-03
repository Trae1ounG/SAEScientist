#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_RUN_METRICS = (
    "gt_normalized_activation",
    "steering_effect",
    "pe_target_relevance",
    "pe_task_preservation",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(
    behavior: dict[str, Any],
    leaderboard: dict[str, Any],
    *,
    expected_tasks: int,
    expected_configurations: int,
) -> dict[str, int]:
    expected_runs = expected_tasks * expected_configurations
    coverage = behavior.get("coverage", {})
    selection = behavior.get("selection", {})
    if behavior.get("status") != "complete":
        raise ValueError("behavior analysis is not complete")
    if coverage.get("expected_cells") != expected_runs:
        raise ValueError("behavior analysis has the wrong expected-cell count")
    if coverage.get("included_cells") != expected_runs or coverage.get("missing_cells"):
        raise ValueError("behavior analysis does not cover every expected cell")
    if selection.get("summary_runs") != expected_runs:
        raise ValueError("behavior analysis has incomplete score-summary coverage")
    if selection.get("trace_eligible_summary_runs") != expected_runs:
        raise ValueError("behavior analysis has incomplete trace-eligible coverage")
    if selection.get("superseded_attempts"):
        raise ValueError("behavior analysis still contains superseded attempts")

    configurations = leaderboard.get("configurations", [])
    runs = leaderboard.get("runs", [])
    task_coverage = leaderboard.get("task_coverage", {})
    run_counts = leaderboard.get("run_counts", {})
    if task_coverage.get("stable_benchmark_tasks") != expected_tasks:
        raise ValueError("leaderboard has the wrong stable-task count")
    if task_coverage.get("covered_tasks") != expected_tasks or task_coverage.get("missing_tasks"):
        raise ValueError("leaderboard does not cover every stable task")
    if run_counts.get("included") != expected_runs or run_counts.get("skipped") != 0:
        raise ValueError("leaderboard does not contain exactly the expected runs")
    if len(configurations) != expected_configurations:
        raise ValueError("leaderboard has the wrong configuration count")
    for row in configurations:
        if (
            row.get("completed_tasks") != expected_tasks
            or row.get("completed_runs") != expected_tasks
            or row.get("latency_runs") != expected_tasks
            or row.get("coverage_rate") != 1.0
            or row.get("missing_tasks")
        ):
            raise ValueError(f"incomplete configuration: {row.get('configuration')}")
    if len(runs) != expected_runs:
        raise ValueError("leaderboard run list has the wrong length")
    for row in runs:
        missing = [metric for metric in REQUIRED_RUN_METRICS if row.get(metric) is None]
        if missing:
            raise ValueError(f"run {row.get('run_id')} is missing {', '.join(missing)}")

    behavior_configs = {
        (row.get("harness"), row.get("model"), row.get("reasoning_effort"))
        for row in behavior.get("configurations", [])
    }
    leaderboard_configs = {
        (row.get("harness"), row.get("model"), row.get("reasoning_effort"))
        for row in configurations
    }
    if behavior_configs != leaderboard_configs:
        raise ValueError("behavior and leaderboard configurations differ")
    return {
        "tasks": expected_tasks,
        "configurations": expected_configurations,
        "runs": expected_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a complete formal SAE-Bench release.")
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--blog", type=Path)
    parser.add_argument("--plot", type=Path, action="append", default=[])
    parser.add_argument("--expected-tasks", type=int, default=20)
    parser.add_argument("--expected-configurations", type=int, default=8)
    args = parser.parse_args()

    result = validate(
        read_json(args.behavior),
        read_json(args.leaderboard),
        expected_tasks=args.expected_tasks,
        expected_configurations=args.expected_configurations,
    )
    if args.blog:
        blog = args.blog.read_text(encoding="utf-8")
        if "{{FORMAL_" in blog:
            raise ValueError("blog still contains formal-result placeholders")
    for path in args.plot:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty plot: {path}")
    print(json.dumps({"status": "passed", **result}))


if __name__ == "__main__":
    main()

