#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _benchmark_tasks(benchmark_path: Path) -> dict[str, dict[str, Any]]:
    benchmark = _read_json(benchmark_path)
    tasks: dict[str, dict[str, Any]] = {}
    for row in benchmark.get("tasks", []):
        task = row.get("task")
        reference_path = ROOT / row.get("reference", "")
        if not isinstance(task, str) or not reference_path.exists():
            continue
        reference = _read_json(reference_path)
        if reference.get("status") != "stable_reference":
            continue
        tasks[task] = {
            "task": task,
            "concept_id": row.get("concept_id") or Path(task).stem,
            "expert_feature_id": row.get("expert_feature_id"),
            "reference": _repo_path(reference_path),
        }
    return tasks


def _skip(counter: dict[str, int], reason: str) -> None:
    counter[reason] += 1


def _score_summary_run_ids(path: Path) -> set[str]:
    summary = _read_json(path)
    rows = summary.get("runs")
    if not isinstance(rows, list) or int(summary.get("eligible_runs", -1)) != len(rows):
        raise ValueError("activation score summary has invalid run coverage")
    run_ids = {
        row.get("run_id")
        for row in rows
        if row.get("status") in {"scored", "skipped"}
    }
    if None in run_ids or len(run_ids) != len(rows):
        raise ValueError("activation score summary has incomplete or duplicate runs")
    return run_ids


def _trace_eligible_run_ids(path: Path) -> set[str]:
    rows = _read_json(path).get("runs")
    if not isinstance(rows, list):
        raise ValueError("trace audit has invalid runs")
    return {
        row["run_id"]
        for row in rows
        if row.get("eligible") and isinstance(row.get("run_id"), str)
    }


def _agent_row(source_path: Path, task_meta: dict[str, Any], agent: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if agent.get("status") != "complete":
        return None, "agent_status_not_complete"
    if any(key in agent for key in ("error", "failure", "exception")):
        return None, "agent_error_present"
    if "feature_id" not in agent:
        return None, "missing_submission_fields"

    gt_score = _finite_number(agent.get("gt_normalized", {}).get("mean_score"))
    auroc = _finite_number(agent.get("activation", {}).get("auroc"))
    positive_rank = _finite_number(agent.get("activation", {}).get("positive_mean_rank"))
    spearman = _finite_number(agent.get("activation", {}).get("expert_pattern_spearman"))
    steering_effect = _finite_number(agent.get("steering", {}).get("target_effect"))
    steering_pattern = _finite_number(
        agent.get("steering", {}).get("pattern_correlation_to_expert")
    )
    target_relevance = _finite_number(
        agent.get("steering", {}).get("pe_target_relevance")
    )
    task_preservation = _finite_number(
        agent.get("steering", {}).get("pe_task_preservation")
    )
    elapsed_seconds = _finite_number(agent.get("elapsed_seconds"))
    if None in (gt_score, auroc, positive_rank, steering_effect):
        return None, "missing_compact_scores"

    return {
        "task": task_meta["task"],
        "concept_id": task_meta["concept_id"],
        "model": agent.get("model") or agent.get("agent") or agent.get("run_id"),
        "harness": agent.get("harness"),
        "reasoning_effort": agent.get("reasoning_effort"),
        "run_id": agent.get("run_id"),
        "source": _repo_path(source_path),
        "selected_feature_id": int(agent["feature_id"]),
        "expert_feature_id": task_meta["expert_feature_id"],
        "exact_match": int(agent["feature_id"]) == int(task_meta["expert_feature_id"]),
        "gt_normalized_activation": gt_score,
        "positive_mean_rank": positive_rank,
        "activation_auroc": auroc,
        "expert_spearman": spearman,
        "steering_effect": steering_effect,
        "steering_pattern_correlation": steering_pattern,
        "pe_target_relevance": target_relevance,
        "pe_task_preservation": task_preservation,
        "causal_stable": bool(agent.get("steering", {}).get("causal_stable", False)),
        "usable_steering": bool(agent.get("steering", {}).get("usable_steering", False)),
        "elapsed_seconds": elapsed_seconds,
    }, None


def _configuration(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["harness"] or "unknown"),
        str(row["model"] or "unknown"),
        str(row["reasoning_effort"] or ""),
    )


def _configuration_name(key: tuple[Any, ...]) -> str:
    harness, model, reasoning_effort = key
    name = f"{harness}/{model}"
    return f"{name} ({reasoning_effort})" if reasoning_effort else name


def collect_leaderboard(
    result_paths: list[Path],
    benchmark_path: Path,
    selected_run_ids: set[str],
    eligible_run_ids: set[str],
) -> dict[str, Any]:
    tasks = _benchmark_tasks(benchmark_path)
    skipped: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for path in result_paths:
        payload = _read_json(path)
        if not isinstance(payload.get("agents"), list):
            _skip(skipped, "not_compact_agent_result")
            continue
        task = payload.get("task")
        if task not in tasks:
            _skip(skipped, "task_not_in_stable_benchmark")
            continue
        for agent in payload["agents"]:
            run_id = agent.get("run_id")
            if run_id not in selected_run_ids:
                _skip(skipped, "run_not_in_score_summary")
                continue
            if run_id not in eligible_run_ids:
                _skip(skipped, "run_not_trace_eligible")
                continue
            row, reason = _agent_row(path, tasks[task], agent)
            if reason:
                _skip(skipped, reason)
                continue
            run_id = row["run_id"]
            if not isinstance(run_id, str) or not run_id:
                _skip(skipped, "missing_run_id")
                continue
            if run_id in seen_run_ids:
                _skip(skipped, "duplicate_run_id")
                continue
            seen_run_ids.add(run_id)
            rows.append(row)

    rows.sort(
        key=lambda row: (
            _configuration(row),
            row["task"],
            row["run_id"],
        )
    )

    by_configuration: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_configuration[_configuration(row)].append(row)

    configuration_rows = []
    for configuration, configuration_runs in by_configuration.items():
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in configuration_runs:
            by_task[row["task"]].append(row)
        task_rows = list(by_task.values())
        latencies = [
            row["elapsed_seconds"]
            for row in configuration_runs
            if row["elapsed_seconds"] is not None
        ]
        configuration_rows.append(
            {
                "configuration": _configuration_name(configuration),
                "harness": configuration[0],
                "model": configuration[1],
                "reasoning_effort": configuration[2] or None,
                "benchmark_tasks": len(tasks),
                "completed_tasks": len(task_rows),
                "coverage_rate": len(task_rows) / len(tasks) if tasks else 0.0,
                "missing_tasks": sorted(set(tasks) - set(by_task)),
                "completed_runs": len(configuration_runs),
                "macro_gt_normalized_activation": mean(
                    [mean([row["gt_normalized_activation"] for row in task]) for task in task_rows]
                ),
                "exact_matches": sum(row["exact_match"] for row in configuration_runs),
                "exact_match_rate": mean(
                    [mean([row["exact_match"] for row in task]) for task in task_rows]
                ),
                "causal_steering_rate": mean(
                    [mean([row["causal_stable"] for row in task]) for task in task_rows]
                ),
                "usable_steering_rate": mean(
                    [mean([row["usable_steering"] for row in task]) for task in task_rows]
                ),
                "median_elapsed_seconds": median(latencies) if latencies else None,
                "latency_runs": len(latencies),
            }
        )
    configuration_rows.sort(
        key=lambda row: (
            -row["coverage_rate"],
            -row["macro_gt_normalized_activation"],
            -row["exact_match_rate"],
            row["configuration"],
        )
    )

    covered_tasks = sorted({row["task"] for row in rows})
    return {
        "schema": 1,
        "benchmark": _repo_path(benchmark_path),
        "task_coverage": {
            "stable_benchmark_tasks": len(tasks),
            "covered_tasks": len(covered_tasks),
            "missing_tasks": sorted(set(tasks) - set(covered_tasks)),
        },
        "run_counts": {
            "included": len(rows),
            "skipped": sum(skipped.values()),
            "skipped_by_reason": dict(sorted(skipped.items())),
        },
        "configurations": configuration_rows,
        "runs": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SAE-Bench Leaderboard",
        "",
        f"Benchmark: `{payload['benchmark']}`",
        "",
        "## Coverage",
        "",
        f"- Stable benchmark tasks: {payload['task_coverage']['stable_benchmark_tasks']}",
        f"- Covered tasks: {payload['task_coverage']['covered_tasks']}",
        f"- Included runs: {payload['run_counts']['included']}",
        f"- Skipped runs/files: {payload['run_counts']['skipped']}",
        "",
        "## Configuration Summary",
        "",
        "GT-normalized activation (expert = 1.0 per task) is the primary discovery ordering; steering is reported separately.",
        "",
        "| Configuration | Coverage | Macro GT activation | Exact | Causal | Usable | Median discovery time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["configurations"]:
        latency = row["median_elapsed_seconds"]
        latency_text = "" if latency is None else f"{latency / 60:.1f} min"
        lines.append(
            f"| {row['configuration']} | {row['completed_tasks']}/{row['benchmark_tasks']} | "
            f"{row['macro_gt_normalized_activation']:.3f} | {row['exact_match_rate']:.3f} | "
            f"{row['causal_steering_rate']:.3f} | {row['usable_steering_rate']:.3f} | "
            f"{latency_text} |"
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Configuration | Task | Selected ID | Exact | GT activation | Mean rank | AUROC | Activation corr. | Steering | Steering corr. |",
            "| --- | --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["runs"]:
        exact = "yes" if row["exact_match"] else "no"
        spearman = "" if row["expert_spearman"] is None else f"{row['expert_spearman']:.3f}"
        steering_pattern = row["steering_pattern_correlation"]
        steering_pattern_text = "" if steering_pattern is None else f"{steering_pattern:.3f}"
        lines.append(
            f"| {_configuration_name(_configuration(row))} | {row['concept_id']} | {row['selected_feature_id']} | {exact} | "
            f"{row['gt_normalized_activation']:.3f} | {row['positive_mean_rank']:.1f} | "
            f"{row['activation_auroc']:.3f} | {spearman} | {row['steering_effect']:.3f} | "
            f"{steering_pattern_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def _result_paths(results_dir: list[Path], result_file: list[Path]) -> list[Path]:
    paths = [path.resolve() for path in result_file]
    for directory in results_dir:
        paths.extend(
            sorted(path.resolve() for path in directory.glob("*.json") if path.is_file())
        )
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact SAE-Bench leaderboard from scored agent runs.")
    parser.add_argument("--results-dir", type=Path, action="append", default=[])
    parser.add_argument("--result-file", type=Path, action="append", default=[])
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark_v2.json")
    parser.add_argument("--activation-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()

    result_dirs = args.results_dir or [ROOT / "results" / "agent_eval"]
    payload = collect_leaderboard(
        _result_paths(result_dirs, args.result_file),
        args.benchmark,
        _score_summary_run_ids(args.activation_summary),
        _trace_eligible_run_ids(args.audit),
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    if not args.output_json and not args.output_markdown:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()

