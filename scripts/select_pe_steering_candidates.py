#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


FEATURE_RE = re.compile(r"feature_(\d+)\.npz$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def complete_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    expected_rows = summary.get("expected_rows")
    if (
        not isinstance(expected_rows, int)
        or expected_rows <= 0
        or summary.get("valid_rows") != expected_rows
        or summary.get("error_rows") != 0
    ):
        raise ValueError(f"incomplete or errored judge summary: {path}")
    return summary


def feature_id_from_task(task: dict[str, Any]) -> int:
    if "feature_id" in task:
        return int(task["feature_id"])
    match = FEATURE_RE.search(str(task["feature"]))
    if not match:
        raise ValueError(f"cannot infer feature id for task {task['id']}")
    return int(match.group(1))


def concept_id_from_raw(raw: dict[str, Any]) -> str:
    suite = raw.get("suite", {})
    return str(suite.get("concept_id") or suite["id"].removesuffix("_v1"))


def validate_raw_result(
    task: dict[str, Any], raw: dict[str, Any], result_path: Path, summary: dict[str, Any]
) -> None:
    feature_id = int(raw["feature"]["feature_id"])
    expected_feature_id = feature_id_from_task(task)
    if feature_id != expected_feature_id:
        raise ValueError(
            f"feature id mismatch for {task['id']}: "
            f"manifest={expected_feature_id} raw={feature_id}"
        )
    concept_id = concept_id_from_raw(raw)
    if concept_id != task["concept_id"]:
        raise ValueError(
            f"concept id mismatch for {task['id']}: "
            f"manifest={task['concept_id']} raw={concept_id}"
        )
    summary_result = summary.get("result")
    if summary_result and Path(summary_result).name != result_path.name:
        raise ValueError(
            f"summary result mismatch for {task['id']}: "
            f"summary={summary_result} raw={result_path}"
        )


def screen_row(
    task: dict[str, Any],
    summary: dict[str, Any],
    raw: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    evaluation = summary["admission_evaluation"]
    feature = float(evaluation["feature_target_score"])
    baseline_delta = feature - float(evaluation["baseline_target_score"])
    random_delta = feature - float(evaluation["random_target_score"])
    failures = []
    if baseline_delta < protocol["min_target_delta_over_baseline"]:
        failures.append("target effect over baseline is too small")
    if random_delta < protocol["min_target_delta_over_random"]:
        failures.append("target effect over random control is too small")
    if float(evaluation["feature_success_rate"]) < protocol["min_target_success_rate"]:
        failures.append("target success rate is too low")
    if float(evaluation["nondegenerate_rate"]) < protocol["min_nondegenerate_rate"]:
        failures.append("non-degenerate rate is too low")
    if float(evaluation["rerun_agreement"]) < protocol["min_rerun_agreement"]:
        failures.append("rerun agreement is too low")
    return {
        "task_id": task["id"],
        "concept_id": task["concept_id"],
        "feature_id": int(raw["feature"]["feature_id"]),
        "selected_alpha": raw["steering"]["selected_alpha"],
        "target_delta_over_baseline": baseline_delta,
        "target_delta_over_random": random_delta,
        "feature_target_score": feature,
        "feature_success_rate": evaluation["feature_success_rate"],
        "nondegenerate_rate": evaluation["nondegenerate_rate"],
        "rerun_agreement": evaluation["rerun_agreement"],
        "screen_pass": not failures,
        "screen_failures": failures,
    }


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda row: (
            not row["screen_pass"],
            -min(row["target_delta_over_baseline"], row["target_delta_over_random"]),
            -float(row["feature_success_rate"]),
            -float(row["nondegenerate_rate"]),
            -float(row["rerun_agreement"]),
            row["concept_id"],
            row["feature_id"],
        )
    )
    return rows


def protocol_for_tier(document: dict[str, Any], tier: str) -> dict[str, Any]:
    if tier == "screen":
        return document["steering_screen"]
    causal = document["formal_steering_gate"]["causal_stable"]
    return {
        "min_target_delta_over_baseline": causal["min_target_delta_over_baseline"],
        "min_target_delta_over_random": causal["min_target_delta_over_random"],
        "min_target_success_rate": causal["min_target_success_rate"],
        "min_nondegenerate_rate": causal["min_nondegenerate_rate"],
        "min_rerun_agreement": causal["min_rerun_agreement"],
    }


def select_rows(
    manifest: dict[str, Any],
    summary_dir: Path,
    result_dir: Path,
    protocol: dict[str, Any],
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    if manifest.get("stage") != "steering_screen":
        raise ValueError("manifest stage must be steering_screen")
    rows = []
    for task in manifest.get("tasks", []):
        summary_path = summary_dir / f"{task['id']}_summary.json"
        result_path = result_dir / f"{task['id']}.json"
        if allow_missing and (not summary_path.exists() or not result_path.exists()):
            continue
        summary = complete_summary(summary_path)
        raw = load_json(result_path)
        validate_raw_result(task, raw, result_path, summary)
        rows.append(screen_row(task, summary, raw, protocol))
    return rank_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select PE-judged steering-screen candidates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--admission-tier",
        choices=("screen", "causal"),
        default="screen",
        help="Use the strict screen gate or retain candidates matching the formal causal tier.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Select only completed task pairs for incremental screening.",
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    protocol = protocol_for_tier(load_json(args.protocol), args.admission_tier)
    rows = select_rows(
        manifest,
        args.summary_dir,
        args.result_dir,
        protocol,
        allow_missing=args.allow_missing,
    )
    passed = [row for row in rows if row["screen_pass"]]
    payload = {"schema": 1, "screened": len(rows), "passed": len(passed), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"screened": len(rows), "passed": len(passed)}))


if __name__ == "__main__":
    main()

