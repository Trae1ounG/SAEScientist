#!/usr/bin/env python3
"""List benchmark tasks and validate evaluation inputs without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from sae_scientist.suites import load_suite, steering_sets

ROOT = Path(__file__).resolve().parents[1]


def inspect_dataset(benchmark_path: Path, root: Path = ROOT) -> list[dict]:
    benchmark = json.loads(benchmark_path.read_text())
    rows = []
    seen = set()
    for entry in benchmark["tasks"]:
        task = json.loads((root / entry["task"]).read_text())
        if task["task_id"] in seen:
            raise ValueError(f"duplicate task: {task['task_id']}")
        seen.add(task["task_id"])
        suite_path = root / entry["suite"]
        suite = load_suite(suite_path, entry.get("concept_id"))
        calibration, evaluation = steering_sets(suite, suite_path)
        reference = json.loads((root / entry["reference"]).read_text())
        source = json.loads((root / reference["feature_case"]).read_text())
        if source["feature_id"] != entry["expert_feature_id"]:
            raise ValueError(f"Expert ID mismatch: {task['task_id']}")
        layer = int(re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", task["sae"]["hook"]).group(1))
        if source["layer"] != layer:
            raise ValueError(f"SAE layer mismatch: {task['task_id']}")
        labels = [case["label"] for case in suite["activation_cases"]]
        counts = {label: labels.count(label) for label in ("positive", "hard_negative", "neutral")}
        if not all(counts.values()) or set(labels) != set(counts):
            raise ValueError(f"invalid activation groups: {task['task_id']}")
        if not calibration or not evaluation:
            raise ValueError(f"missing steering prompts: {task['task_id']}")
        if len({case["id"] for case in suite["activation_cases"]}) != len(labels):
            raise ValueError(f"duplicate activation IDs: {task['task_id']}")
        for name, prompts in (("calibration", calibration), ("evaluation", evaluation)):
            if len({row["id"] for row in prompts}) != len(prompts):
                raise ValueError(f"duplicate {name} prompt IDs: {task['task_id']}")
            if any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in prompts):
                raise ValueError(f"empty {name} prompt: {task['task_id']}")
        rows.append({
            "task_id": task["task_id"], "layer": layer,
            "feature_count": task["sae"]["feature_count"],
            "expert_feature_id": entry["expert_feature_id"],
            "activation_texts": counts,
            "calibration_prompts": len(calibration),
            "evaluation_prompts": len(evaluation),
            "shared_calibration_evaluation_prompts": len(
                {row["prompt"] for row in calibration} & {row["prompt"] for row in evaluation}
            ),
        })
    if not rows:
        raise ValueError("benchmark has no tasks")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark.json")
    args = parser.parse_args()
    rows = inspect_dataset(args.benchmark)
    print(json.dumps({"tasks": len(rows), "layers": sorted({r["layer"] for r in rows}), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
