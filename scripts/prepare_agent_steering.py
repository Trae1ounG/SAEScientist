#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PROVENANCE_FIELDS = (
    "publisher",
    "official_source",
    "repo",
    "resolved_revision",
    "base_model",
    "checkpoint",
    "hookpoint",
    "layer",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def candidate_scores(
    summary_path: Path,
    activation_dir: Path,
    audit_path: Path,
    task_rows: dict[str, dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    eligible = {
        row["run_id"] for row in read_json(audit_path)["runs"] if row["eligible"]
    }
    candidates: dict[tuple[str, int], dict[str, Any]] = {}
    for path in activation_score_paths(summary_path, activation_dir):
        score = read_json(path)
        if score.get("run_id") not in eligible or score.get("exact_match"):
            continue
        task_name = score["task"]
        if task_name not in task_rows:
            continue
        key = (task_name, int(score["feature_id"]))
        candidates[key] = score
    return candidates


def parse_layer_paths(values: list[str]) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for value in values:
        layer, separator, path = value.partition("=")
        if not separator or not layer.isdigit() or not path:
            raise ValueError("layer paths must use LAYER=PATH")
        output[int(layer)] = Path(path)
    return output


def alpha_grid(expert_alpha: float) -> list[float]:
    return [round(expert_alpha * multiplier, 6) for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5)]


def fallback_alpha_grid(expert_alpha: float) -> list[float]:
    return [round(expert_alpha * multiplier, 6) for multiplier in (0.0625, 0.125, 0.25, 0.375)]


def task_layer(task: dict[str, Any]) -> int:
    match = re.match(r"blocks\.(\d+)\.", task["sae"]["hook"])
    if not match:
        raise ValueError(f"cannot infer layer from {task['sae']['hook']}")
    return int(match.group(1))


def checkpoint_provenance(source: dict[str, Any]) -> dict[str, Any]:
    return {key: source[key] for key in CHECKPOINT_PROVENANCE_FIELDS}


def feature_path(feature_dir: Path, layer: int, feature_id: int) -> str:
    return str(
        feature_dir / f"gemma2_9b_it_l{layer}_w131k_feature_{feature_id}.npz"
    )


def reusable_candidate_ids(directories: list[Path]) -> set[str]:
    task_ids: set[str] = set()
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            if path.name == "summary.json":
                continue
            result = read_json(path)
            if "feature" not in result or "steering" not in result:
                raise ValueError(f"invalid reusable steering result: {path}")
            task_ids.add(path.stem)
    return task_ids


def extract_features(
    params_path: Path,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    feature_ids = sorted({int(row["feature_id"]) for row in rows})
    if not feature_ids:
        return
    with np.load(params_path) as params:
        w_dec = params["W_dec"]
        w_enc = params["W_enc"]
        b_dec = params["b_dec"].astype(np.float32)
        b_enc = params["b_enc"]
        threshold = params["threshold"]
        output_dir.mkdir(parents=True, exist_ok=True)
        for feature_id in feature_ids:
            row = next(item for item in rows if int(item["feature_id"]) == feature_id)
            stem = output_dir / f"gemma2_9b_it_l{row['layer']}_w131k_feature_{feature_id}"
            np.savez(
                stem.with_suffix(".npz"),
                decoder=w_dec[feature_id].astype(np.float32),
                encoder=w_enc[:, feature_id].astype(np.float32),
                b_dec=b_dec,
                b_enc=np.float32(b_enc[feature_id]),
                threshold=np.float32(threshold[feature_id]),
            )
            source = checkpoint_provenance(row["source"])
            stem.with_suffix(".json").write_text(
                json.dumps(source, indent=2) + "\n", encoding="utf-8"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deduplicated agent steering jobs.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--activation-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--full-sae", action="append", required=True)
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-candidate-result-dir",
        action="append",
        type=Path,
        default=[],
        help="Omit candidate task IDs with an existing reusable steering result.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trial-id", required=True)
    args = parser.parse_args()

    benchmark = read_json(args.benchmark)
    task_rows = {row["task"]: row for row in benchmark["tasks"]}
    candidates = candidate_scores(
        args.activation_summary,
        args.activation_dir,
        args.audit,
        task_rows,
    )
    reusable_ids = reusable_candidate_ids(args.exclude_candidate_result_dir)

    prepared = []
    reused_candidate_pairs = 0
    expert_rows = []
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for (task_name, feature_id), score in candidates.items():
        benchmark_row = task_rows[task_name]
        task = read_json(ROOT / task_name)
        reference = read_json(ROOT / benchmark_row["reference"])
        feature_case = read_json(ROOT / reference["feature_case"])
        layer = task_layer(task)
        max_new_tokens = int(reference["protocol"]["max_new_tokens"])
        candidate_id = f"{task['task_id']}__candidate_{feature_id}"
        if candidate_id in reusable_ids:
            reused_candidate_pairs += 1
            continue
        row = {
            "id": candidate_id,
            "task": task_name,
            "feature_id": feature_id,
            "layer": layer,
            "source": feature_case,
            "feature": feature_path(args.feature_dir, layer, feature_id),
            "suite": reference["suite"],
            "alphas": ",".join(str(value) for value in alpha_grid(float(reference["protocol"]["alpha"]))),
            "fallback_alphas": ",".join(
                str(value)
                for value in fallback_alpha_grid(float(reference["protocol"]["alpha"]))
            ),
            "screen_selected_alpha": reference["protocol"]["alpha"],
        }
        if reference.get("concept_id"):
            row["concept_id"] = reference["concept_id"]
        prepared.append(row)
        groups[(layer, max_new_tokens)].append(row)

    expert_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for task_name, benchmark_row in task_rows.items():
        task = read_json(ROOT / task_name)
        reference = read_json(ROOT / benchmark_row["reference"])
        feature_case = read_json(ROOT / reference["feature_case"])
        layer = task_layer(task)
        feature_id = int(benchmark_row["expert_feature_id"])
        max_new_tokens = int(reference["protocol"]["max_new_tokens"])
        alpha = float(reference["protocol"]["alpha"])
        row = {
            "id": f"{task['task_id']}__expert_anchor",
            "task": task_name,
            "feature_id": feature_id,
            "layer": layer,
            "source": feature_case,
            "feature": feature_path(args.feature_dir, layer, feature_id),
            "suite": reference["suite"],
            "alphas": str(alpha),
            "screen_selected_alpha": alpha,
        }
        if reference.get("concept_id"):
            row["concept_id"] = reference["concept_id"]
        expert_rows.append(row)
        expert_groups[(layer, max_new_tokens)].append(row)

    sae_paths = parse_layer_paths(args.full_sae)
    extraction_rows = prepared + expert_rows
    for layer in sorted({row["layer"] for row in extraction_rows}):
        if layer not in sae_paths:
            parser.error(f"missing full SAE for layer {layer}")
        extract_features(
            sae_paths[layer],
            [row for row in extraction_rows if row["layer"] == layer],
            args.feature_dir,
        )

    manifests = []
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for (layer, max_new_tokens), rows in sorted(groups.items()):
        output = args.manifest_dir / f"agent_candidates_l{layer}_t{max_new_tokens}.json"
        tasks = [
            {
                key: row[key]
                for key in ("id", "feature", "suite", "alphas", "fallback_alphas", "screen_selected_alpha", "concept_id")
                if key in row
            }
            for row in sorted(rows, key=lambda item: item["id"])
        ]
        payload = {
            "schema": 1,
            "stage": "formal_steering",
            "trial_id": args.trial_id,
            "model_path": args.model_path,
            "positions": "all",
            "alphas": "",
            "max_new_tokens": max_new_tokens,
            "seed": 0,
            "strength_policy": "expert_centered_calibration_grid",
            "tasks": tasks,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifests.append({"path": str(output), "layer": layer, "tasks": len(tasks)})
    for (layer, max_new_tokens), rows in sorted(expert_groups.items()):
        output = args.manifest_dir / f"expert_replay_l{layer}_t{max_new_tokens}.json"
        tasks = [
            {
                key: row[key]
                for key in ("id", "feature", "suite", "alphas", "screen_selected_alpha", "concept_id")
                if key in row
            }
            for row in sorted(rows, key=lambda item: item["id"])
        ]
        payload = {
            "schema": 1,
            "stage": "formal_steering",
            "trial_id": args.trial_id,
            "model_path": args.model_path,
            "positions": "all",
            "alphas": "",
            "max_new_tokens": max_new_tokens,
            "seed": 0,
            "strength_policy": "frozen_expert_replay",
            "tasks": tasks,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifests.append({"path": str(output), "layer": layer, "tasks": len(tasks)})
    print(
        json.dumps(
            {
                "candidate_pairs": len(prepared),
                "reused_candidate_pairs": reused_candidate_pairs,
                "expert_tasks": len(expert_rows),
                "manifests": manifests,
            }
        )
    )


if __name__ == "__main__":
    main()
