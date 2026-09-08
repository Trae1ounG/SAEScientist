#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
JUDGE_SCRIPT = PROJECT / "scripts" / "judge_feature_steering.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def task_summary_path(output_dir: Path, task_id: str) -> Path:
    return output_dir / f"{task_id}_summary.json"


def is_complete_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        summary = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    expected_rows = summary.get("expected_rows")
    return (
        isinstance(expected_rows, int)
        and expected_rows > 0
        and summary.get("valid_rows") == expected_rows
        and summary.get("error_rows") == 0
    )


def compact_task_summary(task: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    evaluation = summary.get("admission_evaluation", {})
    quality = summary.get("quality", {})
    return {
        "id": task["id"],
        "concept_id": task.get("concept_id"),
        "summary": str(summary_path),
        "stable": bool(quality.get("stable", False)),
        "causal_stable": bool(quality.get("causal_stable", False)),
        "usable_steering": bool(quality.get("usable_steering", quality.get("stable", False))),
        "feature_target_score": evaluation.get("feature_target_score"),
        "feature_success_rate": evaluation.get("feature_success_rate"),
        "usable_target_rate": evaluation.get("usable_target_rate"),
        "nondegenerate_rate": evaluation.get("nondegenerate_rate"),
        "rerun_agreement": evaluation.get("rerun_agreement"),
        "activation_failures": quality.get("activation_failures", []),
        "causal_failures": quality.get("causal_failures", []),
    }


def build_command(
    *,
    task: dict[str, Any],
    result_dir: Path,
    output_dir: Path,
    suite: Path | None,
    provider: str,
    model_name: str,
    api_key_env: str,
    judge_workers: int,
    repeats: int,
    seed: int,
    stage: str = "formal_steering",
) -> list[str]:
    task_suite = suite or resolve_project_path(task["suite"])
    command = [
        sys.executable,
        str(JUDGE_SCRIPT),
        "--result",
        str(result_dir / f"{task['id']}.json"),
        "--suite",
        str(task_suite),
        "--output-prefix",
        str(output_dir / task["id"]),
        "--model-name",
        model_name,
        "--provider",
        provider,
        "--api-key-env",
        api_key_env,
        "--workers",
        str(judge_workers),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--stage",
        stage,
    ]
    if task.get("concept_id"):
        command.extend(["--concept-id", str(task["concept_id"])])
    return command


def run_task(
    task: dict[str, Any],
    *,
    result_dir: Path,
    output_dir: Path,
    suite: Path | None,
    provider: str,
    model_name: str,
    api_key_env: str,
    judge_workers: int,
    repeats: int,
    seed: int,
    stage: str = "formal_steering",
) -> dict[str, Any]:
    summary_path = task_summary_path(output_dir, task["id"])
    if is_complete_summary(summary_path):
        return {"status": "skipped", **compact_task_summary(task, summary_path)}

    result_path = result_dir / f"{task['id']}.json"
    if not result_path.exists():
        return {
            "id": task["id"],
            "concept_id": task.get("concept_id"),
            "status": "failed",
            "returncode": None,
            "error": f"missing result: {result_path}",
        }

    command = build_command(
        task=task,
        result_dir=result_dir,
        output_dir=output_dir,
        suite=suite,
        provider=provider,
        model_name=model_name,
        api_key_env=api_key_env,
        judge_workers=judge_workers,
        repeats=repeats,
        seed=seed,
        stage=stage,
    )
    run = subprocess.run(
        command,
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        return {
            "id": task["id"],
            "concept_id": task.get("concept_id"),
            "status": "failed",
            "returncode": run.returncode,
            "stdout_tail": run.stdout[-2000:],
            "stderr_tail": run.stderr[-2000:],
        }
    if not is_complete_summary(summary_path):
        return {
            "id": task["id"],
            "concept_id": task.get("concept_id"),
            "status": "failed",
            "returncode": 0,
            "error": f"incomplete judge summary: {summary_path}",
        }
    return {"status": "judged", **compact_task_summary(task, summary_path)}


def validate_manifest(
    manifest: dict[str, Any], stage: str = "formal_steering"
) -> list[dict[str, Any]]:
    if manifest.get("stage") != stage:
        raise ValueError(f"manifest stage must be {stage}")
    tasks = manifest.get("tasks", [])
    if not tasks:
        raise ValueError("manifest contains no tasks")
    ids = [task.get("id") for task in tasks]
    if any(not task_id for task_id in ids):
        raise ValueError("every task needs an id")
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")
    missing_suite = [task["id"] for task in tasks if "suite" not in task]
    if missing_suite:
        raise ValueError(f"tasks missing suite: {missing_suite}")
    return tasks


def summarize(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": 1,
        "manifest": str(args.manifest),
        "result_dir": str(args.result_dir),
        "output_dir": str(args.output_dir),
        "judge_model": args.model_name,
        "judge_provider": args.provider,
        "stage": args.stage,
        "workers": args.workers,
        "judge_workers": args.judge_workers,
        "repeats": args.repeats,
        "totals": {
            "tasks": len(rows),
            "judged": sum(row["status"] == "judged" for row in rows),
            "skipped": sum(row["status"] == "skipped" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "stable": sum(bool(row.get("stable")) for row in rows),
            "causal_stable": sum(bool(row.get("causal_stable")) for row in rows),
            "usable_steering": sum(bool(row.get("usable_steering")) for row in rows),
        },
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch PE-judge completed formal steering results."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("formal_steering", "steering_screen"),
        default="formal_steering",
    )
    parser.add_argument("--suite", type=Path)
    parser.add_argument(
        "--provider", choices=("azure-openai",), default="azure-openai"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key-env", default="AZURE_OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    if args.workers < 1 or args.judge_workers < 1:
        raise ValueError("workers and judge-workers must be positive")
    if not args.model_name.strip() or "$" in args.model_name:
        raise ValueError("model-name must be a resolved endpoint, not a shell variable")

    manifest = load_json(args.manifest)
    tasks = validate_manifest(manifest, args.stage)
    args.result_dir = resolve_project_path(args.result_dir)
    args.output_dir = resolve_project_path(args.output_dir)
    args.suite = resolve_project_path(args.suite) if args.suite else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        resolve_project_path(args.summary)
        if args.summary
        else args.output_dir / "batch_summary.json"
    )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_task,
                task,
                result_dir=args.result_dir,
                output_dir=args.output_dir,
                suite=args.suite,
                provider=args.provider,
                model_name=args.model_name,
                api_key_env=args.api_key_env,
                judge_workers=args.judge_workers,
                repeats=args.repeats,
                seed=args.seed,
                stage=args.stage,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["id"])

    summary = summarize(args, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), **summary["totals"]}))


if __name__ == "__main__":
    main()
