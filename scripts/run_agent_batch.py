#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT = ROOT / "scripts" / "run_agent.py"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_probe_urls(values: list[str]) -> dict[int, str]:
    output: dict[int, str] = {}
    for value in values:
        layer, separator, url = value.partition("=")
        if not separator or not layer.isdigit() or not url:
            raise ValueError("probe URLs must use LAYER=URL")
        output[int(layer)] = url
    return output


def task_layer(task: dict[str, Any]) -> int:
    match = re.match(r"blocks\.(\d+)\.", task["sae"]["hook"])
    if not match:
        raise ValueError(f"cannot infer SAE layer from hook: {task['sae']['hook']}")
    return int(match.group(1))


def select_run_id(
    task_id: str,
    model_id: str,
    retry_failed: bool,
    max_retries: int,
    eligible_run_ids: set[str] | None = None,
    audited_run_ids: set[str] | None = None,
    replicate_index: int = 1,
) -> tuple[str, str]:
    base = f"{task_id.replace('_', '-')}-offline-{model_id}"
    if replicate_index > 1:
        base = f"{base}-rep-{replicate_index:02d}"
    run_ids = [f"{base}-01"] + [
        f"{base}-retry-{attempt:02d}" for attempt in range(1, max_retries + 1)
    ]
    for index, run_id in enumerate(run_ids):
        run_dir = ROOT / "runs" / run_id
        if not run_dir.exists():
            return run_id, "new"
        result_path = run_dir / "result.json"
        if not result_path.exists():
            return run_id, "incomplete"
        status = read_json(result_path).get("status")
        if status == "submitted":
            if eligible_run_ids is None or run_id in eligible_run_ids:
                return run_id, "complete"
            if audited_run_ids is None or run_id not in audited_run_ids:
                return run_id, "incomplete"
        elif status != "failed":
            return run_id, "incomplete"
        if not retry_failed or index == max_retries:
            return run_id, "incomplete"
    return f"{base}-retry-{max_retries:02d}", "incomplete"


def run_one(
    task_path: Path,
    model: dict[str, Any],
    probe_urls: dict[int, str],
    timeout_minutes: float,
    retry_failed: bool,
    max_retries: int,
    eligible_run_ids: set[str] | None,
    audited_run_ids: set[str] | None,
    replicate_index: int,
) -> dict[str, Any]:
    task = read_json(task_path)
    layer = task_layer(task)
    if layer not in probe_urls:
        return {
            "task": str(task_path),
            "model_id": model["id"],
            "status": "failed",
            "error": f"no probe URL for layer {layer}",
        }
    run_id, run_state = select_run_id(
        task["task_id"],
        model["id"],
        retry_failed,
        max_retries,
        eligible_run_ids,
        audited_run_ids,
        replicate_index,
    )
    run_dir = ROOT / "runs" / run_id
    if run_state == "complete":
        return {
            "task": str(task_path),
            "task_id": task["task_id"],
            "model_id": model["id"],
            "run_id": run_id,
            "status": "skipped",
        }
    if run_state == "incomplete":
        return {
            "task": str(task_path),
            "task_id": task["task_id"],
            "model_id": model["id"],
            "run_id": run_id,
            "status": "failed",
            "error": "run directory already exists but is not complete",
        }

    cli_path = shutil.which(model["cli"])
    if not cli_path:
        return {
            "task": str(task_path),
            "task_id": task["task_id"],
            "model_id": model["id"],
            "run_id": run_id,
            "status": "failed",
            "error": f"CLI not found: {model['cli']}",
        }
    command = [
        sys.executable,
        str(RUN_AGENT),
        "--task",
        str(task_path),
        "--run-id",
        run_id,
        "--harness",
        model["harness"],
        "--model",
        model["model"],
        "--cli-path",
        cli_path,
        "--probe-url",
        probe_urls[layer],
        "--timeout-minutes",
        str(timeout_minutes),
    ]
    if model.get("reasoning_effort"):
        command.extend(["--reasoning-effort", model["reasoning_effort"]])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "task": str(task_path),
        "task_id": task["task_id"],
        "model_id": model["id"],
        "run_id": run_id,
        "status": "submitted" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one comparable agent grid.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--probe-url", action="append", required=True)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-minutes", type=float, default=60)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--replicate-index", type=int, default=1)
    parser.add_argument(
        "--audit",
        type=Path,
        help="Treat only run IDs marked eligible by this trace audit as complete.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.replicate_index < 1:
        raise ValueError("replicate index must be positive")

    benchmark = read_json(args.benchmark)
    model_rows = read_json(args.models)["models"]
    if args.model_id:
        selected = set(args.model_id)
        model_rows = [row for row in model_rows if row["id"] in selected]
        missing = selected - {row["id"] for row in model_rows}
        if missing:
            raise ValueError(f"unknown model ids: {sorted(missing)}")
    probe_urls = parse_probe_urls(args.probe_url)
    eligible_run_ids = None
    audited_run_ids = None
    if args.audit:
        audit_rows = read_json(args.audit)["runs"]
        audited_run_ids = {row["run_id"] for row in audit_rows}
        eligible_run_ids = {row["run_id"] for row in audit_rows if row["eligible"]}
    task_paths = [ROOT / row["task"] for row in benchmark["tasks"]]
    if args.task_id:
        selected_tasks = set(args.task_id)
        task_paths = [
            path for path in task_paths
            if read_json(path)["task_id"] in selected_tasks
        ]
        missing_tasks = selected_tasks - {
            read_json(path)["task_id"] for path in task_paths
        }
        if missing_tasks:
            raise ValueError(f"unknown task ids: {sorted(missing_tasks)}")
    jobs = [
        (task_path, model)
        for task_path in task_paths
        for model in model_rows
    ]

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_one,
                task_path,
                model,
                probe_urls,
                args.timeout_minutes,
                args.retry_failed,
                args.max_retries,
                eligible_run_ids,
                audited_run_ids,
                args.replicate_index,
            )
            for task_path, model in jobs
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row.get("task_id", ""), row["model_id"]))
    payload = {
        "schema": 1,
        "benchmark": str(args.benchmark),
        "models": [row["id"] for row in model_rows],
        "replicate_index": args.replicate_index,
        "jobs": len(rows),
        "submitted": sum(row["status"] == "submitted" for row in rows),
        "skipped": sum(row["status"] == "skipped" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("jobs", "submitted", "skipped", "failed")}))


if __name__ == "__main__":
    main()

