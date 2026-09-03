#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate independent SAE steering tasks with one Ray GPU each."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--address", default="auto")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if not tasks:
        raise ValueError("manifest contains no tasks")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("task ids must be unique")

    project = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)

    import ray

    ray.init(address=args.address)
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    if cluster_gpus < 1:
        raise RuntimeError("Ray cluster exposes no GPUs")

    @ray.remote(num_gpus=1, max_calls=1)
    def evaluate(task: dict) -> dict:
        output = output_dir / f"{task['id']}.json"
        command = [
            sys.executable,
            str(project / "scripts" / "evaluate_gemma_feature.py"),
            "--model-path",
            manifest["model_path"],
            "--feature",
            str(project / task["feature"]),
            "--suite",
            str(project / task["suite"]),
            "--alphas",
            task["alphas"] if "alphas" in task else manifest["alphas"],
            "--positions",
            manifest.get("positions", "all"),
            "--max-new-tokens",
            str(manifest.get("max_new_tokens", 64)),
            "--seed",
            str(manifest.get("seed", 0)),
            "--trial-id",
            str(manifest["trial_id"]),
            "--output",
            str(output),
        ]
        fallback_alphas = task.get("fallback_alphas", manifest.get("fallback_alphas"))
        if fallback_alphas:
            command.extend(["--fallback-alphas", str(fallback_alphas)])
        if task.get("concept_id"):
            command.extend(["--concept-id", task["concept_id"]])
        if manifest.get("evaluation_limit") is not None:
            command.extend(["--evaluation-limit", str(manifest["evaluation_limit"])])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project / "src")
        env["TOKENIZERS_PARALLELISM"] = "false"
        run = subprocess.run(
            command,
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
        )
        return {
            "id": task["id"],
            "returncode": run.returncode,
            "output": str(output),
            "stdout_tail": run.stdout[-4000:],
            "stderr_tail": run.stderr[-4000:],
        }

    results = ray.get([evaluate.remote(task) for task in tasks])
    summary = {
        "schema": 1,
        "manifest": str(args.manifest),
        "cluster_gpus": cluster_gpus,
        "tasks": results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tasks": len(results),
                "succeeded": sum(row["returncode"] == 0 for row in results),
                "failed": sum(row["returncode"] != 0 for row in results),
                "output": str(summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()

