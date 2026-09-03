#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

from sae_bench.scoring import (
    gt_normalized_metrics,
    spearman_correlation,
    summarize_rank_rows,
)
from sae_bench.suites import load_suite


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_probe_urls(values: list[str]) -> dict[int, str]:
    urls: dict[int, str] = {}
    for value in values:
        layer, separator, url = value.partition("=")
        if not separator or not layer.isdigit() or not url:
            raise ValueError("probe URLs must use LAYER=URL")
        urls[int(layer)] = url
    return urls


def task_layer(task: dict[str, Any]) -> int:
    match = re.match(r"blocks\.(\d+)\.", task["sae"]["hook"])
    if not match:
        raise ValueError(f"cannot infer SAE layer from hook: {task['sae']['hook']}")
    return int(match.group(1))


def submitted_run(
    task_id: str,
    model_id: str,
    runs_root: Path,
    eligible_run_ids: set[str] | None = None,
) -> tuple[str, Path] | None:
    base = f"{task_id.replace('_', '-')}-offline-{model_id}"
    retry_pattern = re.compile(rf"^{re.escape(base)}-retry-(\d+)$")
    retries = sorted(
        (
            (int(match.group(1)), path)
            for path in runs_root.glob(f"{base}-retry-*")
            if (match := retry_pattern.fullmatch(path.name))
        )
    )
    candidates = [runs_root / f"{base}-01"] + [path for _, path in retries]
    for path in candidates:
        result = path / "result.json"
        if (
            (eligible_run_ids is None or path.name in eligible_run_ids)
            and result.exists()
            and read_json(result).get("status") == "submitted"
        ):
            return path.name, path
    return None


def score_one(job: dict[str, Any], probe_url: str, output_dir: Path) -> dict[str, Any]:
    run_root = Path(job["run_root"])
    output = output_dir / f"{job['run_id']}.json"
    if output.exists():
        return {"run_id": job["run_id"], "status": "skipped", "output": str(output)}

    task = read_json(Path(job["task_path"]))
    reference = read_json(Path(job["reference_path"]))
    result = read_json(run_root / "result.json")
    metadata = read_json(run_root / "run.json")
    feature_id = int(read_json(run_root / "workspace" / "submission.json")["feature_id"])
    expert_feature_id = int(job["expert_feature_id"])
    suite_path = ROOT / reference["suite"]
    suite = load_suite(suite_path, reference.get("concept_id"))
    cases = suite["activation_cases"]

    request = Request(
        probe_url.rstrip("/") + "/probe",
        data=json.dumps(
            {
                "texts": [case["text"] for case in cases],
                "top_k": 1,
                "feature_ids": [feature_id, expert_feature_id],
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=600) as response:
        probed = json.loads(response.read())["results"]
    if len(probed) != len(cases):
        raise ValueError("probe response length does not match the evaluation suite")

    rows = []
    for case, measured in zip(cases, probed):
        candidate, expert = measured["selected_features"]
        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "activation": candidate["activation"],
                "rank": candidate["rank"],
                "expert_activation": expert["activation"],
                "expert_rank": expert["rank"],
            }
        )
    expert_rows = [
        {
            "id": row["id"],
            "label": row["label"],
            "activation": row["expert_activation"],
            "rank": row["expert_rank"],
        }
        for row in rows
    ]
    feature_count = int(task["sae"]["feature_count"])
    activation_rank = summarize_rank_rows(rows, feature_count)
    expert_activation_rank = summarize_rank_rows(expert_rows, feature_count)
    spearman = spearman_correlation(
        [row["activation"] for row in rows],
        [row["expert_activation"] for row in rows],
    )
    payload = {
        "schema": 1,
        "run_id": job["run_id"],
        "task": job["task"],
        "concept_id": reference.get("concept_id"),
        "layer": job["layer"],
        "harness": metadata["harness"],
        "model": metadata["agent_model"],
        "source_commit": metadata["source_commit"],
        "elapsed_seconds": result["elapsed_seconds"],
        "feature_id": feature_id,
        "expert_feature_id": expert_feature_id,
        "exact_match": feature_id == expert_feature_id,
        "activation_rank": activation_rank,
        "expert_activation_rank": expert_activation_rank,
        "expert_activation_spearman": spearman,
        "gt_normalized": gt_normalized_metrics(
            activation_rank, expert_activation_rank, spearman
        ),
        "cases": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"run_id": job["run_id"], "status": "scored", "output": str(output)}


def collect_jobs(
    benchmark: dict[str, Any],
    models: list[dict[str, Any]],
    runs_root: Path,
    eligible_run_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    jobs = []
    for row in benchmark["tasks"]:
        task_path = ROOT / row["task"]
        task = read_json(task_path)
        layer = task_layer(task)
        for model in models:
            selected = submitted_run(
                task["task_id"], model["id"], runs_root, eligible_run_ids
            )
            if selected is None:
                continue
            episode_id, run_root = selected
            jobs.append(
                {
                    "run_id": episode_id,
                    "run_root": str(run_root),
                    "task": row["task"],
                    "task_path": str(task_path),
                    "reference_path": str(ROOT / row["reference"]),
                    "expert_feature_id": row["expert_feature_id"],
                    "layer": layer,
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Score completed offline agent submissions.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--probe-url", action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--audit",
        type=Path,
        help="Score only run IDs marked eligible by this trace audit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    benchmark = read_json(args.benchmark)
    models = read_json(args.models)["models"]
    urls = parse_probe_urls(args.probe_url)
    eligible_run_ids = None
    if args.audit:
        eligible_run_ids = {
            row["run_id"] for row in read_json(args.audit)["runs"] if row["eligible"]
        }
    jobs = collect_jobs(benchmark, models, args.runs_root, eligible_run_ids)
    missing_layers = sorted({job["layer"] for job in jobs} - set(urls))
    if missing_layers:
        parser.error(f"missing probe URLs for layers: {missing_layers}")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(score_one, job, urls[job["layer"]], args.output_dir): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"run_id": job["run_id"], "status": "failed", "error": str(exc)})
    rows.sort(key=lambda row: row["run_id"])
    summary = {
        "schema": 1,
        "benchmark": str(args.benchmark),
        "eligible_runs": len(jobs),
        "scored": sum(row["status"] == "scored" for row in rows),
        "skipped": sum(row["status"] == "skipped" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "runs": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("eligible_runs", "scored", "skipped", "failed")}))


if __name__ == "__main__":
    main()

