#!/usr/bin/env python3
"""Run one Agent on benchmark tasks, then score its locally generated evaluations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, stdev
import tempfile

from sae_scientist.cli import reproduce
from sae_scientist.paper_scores import SCORING_VERSION, paper_scores
from sae_scientist.preflight import check_environment
from sae_scientist.provenance import environment_record
from sae_scientist.suites import load_suite, steering_sets

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("Rank", "Activation", "Steering", "Overall")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def plan(config: dict, task_ids: list[str] | None = None,
         replicates: int | None = None, root: Path = ROOT) -> list[dict]:
    count = config.get("replicates", 3) if replicates is None else replicates
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("replicates must be a positive integer")
    if config["agent"]["harness"] not in ("codex", "cursor"):
        raise ValueError("the audited runner supports codex and cursor harnesses")
    for section, field in (("agent", "model"), ("agent", "cli"), ("judge", "model")):
        value = config[section].get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{field} must be a nonempty string")
    timeout = config["agent"].get("timeout_minutes", 60)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("agent.timeout_minutes must be positive and finite")
    probe = config.get("probe", {})
    if type(probe.get("workers", 1)) is not int or probe.get("workers", 1) != 1:
        raise ValueError("this sequential runner uses one probe worker")
    if probe.get("host", "127.0.0.1") not in ("127.0.0.1", "localhost"):
        raise ValueError("probe.host must be 127.0.0.1 or localhost; the service is unauthenticated")
    port = probe.get("port", 8765)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("probe.port must be an integer from 1 to 65535")
    if config["judge"].get("provider") != "azure-openai":
        raise ValueError("the judge adapter supports azure-openai")
    if config["judge"].get("repeats", 2) != 2:
        raise ValueError("the benchmark protocol uses two judge passes")
    benchmark = read(resolve(config["benchmark"], root))
    selected = []
    known = set()
    for entry in benchmark["tasks"]:
        task_path = resolve(entry["task"], root)
        task = read(task_path)
        task_id = task["task_id"]
        if task_id in known or task_id in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
            raise ValueError(f"invalid or duplicate task ID: {task_id}")
        known.add(task_id)
        if task_ids and task_id not in task_ids:
            continue
        reference = read(resolve(entry["reference"], root))
        source = read(resolve(reference["feature_case"], root))
        suite_path = resolve(entry["suite"], root)
        suite = load_suite(suite_path, entry.get("concept_id"))
        calibration, evaluation = steering_sets(suite, suite_path)
        if not calibration or not evaluation or not suite["activation_cases"]:
            raise ValueError(f"incomplete dataset: {task_id}")
        match = re.fullmatch(r"layer_(\d+)/width_(131k)/average_l0_(\d+)/params.npz", source["checkpoint"])
        if not match:
            raise ValueError(f"unsupported SAE checkpoint: {task_id}")
        layer, width, l0 = match.groups()
        if (source["feature_id"] != entry["expert_feature_id"]
                or task["sae"]["hook"] != f"blocks.{layer}.hook_resid_post"
                or source["base_model"] != task["model"]
                or source["repo"] != task["sae"]["repo"]
                or source["checkpoint"] != task["sae"]["release"] + "/params.npz"):
            raise ValueError(f"task and Expert dictionary disagree: {task_id}")
        base = {
            "model_path": str(resolve(config["model_path"], root)),
            "sae": {
                "path": str(resolve(config["sae_root"], root) / source["checkpoint"]),
                "layer": int(layer), "width": width, "average_l0": int(l0),
                "resolved_revision": source["resolved_revision"],
            },
            "task": str(task_path), "suite": str(suite_path),
            "expert_feature_id": entry["expert_feature_id"],
            "input_sha256": {
                name: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                               allow_nan=False).encode("utf-8")).hexdigest()
                for name, value in {
                    "task": task, "expert_config": reference, "expert_source": source,
                    "suite": suite, "calibration": calibration, "evaluation": evaluation,
                }.items()
            },
            "agent": copy.deepcopy(config["agent"]),
            "probe": copy.deepcopy(config.get("probe", {})),
            "judge": copy.deepcopy(config["judge"]),
            "steering": {
                "policy": "expert_centered",
                "expert_alpha": reference["protocol"]["alpha"],
                "max_new_tokens": reference["protocol"]["max_new_tokens"],
            },
        }
        if entry.get("concept_id"):
            base["concept_id"] = entry["concept_id"]
        selected.append((task_id, base))
    if task_ids and set(task_ids) - known:
        raise ValueError(f"unknown task IDs: {sorted(set(task_ids) - known)}")
    if not selected:
        raise ValueError("no tasks selected")
    jobs = []
    destination = resolve(config["output_dir"], root)
    for replicate in range(1, count + 1):
        for task_id, base in selected:
            cell = copy.deepcopy(base)
            cell["agent"]["run_id"] = f"{task_id}-rep-{replicate:02d}"
            cell["output_dir"] = str(destination / f"rep{replicate:02d}" / task_id)
            jobs.append({"task_id": task_id, "replicate": replicate, "config": cell})
    return jobs


def summarize(jobs: list[dict]) -> dict:
    cells = []
    seen = set()
    for job in jobs:
        key = (job["replicate"], job["task_id"])
        if key in seen:
            raise ValueError(f"duplicate task/replicate: {key}")
        seen.add(key)
        folder = Path(job["config"]["output_dir"])
        if read(folder / "config.json") != job["config"]:
            raise ValueError(f"configuration changed: {folder}")
        activation = read(folder / "activation.json")
        if (activation["task_id"] != job["task_id"]
                or activation["run_id"] != job["config"]["agent"]["run_id"]
                or activation["model"] != job["config"]["agent"]["model"]
                or activation["harness"] != job["config"]["agent"]["harness"]):
            raise ValueError(f"activation belongs to another run: {folder}")
        audit = read(folder / "audit.json")["runs"]
        if len(audit) != 1 or audit[0]["run_id"] != activation["run_id"] or not audit[0]["eligible"]:
            raise ValueError(f"ineligible run: {folder}")
        judgment = read(folder / "judgment_summary.json")
        suite_path = Path(job["config"]["suite"])
        suite = load_suite(suite_path, job["config"].get("concept_id"))
        _, evaluation = steering_sets(suite, suite_path)
        expected_rows = len(evaluation) * job["config"]["judge"].get("repeats", 2)
        if judgment["expected_rows"] != expected_rows:
            raise ValueError(f"judge count does not match the evaluation protocol: {folder}")
        if resolve(judgment["result"], ROOT) != (folder / "steering.json").resolve():
            raise ValueError(f"judge belongs to another steering evaluation: {folder}")
        steering = read(folder / "steering.json")
        if steering["feature"]["feature_id"] != activation["feature_id"]:
            raise ValueError(f"steering feature does not match submission: {folder}")
        result = paper_scores(activation, judgment)
        cells.append({"task_id": job["task_id"], "replicate": job["replicate"],
                      "scores": result["scores"]})
    if not cells:
        raise ValueError("no completed evaluations")
    replicates = sorted({c["replicate"] for c in cells})
    expected = {j["task_id"] for j in jobs}
    run_rows = []
    for replicate in replicates:
        rows = [c for c in cells if c["replicate"] == replicate]
        if {c["task_id"] for c in rows} != expected:
            raise ValueError("task coverage differs between replicates")
        run_rows.append({
            "replicate": replicate, "tasks": len(rows),
            "scores": {m: mean(c["scores"][m] for c in rows) for m in METRICS},
        })
    return {
        "scoring_version": SCORING_VERSION,
        "agent": jobs[0]["config"]["agent"]["model"],
        "task_count": len(expected), "replicates": len(replicates),
        "scores": {
            m: {"mean": mean(r["scores"][m] for r in run_rows),
                "sample_std": stdev(r["scores"][m] for r in run_rows) if len(run_rows) > 1 else None}
            for m in METRICS
        },
        "per_run": run_rows, "per_task": cells,
    }


def write_summary(destination: Path, summary: dict) -> None:
    """Replace a summary only after the complete new JSON has been written."""
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".summary-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", action="append", help="select a task; repeat to select several")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--output-dir", type=Path, help="override the configured local output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate inputs and show the plan without inference or writes")
    mode.add_argument("--score-only", action="store_true", help="recompute summary from this plan's completed local outputs")
    mode.add_argument("--check", action="store_true", help="check local weights, GPU, packages, CLI, and judge configuration without model calls")
    args = parser.parse_args()
    config = read(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    jobs = plan(config, args.task_id, args.replicates)
    if args.dry_run:
        print(json.dumps({"planned_runs": len(jobs), "tasks": sorted({j["task_id"] for j in jobs}),
                          "output_dir": str(resolve(config["output_dir"], ROOT))}, indent=2))
        return
    if args.check:
        report = check_environment(jobs)
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
        return
    if not args.score_only:
        destination = resolve(config["output_dir"], ROOT)
        if destination.exists():
            raise FileExistsError(f"use a new output_dir for an independent experiment: {destination}")
        report = check_environment(jobs)
        if not report["ok"]:
            raise ValueError("Experiment setup is incomplete:\n- " + "\n- ".join(report["errors"]))
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "environment.json").write_text(
            json.dumps(environment_record(ROOT), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for index, job in enumerate(jobs, 1):
            path = Path(job["config"]["output_dir"]) / "config.json"
            path.parent.mkdir(parents=True, exist_ok=False)
            path.write_text(json.dumps(job["config"], indent=2) + "\n")
            print(f"[{index}/{len(jobs)}] {job['task_id']} replicate {job['replicate']}", flush=True)
            reproduce(path)
    summary = summarize(jobs)
    destination = resolve(config["output_dir"], ROOT)
    write_summary(destination / "summary.json", summary)
    print(json.dumps(summary["scores"], indent=2))


if __name__ == "__main__":
    main()
