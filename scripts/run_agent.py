#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sae_bench.episode import (
    initialize_run,
    read_submission,
    recover_submission_from_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument(
        "--harness", choices=("codex", "cursor", "claude"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"),
    )
    parser.add_argument("--cli-path", type=Path, required=True)
    parser.add_argument("--probe-url", required=True)
    parser.add_argument("--timeout-minutes", type=float, default=60)
    args = parser.parse_args()
    if args.reasoning_effort is not None and args.harness != "codex":
        parser.error("--reasoning-effort is supported only by the codex harness")

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    layout = initialize_run(
        runs_root=args.runs_root,
        run_id=args.run_id,
        task_path=args.task,
        prompt_path=REPO_ROOT / "prompts" / "discover_feature.md",
        harness=args.harness,
        agent_model=args.model,
        source_commit=source_commit,
        reasoning_effort=args.reasoning_effort,
    )
    task = json.loads((layout.workspace / "task.json").read_text(encoding="utf-8"))
    agent_input = (layout.workspace / "prompt.md").read_text(encoding="utf-8")
    agent_input += "\n\n## Task JSON\n\n```json\n"
    agent_input += json.dumps(task, indent=2) + "\n```\n"
    (layout.workspace / "agent_input.md").write_text(agent_input, encoding="utf-8")
    isolation_canary = layout.root / ".isolation-canary"
    isolation_canary.write_text("not visible to the agent\n", encoding="utf-8")
    adapter = REPO_ROOT / "agents" / args.harness / "solve.sh"
    environment = os.environ.copy()
    environment.update(
        {
            "SAE_AGENT_HOME": str(layout.workspace),
            "SAE_AGENT_MODEL": args.model,
            "SAE_AGENT_CLI": str(args.cli_path.resolve()),
            "SAE_AGENT_ISOLATION_CANARY": str(isolation_canary),
            "SAE_AGENT_PYTHON": sys.executable,
            "SAE_PROBE_MCP": str((REPO_ROOT / "scripts" / "probe_mcp.py").resolve()),
            "SAE_PROBE_URL": args.probe_url,
        }
    )
    if args.reasoning_effort is not None:
        environment["SAE_AGENT_REASONING_EFFORT"] = args.reasoning_effort
    started = time.monotonic()
    status = "failed"
    error = None
    exit_code = None
    submission_source = None
    with (layout.logs / "agent.jsonl").open("w", encoding="utf-8") as stdout, (
        layout.logs / "agent.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                [str(adapter)],
                cwd=layout.workspace,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout_minutes * 60,
                check=False,
            )
            exit_code = completed.returncode
            if exit_code != 0:
                error = f"agent exited with code {exit_code}"
            else:
                feature_count = int(task["sae"]["feature_count"])
                if (layout.workspace / "submission.json").is_file():
                    read_submission(layout.workspace, feature_count)
                    submission_source = "workspace"
                else:
                    recover_submission_from_trace(
                        layout.logs / "agent.jsonl", layout.workspace, feature_count
                    )
                    submission_source = "final_message"
                status = "submitted"
        except subprocess.TimeoutExpired:
            error = "agent timed out"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)

    result = {
        "schema": 1,
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - started,
        "submission": "workspace/submission.json" if status == "submitted" else None,
        "submission_source": submission_source,
        "error": error,
    }
    isolation_canary.unlink(missing_ok=True)
    (layout.root / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run": str(layout.root), **result}, indent=2))
    if status != "submitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

