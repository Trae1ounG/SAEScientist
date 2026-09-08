from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunLayout:
    root: Path
    workspace: Path
    logs: Path


def initialize_run(
    *,
    runs_root: Path,
    run_id: str,
    task_path: Path,
    prompt_path: Path,
    harness: str,
    agent_model: str,
    source_commit: str,
    reasoning_effort: str | None = None,
) -> RunLayout:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("schema") != 1 or not task.get("task_id"):
        raise ValueError("task must have schema 1 and a task_id")

    root = runs_root / run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {root}")
    workspace = root / "workspace"
    logs = root / "logs"
    workspace.mkdir(parents=True)
    logs.mkdir()
    shutil.copy2(task_path, workspace / "task.json")
    shutil.copy2(prompt_path, workspace / "prompt.md")

    manifest = {
        "schema": 1,
        "run_id": run_id,
        "task_id": task["task_id"],
        "harness": harness,
        "agent_model": agent_model,
        "network": "disabled",
        "source_commit": source_commit,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if reasoning_effort is not None:
        manifest["reasoning_effort"] = reasoning_effort
    (root / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return RunLayout(root=root, workspace=workspace, logs=logs)


def read_submission(workspace: Path, feature_count: int) -> dict[str, int]:
    path = workspace / "submission.json"
    if not path.is_file():
        raise ValueError("submission.json is missing")
    submission = json.loads(path.read_text(encoding="utf-8"))
    if set(submission) != {"feature_id"}:
        raise ValueError("submission.json must contain only feature_id")
    feature_id = submission["feature_id"]
    if isinstance(feature_id, bool) or not isinstance(feature_id, int):
        raise ValueError("feature_id must be an integer")
    if not 0 <= feature_id < feature_count:
        raise ValueError("feature_id is outside the SAE feature range")
    return {"feature_id": feature_id}


def recover_submission_from_trace(
    trace_path: Path, workspace: Path, feature_count: int
) -> dict[str, int]:
    assistant_texts: list[str] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            content = event.get("message", {}).get("content", [])
            if isinstance(content, str):
                assistant_texts.append(content)
            elif isinstance(content, list):
                assistant_texts.extend(
                    item["text"]
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
        elif event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                assistant_texts.append(item["text"])
        elif event.get("type") == "result" and isinstance(event.get("result"), str):
            assistant_texts.append(event["result"])

    if not assistant_texts:
        raise ValueError("submission.json is missing and the trace has no final answer")
    candidates: list[dict[str, int]] = []
    for raw in re.findall(r"\{[^{}]*\}", assistant_texts[-1]):
        try:
            submission = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if set(submission) != {"feature_id"}:
            continue
        feature_id = submission["feature_id"]
        if (
            not isinstance(feature_id, bool)
            and isinstance(feature_id, int)
            and 0 <= feature_id < feature_count
        ):
            candidates.append({"feature_id": feature_id})
    unique = {candidate["feature_id"] for candidate in candidates}
    if len(unique) != 1:
        raise ValueError(
            "submission.json is missing and the final answer does not contain one "
            "unambiguous feature_id"
        )
    submission = {"feature_id": unique.pop()}
    (workspace / "submission.json").write_text(
        json.dumps(submission, indent=2) + "\n", encoding="utf-8"
    )
    return submission

