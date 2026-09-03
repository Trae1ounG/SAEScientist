#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


NETWORK_PATTERN = re.compile(r"(?:https?://|\bgit\s+clone\b)", re.I)
NETWORK_EXECUTABLES = {"curl", "wget", "ssh", "scp", "nc", "ncat"}
PROCESS_INSPECTION_EXECUTABLES = {"env", "printenv", "ps", "pgrep", "top", "lsof"}
CURSOR_LOCAL_TOOLS = {
    "askQuestionToolCall",
    "awaitToolCall",
    "deleteToolCall",
    "editToolCall",
    "getMcpToolsToolCall",
    "globToolCall",
    "grepToolCall",
    "mcpToolCall",
    "readToolCall",
    "shellToolCall",
    "updateTodosToolCall",
    "writeToolCall",
}
CURSOR_OPEN_WORLD_TOOLS = {"taskToolCall", "webFetchToolCall", "webSearchToolCall"}
CODEX_ITEM_TYPES = {
    "agent_message",
    "command_execution",
    "error",
    "file_change",
    "mcp_tool_call",
    "reasoning",
    "todo_list",
    "web_search",
    "web_search_call",
}


def cursor_network_command(body: dict[str, Any]) -> bool:
    args = body.get("args", {})
    if NETWORK_PATTERN.search(str(args.get("command", ""))):
        return True
    parsed = args.get("parsingResult", {}).get("executableCommands", [])
    return any(
        Path(str(command.get("name", ""))).name.lower() in NETWORK_EXECUTABLES
        for command in parsed
    )


def cursor_process_inspection(body: dict[str, Any]) -> bool:
    args = body.get("args", {})
    command = str(args.get("command", ""))
    if re.search(r"(?:^|[\s;&|])/(?:proc|sys)(?:/|\s|$)", command):
        return True
    parsed = args.get("parsingResult", {}).get("executableCommands", [])
    return any(
        Path(str(entry.get("name", ""))).name.lower()
        in PROCESS_INSPECTION_EXECUTABLES
        for entry in parsed
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def current_task_path(metadata: dict[str, Any]) -> Path:
    return Path(__file__).resolve().parents[1] / "tasks" / metadata["task_id"] / "task.json"


def task_snapshot_matches(run_root: Path, metadata: dict[str, Any]) -> bool:
    frozen = run_root / "workspace" / "task.json"
    current = current_task_path(metadata)
    return frozen.exists() and current.exists() and read_json(frozen) == read_json(current)


def inside(path: str, workspace: Path) -> bool:
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        candidate.resolve().relative_to(workspace.resolve())
        return True
    except ValueError:
        return False


def cursor_rule_paths(value: Any) -> list[str]:
    paths = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "relatedCursorRulePaths" and isinstance(nested, list):
                paths.extend(path for path in nested if isinstance(path, str))
            else:
                paths.extend(cursor_rule_paths(nested))
    elif isinstance(value, list):
        for nested in value:
            paths.extend(cursor_rule_paths(nested))
    return paths


def cursor_audit(run_root: Path) -> dict[str, Any]:
    workspace = run_root / "workspace"
    trace = run_root / "logs" / "agent.jsonl"
    violations: list[str] = []
    calls: dict[str, int] = {}
    probe_calls = 0
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            violations.append("invalid_jsonl")
            continue
        payload = json.dumps(row, ensure_ascii=False)
        if "webSearchToolCall" in payload or "webFetchToolCall" in payload:
            violations.append("web_tool")

        for rule_path in cursor_rule_paths(row):
            if not inside(rule_path, workspace):
                violations.append(f"outside_rule:{rule_path}")
        if row.get("type") != "tool_call":
            continue
        tool_call = row.get("tool_call", {})
        kind = next(iter(tool_call), "unknown")
        body = tool_call.get(kind, {})
        if kind in CURSOR_OPEN_WORLD_TOOLS:
            violations.append(f"open_world_tool:{kind}")
        elif kind not in CURSOR_LOCAL_TOOLS:
            violations.append(f"unknown_tool:{kind}")
        if row.get("subtype") == "completed" and kind == "mcpToolCall":
            detail = body.get("args", {})
            if (
                detail.get("serverIdentifier") == "sae_probe"
                and detail.get("toolName") == "probe_sae"
                and "success" in body.get("result", {})
            ):
                probe_calls += 1
        if row.get("subtype") != "started":
            continue
        calls[kind] = calls.get(kind, 0) + 1
        if kind == "mcpToolCall":
            detail = body.get("args", {})
            if detail.get("serverIdentifier") != "sae_probe" or detail.get("toolName") != "probe_sae":
                violations.append("non_probe_mcp")
        elif kind == "getMcpToolsToolCall":
            if body.get("args", {}).get("server") != "sae_probe":
                violations.append("non_probe_mcp")
        elif kind in {"readToolCall", "writeToolCall", "editToolCall", "deleteToolCall", "grepToolCall"}:
            path = body.get("args", {}).get("path")
            if path and not inside(path, workspace):
                violations.append(f"outside_path:{path}")
        elif kind == "shellToolCall":
            if cursor_network_command(body):
                violations.append("network_shell_command")
            if cursor_process_inspection(body):
                violations.append("process_or_env_inspection")
    return {
        "trace": str(trace),
        "tool_calls": calls,
        "probe_calls": probe_calls,
        "violations": sorted(set(violations)),
    }


def codex_audit(run_root: Path) -> dict[str, Any]:
    trace = run_root / "logs" / "agent.jsonl"
    stderr = run_root / "logs" / "agent.stderr.log"
    violations: list[str] = []
    probe_calls = 0
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            violations.append("invalid_jsonl")
            continue
        item = row.get("item", {})
        kind = item.get("type")
        if kind is not None and kind not in CODEX_ITEM_TYPES:
            violations.append(f"unknown_tool:{kind}")
        if kind in {"web_search", "web_search_call"}:
            violations.append("web_tool")
        if kind == "mcp_tool_call":
            if item.get("server") == "sae_probe" and item.get("tool") == "probe_sae":
                if (
                    row.get("type") == "item.completed"
                    and item.get("status") == "completed"
                    and item.get("error") is None
                    and item.get("result") is not None
                ):
                    probe_calls += 1
            else:
                violations.append("non_probe_mcp")
        if kind == "command_execution" and NETWORK_PATTERN.search(item.get("command", "")):
            violations.append("network_shell_command")
    if "isolation preflight passed" not in stderr.read_text(encoding="utf-8"):
        violations.append("missing_isolation_preflight")
    return {
        "trace": str(trace),
        "probe_calls": probe_calls,
        "violations": sorted(set(violations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed agent traces for benchmark isolation.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for run_root in sorted(args.runs_root.iterdir()):
        result_path = run_root / "result.json"
        metadata_path = run_root / "run.json"
        if not result_path.exists() or not metadata_path.exists():
            continue
        result = read_json(result_path)
        if result.get("status") != "submitted":
            continue
        metadata = read_json(metadata_path)
        harness = metadata.get("harness")
        if harness == "cursor":
            audit = cursor_audit(run_root)
        elif harness == "codex":
            audit = codex_audit(run_root)
        else:
            continue
        if not task_snapshot_matches(run_root, metadata):
            audit["violations"].append("stale_task_snapshot")
        audit["violations"] = sorted(set(audit["violations"]))
        audit.update(
            {
                "run_id": metadata["run_id"],
                "harness": harness,
                "model": metadata["agent_model"],
                "eligible": not audit["violations"] and audit["probe_calls"] > 0,
            }
        )
        rows.append(audit)
    payload = {
        "schema": 1,
        "audited": len(rows),
        "eligible": sum(row["eligible"] for row in rows),
        "ineligible": sum(not row["eligible"] for row in rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("audited", "eligible", "ineligible")}))


if __name__ == "__main__":
    main()

