import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_agent_runs.py"
SPEC = importlib.util.spec_from_file_location("audit_agent_runs", SCRIPT)
audit_agent_runs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_agent_runs)


class AuditAgentRunsTest(unittest.TestCase):
    def test_optional_run_id_filter(self):
        pattern = audit_agent_runs.re.compile(r"-rep-03-(?:retry-)?\d+$")
        self.assertTrue(
            audit_agent_runs.matches_run_id(
                "gemma-cat-001-offline-codex-sol-high-rep-03-01", pattern
            )
        )
        self.assertFalse(
            audit_agent_runs.matches_run_id(
                "gemma-cat-001-offline-codex-sol-high-rep-04-01", pattern
            )
        )
        self.assertTrue(audit_agent_runs.matches_run_id("anything", None))

    def test_python_variable_named_nc_is_not_network(self):
        body = {
            "args": {
                "command": "python3 - <<'PY'\nnc = 3\nprint(nc)\nPY",
                "parsingResult": {"executableCommands": [{"name": "python3"}]},
            }
        }
        self.assertFalse(audit_agent_runs.cursor_network_command(body))

    def test_network_executable_is_detected(self):
        body = {
            "args": {
                "command": "curl example.com",
                "parsingResult": {"executableCommands": [{"name": "curl"}]},
            }
        }
        self.assertTrue(audit_agent_runs.cursor_network_command(body))

    def test_url_inside_python_is_detected(self):
        body = {
            "args": {
                "command": "python3 -c 'print(\"https://example.com\")'",
                "parsingResult": {"executableCommands": [{"name": "python3"}]},
            }
        }
        self.assertTrue(audit_agent_runs.cursor_network_command(body))

    def test_process_and_environment_inspection_are_detected(self):
        for executable in ("env", "printenv", "ps"):
            body = {
                "args": {
                    "command": executable,
                    "parsingResult": {"executableCommands": [{"name": executable}]},
                }
            }
            self.assertTrue(audit_agent_runs.cursor_process_inspection(body))

    def test_ordinary_python_is_not_process_inspection(self):
        body = {
            "args": {
                "command": "python3 -c 'print(1)'",
                "parsingResult": {"executableCommands": [{"name": "python3"}]},
            }
        }
        self.assertFalse(audit_agent_runs.cursor_process_inspection(body))

    def test_relative_paths_are_resolved_from_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            self.assertTrue(audit_agent_runs.inside("submission.json", workspace))
            self.assertFalse(audit_agent_runs.inside("../outside.json", workspace))

    def test_task_snapshot_must_match_current_task(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "workspace").mkdir()
            current = run_root / "current.json"
            frozen = run_root / "workspace" / "task.json"
            current.write_text('{"feature_count": 8}\n', encoding="utf-8")
            frozen.write_text('{"feature_count": 9}\n', encoding="utf-8")
            original = audit_agent_runs.current_task_path
            audit_agent_runs.current_task_path = lambda _metadata: current
            try:
                self.assertFalse(
                    audit_agent_runs.task_snapshot_matches(run_root, {"task_id": "x"})
                )
                frozen.write_text('{"feature_count": 8}\n', encoding="utf-8")
                self.assertTrue(
                    audit_agent_runs.task_snapshot_matches(run_root, {"task_id": "x"})
                )
            finally:
                audit_agent_runs.current_task_path = original

    def test_cursor_requires_successful_completed_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "workspace").mkdir()
            (run_root / "logs").mkdir()
            started = {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {
                    "mcpToolCall": {
                        "args": {
                            "serverIdentifier": "sae_probe",
                            "toolName": "probe_sae",
                        }
                    }
                },
            }
            completed = {
                **started,
                "subtype": "completed",
                "tool_call": {
                    "mcpToolCall": {
                        **started["tool_call"]["mcpToolCall"],
                        "result": {"success": {"content": []}},
                    }
                },
            }
            trace = run_root / "logs" / "agent.jsonl"
            trace.write_text(json.dumps(started) + "\n", encoding="utf-8")
            self.assertEqual(audit_agent_runs.cursor_audit(run_root)["probe_calls"], 0)
            trace.write_text(
                json.dumps(started) + "\n" + json.dumps(completed) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(audit_agent_runs.cursor_audit(run_root)["probe_calls"], 1)

    def test_cursor_rejects_unknown_tools_and_structures_rule_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            workspace = run_root / "workspace"
            workspace.mkdir()
            (run_root / "logs").mkdir()
            rows = [
                {
                    "relatedCursorRulePaths": [
                        str(workspace / "local.mdc"),
                        "/outside/rule.mdc",
                    ]
                },
                {
                    "type": "tool_call",
                    "subtype": "started",
                    "tool_call": {"browserToolCall": {"args": {}}},
                },
                {
                    "type": "tool_call",
                    "subtype": "started",
                    "tool_call": {"taskToolCall": {"args": {}}},
                },
            ]
            (run_root / "logs" / "agent.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            violations = audit_agent_runs.cursor_audit(run_root)["violations"]
        self.assertIn("outside_rule:/outside/rule.mdc", violations)
        self.assertIn("unknown_tool:browserToolCall", violations)
        self.assertIn("open_world_tool:taskToolCall", violations)

    def test_codex_requires_successful_completed_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "logs").mkdir()
            (run_root / "logs" / "agent.stderr.log").write_text(
                "isolation preflight passed\n", encoding="utf-8"
            )
            started = {
                "type": "item.started",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "sae_probe",
                    "tool": "probe_sae",
                    "status": "in_progress",
                    "result": None,
                    "error": None,
                },
            }
            completed = {
                "type": "item.completed",
                "item": {
                    **started["item"],
                    "status": "completed",
                    "result": {"content": []},
                },
            }
            trace = run_root / "logs" / "agent.jsonl"
            trace.write_text(json.dumps(started) + "\n", encoding="utf-8")
            self.assertEqual(audit_agent_runs.codex_audit(run_root)["probe_calls"], 0)
            trace.write_text(
                json.dumps(started) + "\n" + json.dumps(completed) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(audit_agent_runs.codex_audit(run_root)["probe_calls"], 1)

    def test_codex_rejects_unknown_item_type(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "logs").mkdir()
            (run_root / "logs" / "agent.stderr.log").write_text(
                "isolation preflight passed\n", encoding="utf-8"
            )
            (run_root / "logs" / "agent.jsonl").write_text(
                json.dumps({"type": "item.started", "item": {"type": "browser"}})
                + "\n",
                encoding="utf-8",
            )
            violations = audit_agent_runs.codex_audit(run_root)["violations"]
        self.assertIn("unknown_tool:browser", violations)


if __name__ == "__main__":
    unittest.main()
