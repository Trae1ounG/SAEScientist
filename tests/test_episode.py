import json
from pathlib import Path
import tempfile
import unittest

from sae_bench.episode import (
    initialize_run,
    read_submission,
    recover_submission_from_trace,
)


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = self.root / "task.json"
        self.prompt = self.root / "prompt.md"
        self.task.write_text(
            json.dumps({"schema": 1, "task_id": "case", "sae": {"feature_count": 8}})
        )
        self.prompt.write_text("Find the feature.\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_run_is_created_once(self):
        layout = initialize_run(
            runs_root=self.root / "runs",
            run_id="run-1",
            task_path=self.task,
            prompt_path=self.prompt,
            harness="codex",
            agent_model="model",
            source_commit="commit",
            reasoning_effort="high",
        )
        self.assertTrue((layout.root / "run.json").is_file())
        self.assertTrue((layout.workspace / "task.json").is_file())
        manifest = json.loads((layout.root / "run.json").read_text())
        self.assertEqual(manifest["network"], "disabled")
        self.assertEqual(manifest["reasoning_effort"], "high")
        with self.assertRaises(FileExistsError):
            initialize_run(
                runs_root=self.root / "runs",
                run_id="run-1",
                task_path=self.task,
                prompt_path=self.prompt,
                harness="codex",
                agent_model="model",
                source_commit="commit",
            )

    def test_submission_is_minimal_and_in_range(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        (workspace / "submission.json").write_text('{"feature_id": 3}\n')
        self.assertEqual(read_submission(workspace, 8), {"feature_id": 3})
        (workspace / "submission.json").write_text(
            '{"feature_id": 3, "description": "extra"}\n'
        )
        with self.assertRaises(ValueError):
            read_submission(workspace, 8)

    def test_submission_can_be_recovered_from_cursor_final_answer(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        trace = self.root / "agent.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Evidence omitted. Final:\n```json\n"
                                '{"feature_id": 3}\n```',
                            }
                        ]
                    },
                }
            )
            + "\n"
        )
        self.assertEqual(
            recover_submission_from_trace(trace, workspace, 8), {"feature_id": 3}
        )
        self.assertEqual(read_submission(workspace, 8), {"feature_id": 3})

    def test_trace_fallback_rejects_ambiguous_feature_ids(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        trace = self.root / "agent.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"feature_id": 2} or {"feature_id": 3}',
                            }
                        ]
                    },
                }
            )
            + "\n"
        )
        with self.assertRaises(ValueError):
            recover_submission_from_trace(trace, workspace, 8)


if __name__ == "__main__":
    unittest.main()

