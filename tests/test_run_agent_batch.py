import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_agent_batch.py"
SPEC = importlib.util.spec_from_file_location("run_agent_batch", SCRIPT)
run_agent_batch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_agent_batch)


def submitted(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "result.json").write_text(
        json.dumps({"status": "submitted"}), encoding="utf-8"
    )


def failed(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "result.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )


class RunAgentBatchTest(unittest.TestCase):
    def test_parses_layer_probe_mapping(self):
        self.assertEqual(
            run_agent_batch.parse_probe_urls(
                ["9=http://127.0.0.1:8765", "20=http://127.0.0.1:8766"]
            ),
            {9: "http://127.0.0.1:8765", 20: "http://127.0.0.1:8766"},
        )

    def test_rejects_malformed_probe_mapping(self):
        with self.assertRaisesRegex(ValueError, "LAYER=URL"):
            run_agent_batch.parse_probe_urls(["http://127.0.0.1:8765"])

    def test_reads_layer_from_task_hook(self):
        task = {"sae": {"hook": "blocks.20.hook_resid_post"}}
        self.assertEqual(run_agent_batch.task_layer(task), 20)

    def test_failed_primary_uses_append_only_retry_id(self):
        original_root = run_agent_batch.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            run_agent_batch.ROOT = Path(tmp)
            primary = (
                run_agent_batch.ROOT
                / "runs"
                / "gemma-cat-001-offline-cursor-high-01"
            )
            failed(primary)
            try:
                run_id, state = run_agent_batch.select_run_id(
                    "gemma_cat_001", "cursor-high", True, 1
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual(run_id, "gemma-cat-001-offline-cursor-high-retry-01")
        self.assertEqual(state, "new")

    def test_second_replicate_uses_distinct_run_id(self):
        original_root = run_agent_batch.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            run_agent_batch.ROOT = Path(tmp)
            try:
                run_id, state = run_agent_batch.select_run_id(
                    "gemma_cat_001",
                    "cursor-high",
                    True,
                    1,
                    replicate_index=2,
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual(
            run_id,
            "gemma-cat-001-offline-cursor-high-rep-02-01",
        )
        self.assertEqual(state, "new")

    def test_first_replicate_preserves_existing_run_id(self):
        original_root = run_agent_batch.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            run_agent_batch.ROOT = Path(tmp)
            try:
                run_id, state = run_agent_batch.select_run_id(
                    "gemma_cat_001",
                    "cursor-high",
                    True,
                    1,
                    replicate_index=1,
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual(run_id, "gemma-cat-001-offline-cursor-high-01")
        self.assertEqual(state, "new")

    def test_active_primary_blocks_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            original_root = run_agent_batch.ROOT
            run_agent_batch.ROOT = Path(directory)
            try:
                (Path(directory) / "runs" / "task-offline-model-01").mkdir(
                    parents=True
                )
                run_id, state = run_agent_batch.select_run_id(
                    "task", "model", True, 2
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual((run_id, state), ("task-offline-model-01", "incomplete"))

    def test_submitted_primary_absent_from_audit_awaits_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            original_root = run_agent_batch.ROOT
            run_agent_batch.ROOT = Path(directory)
            try:
                primary_id = "task-offline-model-01"
                submitted(Path(directory) / "runs" / primary_id)
                run_id, state = run_agent_batch.select_run_id(
                    "task", "model", True, 2, set(), set()
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual((run_id, state), (primary_id, "incomplete"))

    def test_submitted_retry_absent_from_audit_awaits_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            original_root = run_agent_batch.ROOT
            run_agent_batch.ROOT = Path(directory)
            try:
                primary_id = "task-offline-model-01"
                retry_id = "task-offline-model-retry-01"
                submitted(Path(directory) / "runs" / primary_id)
                submitted(Path(directory) / "runs" / retry_id)
                run_id, state = run_agent_batch.select_run_id(
                    "task", "model", True, 2, set(), {primary_id}
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual((run_id, state), (retry_id, "incomplete"))

    def test_audited_ineligible_primary_advances_to_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            original_root = run_agent_batch.ROOT
            run_agent_batch.ROOT = Path(directory)
            try:
                primary_id = "task-offline-model-01"
                submitted(Path(directory) / "runs" / primary_id)
                run_id, state = run_agent_batch.select_run_id(
                    "task", "model", True, 2, set(), {primary_id}
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual((run_id, state), ("task-offline-model-retry-01", "new"))

    def test_audit_accepts_eligible_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            original_root = run_agent_batch.ROOT
            run_agent_batch.ROOT = Path(directory)
            try:
                submitted(Path(directory) / "runs" / "task-offline-model-01")
                submitted(Path(directory) / "runs" / "task-offline-model-retry-01")
                run_id, state = run_agent_batch.select_run_id(
                    "task",
                    "model",
                    True,
                    2,
                    {"task-offline-model-retry-01"},
                    {
                        "task-offline-model-01",
                        "task-offline-model-retry-01",
                    },
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual((run_id, state), ("task-offline-model-retry-01", "complete"))

    def test_failed_retry_advances_to_next_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            original_root = run_agent_batch.ROOT
            run_agent_batch.ROOT = Path(directory)
            try:
                primary_id = "task-offline-model-01"
                submitted(Path(directory) / "runs" / primary_id)
                retry = Path(directory) / "runs" / "task-offline-model-retry-01"
                failed(retry)
                run_id, state = run_agent_batch.select_run_id(
                    "task", "model", True, 2, set(), {primary_id}
                )
            finally:
                run_agent_batch.ROOT = original_root
        self.assertEqual((run_id, state), ("task-offline-model-retry-02", "new"))


if __name__ == "__main__":
    unittest.main()

