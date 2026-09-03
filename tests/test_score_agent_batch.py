from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "score_agent_batch", ROOT / "scripts" / "score_agent_batch.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ScoreAgentBatchTest(unittest.TestCase):
    def test_parses_layer_probe_mapping(self) -> None:
        self.assertEqual(
            MODULE.parse_probe_urls(["9=http://one", "20=http://two"]),
            {9: "http://one", 20: "http://two"},
        )

    def test_reads_layer(self) -> None:
        task = {"sae": {"hook": "blocks.20.hook_resid_post"}}
        self.assertEqual(MODULE.task_layer(task), 20)

    def test_rejects_malformed_probe_mapping(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_probe_urls(["layer-nine"])

    def test_primary_precedes_later_eligible_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = "gemma-cat-001-offline-cursor-high"
            for suffix in ("01", "retry-01"):
                path = root / f"{base}-{suffix}"
                path.mkdir()
                (path / "result.json").write_text(
                    json.dumps({"status": "submitted"}), encoding="utf-8"
                )
            selected = MODULE.submitted_run("gemma_cat_001", "cursor-high", root)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], f"{base}-01")

    def test_retry_order_is_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = "gemma-cat-001-offline-cursor-high"
            for suffix in ("retry-2", "retry-10"):
                path = root / f"{base}-{suffix}"
                path.mkdir()
                (path / "result.json").write_text(
                    json.dumps({"status": "submitted"}), encoding="utf-8"
                )
            selected = MODULE.submitted_run("gemma_cat_001", "cursor-high", root)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], f"{base}-retry-2")

    def test_first_eligible_retry_is_selected_when_primary_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = "gemma-cat-001-offline-cursor-high"
            for suffix in ("01", "retry-01", "retry-02"):
                path = root / f"{base}-{suffix}"
                path.mkdir()
                (path / "result.json").write_text(
                    json.dumps({"status": "submitted"}), encoding="utf-8"
                )
            selected = MODULE.submitted_run(
                "gemma_cat_001",
                "cursor-high",
                root,
                {f"{base}-retry-01", f"{base}-retry-02"},
            )
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], f"{base}-retry-01")


if __name__ == "__main__":
    unittest.main()

