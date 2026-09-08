from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_agent_steering", ROOT / "scripts" / "prepare_agent_steering.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareAgentSteeringTest(unittest.TestCase):
    def test_expert_centered_alpha_grid(self) -> None:
        self.assertEqual(MODULE.alpha_grid(120), [60.0, 90.0, 120.0, 150.0, 180.0])
        self.assertEqual(MODULE.fallback_alpha_grid(160), [10.0, 20.0, 40.0, 60.0])

    def test_layer_paths_and_hook(self) -> None:
        self.assertEqual(MODULE.parse_layer_paths(["9=/a", "20=/b"]), {9: Path("/a"), 20: Path("/b")})
        self.assertEqual(
            MODULE.task_layer({"sae": {"hook": "blocks.20.hook_resid_post"}}),
            20,
        )

    def test_feature_path_respects_requested_directory(self) -> None:
        self.assertEqual(
            MODULE.feature_path(Path("custom/features"), 9, 12),
            "custom/features/gemma2_9b_it_l9_w131k_feature_12.npz",
        )

    def test_reusable_candidate_ids_ignore_summary(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task__candidate_7.json").write_text(
                json.dumps({"feature": {"layer": 9}, "steering": {}})
            )
            (root / "summary.json").write_text(json.dumps({"tasks": []}))
            self.assertEqual(
                MODULE.reusable_candidate_ids([root]), {"task__candidate_7"}
            )

    def test_candidates_use_summary_allowlist_and_trace_audit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores"
            scores.mkdir()
            selected = scores / "selected.json"
            ineligible = scores / "ineligible.json"
            stale = scores / "stale.json"
            selected.write_text(
                json.dumps(
                    {
                        "run_id": "selected",
                        "task": "tasks/example.json",
                        "feature_id": 7,
                        "exact_match": False,
                    }
                )
            )
            ineligible.write_text(
                json.dumps(
                    {
                        "run_id": "ineligible",
                        "task": "tasks/example.json",
                        "feature_id": 9,
                        "exact_match": False,
                    }
                )
            )
            stale.write_text(
                json.dumps(
                    {
                        "run_id": "stale",
                        "task": "tasks/example.json",
                        "feature_id": 8,
                        "exact_match": False,
                    }
                )
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "eligible_runs": 2,
                        "runs": [
                            {
                                "run_id": "selected",
                                "status": "scored",
                                "output": str(selected),
                            },
                            {
                                "run_id": "ineligible",
                                "status": "skipped",
                                "output": str(ineligible),
                            },
                        ],
                    }
                )
            )
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "runs": [
                            {"run_id": "selected", "eligible": True},
                            {"run_id": "ineligible", "eligible": False},
                            {"run_id": "stale", "eligible": True},
                        ]
                    }
                )
            )

            candidates = MODULE.candidate_scores(
                summary,
                scores,
                audit,
                {"tasks/example.json": {}},
            )

        self.assertEqual(set(candidates), {("tasks/example.json", 7)})

    def test_activation_summary_rejects_incomplete_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores"
            scores.mkdir()
            selected = scores / "selected.json"
            selected.write_text(json.dumps({"run_id": "selected"}))
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "eligible_runs": 1,
                        "runs": [
                            {
                                "run_id": "selected",
                                "status": "failed",
                                "output": str(selected),
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE.activation_score_paths(summary, scores)

    def test_extracted_sidecar_contains_checkpoint_provenance_only(self) -> None:
        source = {
            "publisher": "Google DeepMind",
            "official_source": True,
            "repo": "google/example",
            "resolved_revision": "rev",
            "base_model": "google/model",
            "checkpoint": "layer/params.npz",
            "hookpoint": "resid_post",
            "layer": 9,
            "feature_id": 7,
            "label": "secret semantic label",
            "label_source": "external",
            "neuronpedia": "https://example.test",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            params = root / "params.npz"
            np.savez(
                params,
                W_dec=np.zeros((2, 3)),
                W_enc=np.zeros((3, 2)),
                b_dec=np.zeros(3),
                b_enc=np.zeros(2),
                threshold=np.zeros(2),
            )
            output = root / "features"
            MODULE.extract_features(
                params,
                [{"feature_id": 1, "layer": 9, "source": source}],
                output,
            )
            sidecar = json.loads(
                (output / "gemma2_9b_it_l9_w131k_feature_1.json").read_text()
            )

        self.assertEqual(set(sidecar), set(MODULE.CHECKPOINT_PROVENANCE_FIELDS))
        self.assertNotIn("feature_id", sidecar)
        self.assertNotIn("label", sidecar)


if __name__ == "__main__":
    unittest.main()
