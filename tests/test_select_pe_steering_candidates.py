import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_pe_steering_candidates.py"
SPEC = importlib.util.spec_from_file_location("select_pe_steering_candidates", SCRIPT)
select_pe_steering_candidates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(select_pe_steering_candidates)


PROTOCOL = {
    "min_target_delta_over_baseline": 0.15,
    "min_target_delta_over_random": 0.15,
    "min_target_success_rate": 0.5,
    "min_nondegenerate_rate": 0.875,
    "min_rerun_agreement": 0.8,
}


class SelectPESteeringCandidatesTest(unittest.TestCase):
    def test_causal_tier_uses_formal_causal_thresholds(self):
        document = {
            "steering_screen": PROTOCOL,
            "formal_steering_gate": {
                "causal_stable": {
                    "min_target_delta_over_baseline": 0.2,
                    "min_target_delta_over_random": 0.2,
                    "min_target_success_rate": 0.7,
                    "min_nondegenerate_rate": 0.5,
                    "min_rerun_agreement": 0.8,
                }
            },
        }
        selected = select_pe_steering_candidates.protocol_for_tier(
            document, "causal"
        )
        self.assertEqual(selected["min_nondegenerate_rate"], 0.5)
        self.assertEqual(selected["min_target_success_rate"], 0.7)

    def raw(self, concept_id, feature_id, selected_alpha):
        return {
            "feature": {"feature_id": feature_id},
            "suite": {"id": f"{concept_id}_v1", "concept_id": concept_id},
            "steering": {"selected_alpha": selected_alpha},
        }

    def summary(self, feature, baseline=0.1, random=0.1, quality_stable=False):
        return {
            "expected_rows": 8,
            "valid_rows": 8,
            "error_rows": 0,
            "result": "alpha_10.json",
            "admission_evaluation": {
                "baseline_target_score": baseline,
                "feature_target_score": feature,
                "random_target_score": random,
                "feature_success_rate": 0.75,
                "nondegenerate_rate": 1.0,
                "rerun_agreement": 1.0,
            },
            "quality": {"stable": quality_stable},
        }

    def test_selects_from_complete_pe_summaries_without_quality_stable_gate(self):
        manifest = {
            "stage": "steering_screen",
            "tasks": [
                {
                    "id": "alpha_10",
                    "concept_id": "alpha",
                    "feature": "artifacts/features/gemma2_9b_it_l9_w131k_feature_10.npz",
                },
                {
                    "id": "beta_20",
                    "concept_id": "beta",
                    "feature": "artifacts/features/gemma2_9b_it_l9_w131k_feature_20.npz",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_dir = tmp_path / "summaries"
            result_dir = tmp_path / "results"
            summary_dir.mkdir()
            result_dir.mkdir()
            alpha_summary = self.summary(0.7, quality_stable=False)
            (summary_dir / "alpha_10_summary.json").write_text(
                json.dumps(alpha_summary),
                encoding="utf-8",
            )
            beta_summary = self.summary(0.2, random=0.15, quality_stable=True)
            beta_summary["result"] = "beta_20.json"
            (summary_dir / "beta_20_summary.json").write_text(
                json.dumps(beta_summary),
                encoding="utf-8",
            )
            (result_dir / "alpha_10.json").write_text(
                json.dumps(self.raw("alpha", 10, 60.0)),
                encoding="utf-8",
            )
            (result_dir / "beta_20.json").write_text(
                json.dumps(self.raw("beta", 20, 45.0)),
                encoding="utf-8",
            )

            rows = select_pe_steering_candidates.select_rows(
                manifest, summary_dir, result_dir, PROTOCOL
            )

        self.assertEqual([row["concept_id"] for row in rows], ["alpha", "beta"])
        self.assertTrue(rows[0]["screen_pass"])
        self.assertFalse(rows[1]["screen_pass"])
        self.assertEqual(rows[0]["feature_id"], 10)
        self.assertEqual(rows[0]["selected_alpha"], 60)

    def test_rejects_incomplete_or_errored_summary(self):
        manifest = {
            "stage": "steering_screen",
            "tasks": [
                {
                    "id": "alpha_10",
                    "concept_id": "alpha",
                    "feature": "artifacts/features/gemma2_9b_it_l9_w131k_feature_10.npz",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_dir = tmp_path / "summaries"
            result_dir = tmp_path / "results"
            summary_dir.mkdir()
            result_dir.mkdir()
            broken = self.summary(0.7)
            broken["error_rows"] = 1
            (summary_dir / "alpha_10_summary.json").write_text(
                json.dumps(broken),
                encoding="utf-8",
            )
            (result_dir / "alpha_10.json").write_text(
                json.dumps(self.raw("alpha", 10, 60.0)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete or errored"):
                select_pe_steering_candidates.select_rows(
                    manifest, summary_dir, result_dir, PROTOCOL
                )

    def test_rejects_manifest_raw_mismatch(self):
        manifest = {
            "stage": "steering_screen",
            "tasks": [
                {
                    "id": "alpha_10",
                    "concept_id": "alpha",
                    "feature": "artifacts/features/gemma2_9b_it_l9_w131k_feature_10.npz",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_dir = tmp_path / "summaries"
            result_dir = tmp_path / "results"
            summary_dir.mkdir()
            result_dir.mkdir()
            (summary_dir / "alpha_10_summary.json").write_text(
                json.dumps(self.summary(0.7)),
                encoding="utf-8",
            )
            (result_dir / "alpha_10.json").write_text(
                json.dumps(self.raw("other", 10, 60.0)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "concept id mismatch"):
                select_pe_steering_candidates.select_rows(
                    manifest, summary_dir, result_dir, PROTOCOL
                )

    def test_incremental_selection_skips_missing_pairs(self):
        manifest = {
            "stage": "steering_screen",
            "tasks": [
                {
                    "id": "alpha_10",
                    "concept_id": "alpha",
                    "feature": "artifacts/features/gemma2_9b_it_l9_w131k_feature_10.npz",
                },
                {
                    "id": "beta_20",
                    "concept_id": "beta",
                    "feature": "artifacts/features/gemma2_9b_it_l9_w131k_feature_20.npz",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_dir = tmp_path / "summaries"
            result_dir = tmp_path / "results"
            summary_dir.mkdir()
            result_dir.mkdir()
            (summary_dir / "alpha_10_summary.json").write_text(
                json.dumps(self.summary(0.7)), encoding="utf-8"
            )
            (result_dir / "alpha_10.json").write_text(
                json.dumps(self.raw("alpha", 10, 60.0)), encoding="utf-8"
            )
            rows = select_pe_steering_candidates.select_rows(
                manifest,
                summary_dir,
                result_dir,
                PROTOCOL,
                allow_missing=True,
            )
        self.assertEqual([row["concept_id"] for row in rows], ["alpha"])


if __name__ == "__main__":
    unittest.main()

