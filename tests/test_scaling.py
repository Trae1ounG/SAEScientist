import unittest

from sae_scientist.scaling import (
    merge_discovery_batches,
    merge_validation_batches,
    rank_steering_screen,
    select_unique_activation_candidates,
)


def batch(concept_id: str = "alpha", checkpoint: str = "official checkpoint"):
    return {
        "schema": 1,
        "checkpoint": checkpoint,
        "concepts": [
            {
                "id": concept_id,
                "target": "a precise target",
                "positive": [f"positive {index}" for index in range(6)],
                "negative": [f"negative {index}" for index in range(6)],
            }
        ],
    }


class ScalingTest(unittest.TestCase):
    def test_merges_valid_independent_batches(self):
        merged = merge_discovery_batches([batch("alpha"), batch("beta")])
        self.assertEqual([row["id"] for row in merged["concepts"]], ["alpha", "beta"])

    def test_rejects_existing_concept_id(self):
        with self.assertRaisesRegex(ValueError, "excluded"):
            merge_discovery_batches([batch("alpha")], {"alpha"})

    def test_rejects_wrong_example_count(self):
        payload = batch()
        payload["concepts"][0]["positive"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 6 positives"):
            merge_discovery_batches([payload])

    def test_rejects_checkpoint_mismatch(self):
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            merge_discovery_batches([batch("alpha", "one"), batch("beta", "two")])

    def test_merges_complete_validation_batches(self):
        def validation(concept_id):
            cases = []
            for label, count in (("positive", 8), ("hard_negative", 8), ("neutral", 4)):
                cases.extend(
                    {"id": f"{concept_id}-{label}-{index}", "label": label, "text": f"{concept_id} {label} {index}"}
                    for index in range(count)
                )
            return {
                "schema": 1,
                "checkpoint": "official checkpoint",
                "steering_calibration": [
                    {"id": f"c{index}", "prompt": f"prompt {index}"} for index in range(5)
                ],
                "steering_prompts": "prompts.json",
                "concepts": [{
                    "id": concept_id,
                    "target": "target",
                    "steering_target": {
                        "concept": "target",
                        "cues": ["cue"],
                        "strong_evidence": "strong",
                        "insufficient_evidence": "weak",
                    },
                    "activation_cases": cases,
                }],
            }

        merged = merge_validation_batches(
            [validation("alpha"), validation("beta")], {"alpha", "beta"}
        )
        self.assertEqual(len(merged["concepts"]), 2)

    def test_rejects_incomplete_validation_cases(self):
        payload = {
            "schema": 1,
            "checkpoint": "official checkpoint",
            "steering_calibration": [{"id": f"c{i}", "prompt": "p"} for i in range(5)],
            "steering_prompts": "prompts.json",
            "concepts": [{
                "id": "alpha",
                "steering_target": {
                    "concept": "target",
                    "cues": ["cue"],
                    "strong_evidence": "strong",
                    "insufficient_evidence": "weak",
                },
                "activation_cases": [],
            }],
        }
        with self.assertRaisesRegex(ValueError, "invalid activation case counts"):
            merge_validation_batches([payload], {"alpha"})

    def test_selects_stable_unique_feature_per_concept(self):
        def candidate(feature_id, stable, rank):
            return {
                "feature_id": feature_id,
                "activation_stable": stable,
                "activation_auroc": 1.0,
                "positive_active_rate": 1.0,
                "hard_negative_to_positive_ratio": 0.1,
                "positive_mean_rank": rank,
            }

        concepts = [
            {"id": "alpha", "candidates": [candidate(10, True, 1)]},
            {
                "id": "beta",
                "candidates": [candidate(10, True, 2), candidate(11, True, 3)],
            },
            {"id": "gamma", "candidates": [candidate(12, False, 1)]},
        ]
        selected = select_unique_activation_candidates(concepts)
        self.assertEqual([row["feature_id"] for row in selected], [10, 11])

    def test_selects_multiple_stable_features_per_concept(self):
        candidates = [
            {
                "feature_id": feature_id,
                "activation_stable": True,
                "activation_auroc": 1.0,
                "positive_active_rate": 1.0,
                "hard_negative_to_positive_ratio": 0.0,
                "positive_mean_rank": rank,
            }
            for rank, feature_id in enumerate((10, 11, 12), start=1)
        ]
        selected = select_unique_activation_candidates(
            [{"id": "alpha", "candidates": candidates}], {10}, max_per_concept=2
        )
        self.assertEqual([row["feature_id"] for row in selected], [11, 12])

    def test_ranks_screen_passes_by_worst_control_margin(self):
        def result(concept_id, feature_id, feature, baseline, random):
            return {
                "suite": {"id": concept_id},
                "feature": {"feature_id": feature_id},
                "steering": {
                    "selected_alpha": 60,
                    "summary": {
                        "feature_target_score": feature,
                        "baseline_target_score": baseline,
                        "random_target_score": random,
                        "feature_success_rate": 0.75,
                        "nondegenerate_rate": 1.0,
                        "rerun_agreement": 1.0,
                    },
                },
            }

        protocol = {
            "min_target_delta_over_baseline": 0.15,
            "min_target_delta_over_random": 0.15,
            "min_target_success_rate": 0.5,
            "min_nondegenerate_rate": 0.875,
            "min_rerun_agreement": 0.8,
        }
        ranked = rank_steering_screen(
            [result("weaker", 1, 0.6, 0.1, 0.3), result("stronger", 2, 0.8, 0.1, 0.2)],
            protocol,
        )
        self.assertEqual([row["concept_id"] for row in ranked], ["stronger", "weaker"])
        self.assertTrue(all(row["screen_pass"] for row in ranked))

    def test_screen_ranking_prefers_explicit_concept_id(self):
        result = {
            "suite": {"id": "public_label_v1", "concept_id": "manifest_concept"},
            "feature": {"feature_id": 3},
            "steering": {
                "selected_alpha": 30,
                "summary": {
                    "feature_target_score": 1.0,
                    "baseline_target_score": 0.0,
                    "random_target_score": 0.0,
                    "feature_success_rate": 1.0,
                    "nondegenerate_rate": 1.0,
                    "rerun_agreement": 1.0,
                },
            },
        }
        protocol = {
            "min_target_delta_over_baseline": 0.15,
            "min_target_delta_over_random": 0.15,
            "min_target_success_rate": 0.5,
            "min_nondegenerate_rate": 0.875,
            "min_rerun_agreement": 0.8,
        }
        self.assertEqual(rank_steering_screen([result], protocol)[0]["concept_id"], "manifest_concept")


if __name__ == "__main__":
    unittest.main()
