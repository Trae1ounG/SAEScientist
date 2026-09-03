import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_replicates.py"
SPEC = importlib.util.spec_from_file_location("aggregate_replicates", SCRIPT)
aggregate_replicates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate_replicates)


def leaderboard(activation: float, exact: float, feature_ids: list[int]) -> dict:
    configuration = {
        "configuration": "codex/model (high)",
        "harness": "codex",
        "model": "model",
        "reasoning_effort": "high",
        "completed_tasks": 2,
        "macro_gt_normalized_activation": activation,
        "exact_match_rate": exact,
        "causal_steering_rate": exact,
        "usable_steering_rate": 0.0,
        "median_elapsed_seconds": 10.0,
    }
    runs = [
        {
            "task": f"task-{index}",
            "harness": "codex",
            "model": "model",
            "reasoning_effort": "high",
            "selected_feature_id": feature_id,
            "pe_target_relevance": float(index + 1),
            "pe_task_preservation": 3.0,
        }
        for index, feature_id in enumerate(feature_ids)
    ]
    return {
        "task_coverage": {"stable_benchmark_tasks": 2},
        "configurations": [configuration],
        "runs": runs,
    }


class AggregateReplicatesTest(unittest.TestCase):
    def test_aggregates_variance_and_feature_selection_agreement(self):
        payload = aggregate_replicates.aggregate(
            [
                ("run1", leaderboard(0.6, 0.0, [1, 2])),
                ("run2", leaderboard(0.8, 0.5, [1, 3])),
                ("run3", leaderboard(1.0, 1.0, [1, 2])),
            ]
        )
        row = payload["configurations"][0]
        self.assertAlmostEqual(row["metrics"]["macro_gt_normalized_activation"]["mean"], 0.8)
        self.assertAlmostEqual(row["metrics"]["macro_gt_normalized_activation"]["std"], 0.1632993162)
        self.assertAlmostEqual(row["metrics"]["exact_match_rate"]["mean"], 0.5)
        self.assertAlmostEqual(row["feature_id_pairwise_agreement"], 2 / 3)
        self.assertAlmostEqual(row["feature_id_all_replicates_same_rate"], 0.5)
        self.assertEqual(row["metrics"]["macro_pe_target_relevance"]["mean"], 1.5)

    def test_rejects_missing_configuration(self):
        second = leaderboard(0.8, 0.5, [1, 2])
        second["configurations"] = []
        with self.assertRaisesRegex(ValueError, "missing from a replicate"):
            aggregate_replicates.aggregate(
                [("run1", leaderboard(0.6, 0.0, [1, 2])), ("run2", second)]
            )


if __name__ == "__main__":
    unittest.main()
