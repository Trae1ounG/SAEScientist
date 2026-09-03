import unittest

from sae_bench.scoring import (
    gt_normalized_metrics,
    spearman_correlation,
    summarize_rank_rows,
)


class ScoringTest(unittest.TestCase):
    def test_spearman_handles_ties(self):
        self.assertAlmostEqual(spearman_correlation([0, 1, 1, 3], [0, 2, 2, 4]), 1.0)

    def test_rank_summary_uses_inactive_rank(self):
        rows = [
            {"label": "positive", "rank": 1, "activation": 4},
            {"label": "positive", "rank": 3, "activation": 2},
            {"label": "hard_negative", "rank": 8, "activation": 0},
            {"label": "neutral", "rank": 8, "activation": 0},
        ]
        summary = summarize_rank_rows(rows, feature_count=8)
        self.assertEqual(summary["positive"]["mean_activation"], 3.0)
        self.assertEqual(summary["positive"]["mean_rank"], 2.0)
        self.assertEqual(summary["hard_negative"]["mean_rank"], 8.0)
        self.assertEqual(summary["activation_auroc"], 1.0)

    def test_gt_normalization_uses_expert_as_one(self):
        expert = {
            "positive": {"mean_rank": 10.0, "mean_activation": 8.0},
            "hard_negative": {"mean_activation": 2.0},
            "neutral": {"mean_activation": 0.0},
            "activation_auroc": 1.0,
        }
        self.assertEqual(
            gt_normalized_metrics(expert, expert, pattern_spearman=1.0)["mean_score"],
            1.0,
        )

    def test_gt_normalization_penalizes_nonselective_feature(self):
        expert = {
            "positive": {"mean_rank": 10.0, "mean_activation": 8.0},
            "hard_negative": {"mean_activation": 2.0},
            "neutral": {"mean_activation": 0.0},
            "activation_auroc": 1.0,
        }
        candidate = {
            "positive": {"mean_rank": 5.0, "mean_activation": 12.0},
            "hard_negative": {"mean_activation": 1.0},
            "neutral": {"mean_activation": 12.0},
            "activation_auroc": 0.5,
        }
        normalized = gt_normalized_metrics(candidate, expert, pattern_spearman=0.0)
        self.assertEqual(normalized["positive_rank_recovery"], 1.0)
        self.assertEqual(normalized["activation_contrast_recovery"], 0.0)
        self.assertEqual(normalized["auroc_recovery"], 0.0)
        self.assertEqual(normalized["mean_score"], 0.25)


if __name__ == "__main__":
    unittest.main()

