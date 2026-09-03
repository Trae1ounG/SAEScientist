import unittest

from sae_bench.admission import activation_failures, admission_failures, is_admitted
from sae_bench.sources import (
    QWEN3_8B_BASE_L0_50,
    require_official_source,
    validate_checkpoint_state,
)


def passing_result():
    source = QWEN3_8B_BASE_L0_50
    return {
        "source": {
            "publisher": source.publisher,
            "repo": source.repo,
            "base_model": source.base_model,
            "resolved_revision": "0123456789abcdef",
        },
        "evaluation": {
            "heldout_prompts": 20,
            "scorer": "pe_judge_v1",
            "baseline_target_score": 0.10,
            "feature_target_score": 0.80,
            "random_target_score": 0.20,
            "feature_success_rate": 0.75,
            "usable_target_rate": 0.70,
            "nondegenerate_rate": 0.95,
            "rerun_agreement": 0.85,
        },
    }


class AdmissionTest(unittest.TestCase):
    def test_official_source_is_allowlisted(self):
        self.assertEqual(
            require_official_source(QWEN3_8B_BASE_L0_50.repo), QWEN3_8B_BASE_L0_50
        )

    def test_community_source_is_rejected(self):
        result = passing_result()
        result["source"]["repo"] = "sammyliu/qwen3-8b-sae-l36-topk64"
        self.assertFalse(is_admitted(result))
        self.assertIn("official allowlist", " ".join(admission_failures(result)))

    def test_changed_output_without_target_effect_is_rejected(self):
        result = passing_result()
        result["evaluation"]["feature_target_score"] = 0.25
        self.assertFalse(is_admitted(result))
        self.assertIn("target effect", " ".join(admission_failures(result)))

    def test_passing_official_result_is_admitted(self):
        self.assertTrue(is_admitted(passing_result()))

    def test_low_usable_target_rate_is_rejected_when_judged(self):
        result = passing_result()
        result["evaluation"]["usable_target_rate"] = 0.45
        self.assertFalse(is_admitted(result))
        self.assertIn("usable target rate", " ".join(admission_failures(result)))

    def test_activation_ratio_boundary_is_explicit(self):
        summary = {
            "auroc": 0.99,
            "positive_active_rate": 1.0,
            "hard_negative_to_positive_ratio": 0.297,
        }
        self.assertEqual(activation_failures(summary), [])
        summary["hard_negative_to_positive_ratio"] = 0.301
        self.assertIn("0.30", " ".join(activation_failures(summary)))

    def test_activation_auroc_must_reach_frozen_scaling_gate(self):
        summary = {
            "auroc": 0.949,
            "positive_active_rate": 1.0,
            "hard_negative_to_positive_ratio": 0.1,
        }
        self.assertIn("0.95", " ".join(activation_failures(summary)))

    def test_official_tensor_layout_is_checked(self):
        class TensorShape:
            def __init__(self, shape):
                self.shape = shape

        source = QWEN3_8B_BASE_L0_50
        state = {
            "W_enc": TensorShape((source.d_sae, source.d_model)),
            "W_dec": TensorShape((source.d_model, source.d_sae)),
            "b_enc": TensorShape((source.d_sae,)),
            "b_dec": TensorShape((source.d_model,)),
        }
        validate_checkpoint_state(state, source)
        state["W_dec"] = TensorShape((source.d_sae, source.d_model))
        with self.assertRaises(ValueError):
            validate_checkpoint_state(state, source)


if __name__ == "__main__":
    unittest.main()

