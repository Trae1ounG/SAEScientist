import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "judge_feature_steering.py"
SPEC = importlib.util.spec_from_file_location("judge_feature_steering", SCRIPT)
judge_feature_steering = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge_feature_steering)


class JudgeFeatureSteeringTest(unittest.TestCase):
    def test_resume_keeps_only_valid_expected_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"case_id": "a", "repeat": 0, "ratings": {}}),
                        json.dumps({"case_id": "a", "repeat": 1, "error": "failed"}),
                        json.dumps({"case_id": "other", "repeat": 0, "ratings": {}}),
                        "not json",
                    ]
                ),
                encoding="utf-8",
            )

            rows = judge_feature_steering.load_resumable_rows(
                path, {("a", 0), ("a", 1)}
            )

        self.assertEqual(set(rows), {("a", 0)})

    def test_azure_openai_client_uses_named_environment(self):
        fake_module = type("OpenAIModule", (), {})()
        fake_module.AzureOpenAI = mock.Mock(return_value="client")
        env = {
            "TEST_KEY": "secret",
            "TEST_ENDPOINT": "https://example.invalid",
            "TEST_VERSION": "2024-02-01",
        }
        with mock.patch.dict("os.environ", env, clear=True), mock.patch.dict(
            "sys.modules", {"openai": fake_module}
        ):
            client = judge_feature_steering.create_judge_client(
                "azure-openai",
                model_name="gpt-4o-2024-11-20",
                api_key_env="TEST_KEY",
                azure_endpoint_env="TEST_ENDPOINT",
                azure_api_version_env="TEST_VERSION",
            )

        self.assertEqual(client, "client")
        fake_module.AzureOpenAI.assert_called_once_with(
            api_key="secret",
            azure_endpoint="https://example.invalid",
            api_version="2024-02-01",
        )

    def test_screen_uses_evaluated_cases_not_full_prompt_bank(self):
        result = {"steering": {"cases": [{"id": "a"}, {"id": "b"}]}}
        self.assertEqual(judge_feature_steering.evaluated_case_count(result), 2)

    def test_single_repeat_uses_generation_rerun_agreement(self):
        self.assertEqual(
            judge_feature_steering.effective_rerun_agreement(1, [], 0.8), 0.8
        )

    def test_formal_repeats_use_judge_agreement(self):
        self.assertEqual(
            judge_feature_steering.effective_rerun_agreement(
                2, [True, False, True, True], 0.0
            ),
            0.75,
        )

    def test_screen_thresholds_match_frozen_screen_gate(self):
        thresholds = judge_feature_steering.thresholds_for_stage("steering_screen")
        self.assertEqual(thresholds.min_heldout_prompts, 8)
        self.assertEqual(thresholds.min_target_delta_over_baseline, 0.15)
        self.assertEqual(thresholds.min_success_rate, 0.5)
        self.assertEqual(thresholds.min_usable_target_rate, 0.0)
        self.assertEqual(thresholds.min_nondegenerate_rate, 0.875)

    def test_formal_causal_tier_allows_topic_shift_but_not_total_collapse(self):
        thresholds = judge_feature_steering.causal_thresholds_for_stage(
            "formal_steering"
        )
        self.assertEqual(thresholds.min_usable_target_rate, 0.0)
        self.assertEqual(thresholds.min_nondegenerate_rate, 0.5)
        self.assertEqual(thresholds.min_success_rate, 0.7)


if __name__ == "__main__":
    unittest.main()

