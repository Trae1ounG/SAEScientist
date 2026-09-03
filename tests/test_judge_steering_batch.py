import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "judge_steering_batch.py"
SPEC = importlib.util.spec_from_file_location("judge_steering_batch", SCRIPT)
judge_steering_batch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge_steering_batch)


class JudgeSteeringBatchTest(unittest.TestCase):
    def write_complete_summary(self, path: Path, stable: bool = True) -> None:
        path.write_text(
            json.dumps(
                {
                    "expected_rows": 4,
                    "valid_rows": 4,
                    "error_rows": 0,
                    "admission_evaluation": {
                        "feature_target_score": 0.75,
                        "feature_success_rate": 1.0,
                        "usable_target_rate": 0.5,
                        "nondegenerate_rate": 1.0,
                        "rerun_agreement": 1.0,
                    },
                    "quality": {
                        "stable": stable,
                        "activation_failures": [],
                        "causal_failures": [],
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_skips_existing_complete_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = {"id": "alpha_10", "suite": "suite.json", "concept_id": "alpha"}
            summary = judge_steering_batch.task_summary_path(output_dir, task["id"])
            self.write_complete_summary(summary)

            with mock.patch.object(judge_steering_batch.subprocess, "run") as run:
                row = judge_steering_batch.run_task(
                    task,
                    result_dir=output_dir,
                    output_dir=output_dir,
                    suite=None,
                    provider="azure-openai",
                    model_name="judge-model",
                    api_key_env="API_KEY",
                    judge_workers=1,
                    repeats=2,
                    seed=0,
                )

            run.assert_not_called()
            self.assertEqual(row["status"], "skipped")
            self.assertTrue(row["stable"])
            self.assertEqual(row["feature_target_score"], 0.75)

    def test_invokes_existing_judge_script_for_incomplete_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result_dir = tmp_path / "results"
            output_dir = tmp_path / "judged"
            result_dir.mkdir()
            output_dir.mkdir()
            task = {"id": "beta_20", "suite": "suite.json", "concept_id": "beta"}
            (result_dir / "beta_20.json").write_text("{}", encoding="utf-8")

            def fake_run(command, **kwargs):
                self.assertIn(str(judge_steering_batch.JUDGE_SCRIPT), command)
                self.assertIn("--concept-id", command)
                self.assertIn("--stage", command)
                self.write_complete_summary(output_dir / "beta_20_summary.json")
                return subprocess_result(0)

            with mock.patch.object(
                judge_steering_batch.subprocess, "run", side_effect=fake_run
            ):
                row = judge_steering_batch.run_task(
                    task,
                    result_dir=result_dir,
                    output_dir=output_dir,
                    suite=None,
                    provider="azure-openai",
                    model_name="judge-model",
                    api_key_env="API_KEY",
                    judge_workers=3,
                    repeats=2,
                    seed=7,
                )

            self.assertEqual(row["status"], "judged")
            self.assertEqual(row["concept_id"], "beta")

    def test_validate_manifest_requires_formal_stage_and_unique_ids(self):
        with self.assertRaisesRegex(ValueError, "formal_steering"):
            judge_steering_batch.validate_manifest({"stage": "screen", "tasks": []})
        with self.assertRaisesRegex(ValueError, "unique"):
            judge_steering_batch.validate_manifest(
                {
                    "stage": "formal_steering",
                    "tasks": [
                        {"id": "dup", "suite": "suite.json"},
                        {"id": "dup", "suite": "suite.json"},
                    ],
                }
            )

    def test_validate_manifest_accepts_screen_stage(self):
        tasks = judge_steering_batch.validate_manifest(
            {
                "stage": "steering_screen",
                "tasks": [{"id": "alpha_10", "suite": "suite.json"}],
            },
            "steering_screen",
        )
        self.assertEqual(tasks[0]["id"], "alpha_10")


def subprocess_result(returncode: int):
    return type(
        "Completed",
        (),
        {"returncode": returncode, "stdout": "", "stderr": ""},
    )()


if __name__ == "__main__":
    unittest.main()

