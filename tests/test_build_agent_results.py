import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_agent_results.py"
SPEC = importlib.util.spec_from_file_location("build_agent_results", SCRIPT)
build_agent_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_agent_results)


class BuildAgentResultsTest(unittest.TestCase):
    def _pe_fixture(self, root: Path) -> dict:
        candidate_id = "task__candidate_7"
        expert_id = "task__expert_anchor"
        paths = {
            "candidate_summary_path": root / f"{candidate_id}_summary.json",
            "expert_summary_path": root / f"{expert_id}_summary.json",
            "candidate_rows_path": root / f"{candidate_id}.jsonl",
            "expert_rows_path": root / f"{expert_id}.jsonl",
            "candidate_result_path": root / f"{candidate_id}.json",
            "expert_result_path": root / f"{expert_id}.json",
        }
        for prefix, expected_id in (
            ("candidate", candidate_id),
            ("expert", expert_id),
        ):
            summary = {
                "result": f"results/{expected_id}.json",
                "judge_provider": "test-provider",
                "judge_model": "judge",
                "stage": "formal_steering",
                "repeats": 2,
                "expected_rows": 4,
                "valid_rows": 4,
                "error_rows": 0,
            }
            paths[f"{prefix}_summary_path"].write_text(json.dumps(summary))
            paths[f"{prefix}_result_path"].write_text(
                json.dumps(
                    {
                        "feature": {"checkpoint": "layer/params.npz"},
                        "suite": {"path": "/checkout/data/suite.json"},
                        "steering": {"selected_alpha": 10},
                    }
                )
            )
            paths[f"{prefix}_rows_path"].write_text(
                "\n".join(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "repeat": repeat,
                            "ratings": {
                                "feature": {"target_relevance": 1},
                                "baseline": {"target_relevance": 0},
                                "random": {"target_relevance": 0},
                            },
                        }
                    )
                    for case_id in ("a", "b")
                    for repeat in range(2)
                )
                + "\n"
            )
        return {
            "candidate_id": candidate_id,
            "expert_id": expert_id,
            "expected_suite": "data/suite.json",
            **paths,
        }

    def test_attempt_order_is_numeric_and_primary_first(self):
        run_ids = ["run-retry-2", "run-retry-10", "run-01"]
        self.assertEqual(
            sorted(run_ids, key=build_agent_results.retry_attempt),
            ["run-01", "run-retry-2", "run-retry-10"],
        )

    def test_reasoning_effort_is_loaded_from_run_manifest(self):
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_root = runs_root / "run-01"
            run_root.mkdir()
            (run_root / "run.json").write_text(
                json.dumps({"reasoning_effort": "high"})
            )
            self.assertEqual(
                build_agent_results.score_reasoning_effort(
                    {"run_id": "run-01"}, runs_root
                ),
                "high",
            )

    def test_activation_score_summary_rejects_wrong_score_run_id(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores"
            scores.mkdir()
            score = scores / "selected.json"
            score.write_text(json.dumps({"run_id": "stale"}))
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "eligible_runs": 1,
                        "runs": [
                            {
                                "run_id": "selected",
                                "status": "skipped",
                                "output": str(score),
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "wrong run ID"):
                build_agent_results.activation_score_paths(summary, scores)

    def test_build_results_keeps_earliest_summary_attempt_and_reasoning(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_name = "tasks/task.json"
            reference_name = "data/reference.json"
            (root / "tasks").mkdir()
            (root / "data").mkdir()
            (root / "scores").mkdir()
            (root / "runs").mkdir()
            (root / task_name).write_text(json.dumps({"task_id": "task"}))
            (root / reference_name).write_text(
                json.dumps(
                    {
                        "protocol": {"alpha": 10},
                        "steering": {
                            "baseline": {"target_relevance": 0},
                            "random": {"target_relevance": 0},
                            "feature": {
                                "target_relevance": 4,
                                "target_success_rate": 1,
                                "task_preservation": 4,
                                "usable_target_rate": 1,
                                "degenerate_rate": 0,
                            },
                        },
                    }
                )
            )
            benchmark = root / "benchmark.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task": task_name,
                                "suite": "data/suite.json",
                                "reference": reference_name,
                                "expert_feature_id": 9,
                            }
                        ]
                    }
                )
            )
            run_ids = ("task-offline-model-01", "task-offline-model-retry-10")
            audit = root / "audit.json"
            audit.write_text(
                json.dumps({"runs": [{"run_id": run_id, "eligible": True} for run_id in run_ids]})
            )
            for run_id in run_ids:
                run_root = root / "runs" / run_id
                run_root.mkdir()
                (run_root / "run.json").write_text(
                    json.dumps({"reasoning_effort": "high"})
                )
                score = {
                    "run_id": run_id,
                    "task": task_name,
                    "harness": "codex",
                    "model": "model",
                    "feature_id": 9,
                    "source_commit": "commit",
                    "elapsed_seconds": 1,
                    "gt_normalized": {"mean_score": 1},
                    "expert_activation_spearman": 1,
                    "activation_rank": {
                        "positive": {"mean_activation": 4, "mean_rank": 1, "mean_percentile": 1},
                        "hard_negative": {"mean_activation": 0, "mean_rank": 2},
                        "neutral": {"mean_activation": 0, "mean_rank": 3},
                        "activation_auroc": 1,
                    },
                    "expert_activation_rank": {
                        "positive": {"mean_activation": 4, "mean_rank": 1},
                        "hard_negative": {"mean_activation": 0, "mean_rank": 2},
                        "neutral": {"mean_activation": 0, "mean_rank": 3},
                        "activation_auroc": 1,
                    },
                }
                (root / "scores" / f"{run_id}.json").write_text(json.dumps(score))
            activation_summary = root / "score_summary.json"
            activation_summary.write_text(
                json.dumps(
                    {
                        "eligible_runs": 2,
                        "runs": [
                            {
                                "run_id": run_id,
                                "status": "skipped",
                                "output": str(root / "scores" / f"{run_id}.json"),
                            }
                            for run_id in reversed(run_ids)
                        ],
                    }
                )
            )

            args = SimpleNamespace(
                benchmark=benchmark,
                activation_dir=root / "scores",
                activation_summary=activation_summary,
                audit=audit,
                runs_root=root / "runs",
                candidate_result_dir=[root / "raw"],
                candidate_judge_dir=[root / "candidate_judge"],
                expert_judge_dir=[root / "expert_judge"],
                output_dir=root / "output",
            )
            original_root = build_agent_results.ROOT
            build_agent_results.ROOT = root
            try:
                summary = build_agent_results.build_results(args)
            finally:
                build_agent_results.ROOT = original_root
            output = json.loads((root / "output" / "task_v2.json").read_text())

        self.assertEqual(output["agents"][0]["run_id"], run_ids[0])
        self.assertEqual(summary["complete_runs"], 1)
        self.assertEqual(
            summary["skipped"][run_ids[1]], "superseded by earlier eligible attempt"
        )
        self.assertEqual(output["agents"][0]["reasoning_effort"], "high")
        self.assertTrue(output["agents"][0]["exact_match"])
        self.assertEqual(output["agents"][0]["scores"]["overall_score"], 1.0)
        self.assertEqual(output["expert"]["score_baseline"]["overall_score"], 1.0)

    def test_benchmark_scores_are_expert_normalized(self):
        score = {
            "activation_rank": {
                "positive": {"mean_activation": 3, "mean_rank": 2},
                "hard_negative": {"mean_activation": 1},
                "neutral": {"mean_activation": 0},
                "activation_auroc": 0.8,
            },
            "expert_activation_rank": {
                "positive": {"mean_activation": 3, "mean_rank": 3},
                "hard_negative": {"mean_activation": 1},
                "neutral": {"mean_activation": 0},
                "activation_auroc": 0.8,
            },
            "expert_activation_spearman": 1.0,
        }
        steering = {"target_effect": 0.2, "pattern_correlation_to_expert": 0.5}
        expert = {"target_effect": 0.4}
        scores = build_agent_results.benchmark_scores(score, steering, expert)
        self.assertAlmostEqual(scores["rank_score"], 1.2)
        self.assertAlmostEqual(scores["activation_score"], 1.0)
        self.assertAlmostEqual(scores["steering_score"], 2 / 3)
        self.assertAlmostEqual(scores["overall_score"], 43 / 45)

    def test_validate_pe_pair_requires_complete_matching_pair(self):
        with TemporaryDirectory() as directory:
            fixture = self._pe_fixture(Path(directory))
            summary, result = build_agent_results.validate_pe_pair(**fixture)
        self.assertEqual(summary["judge_model"], "judge")
        self.assertEqual(result["feature"]["checkpoint"], "layer/params.npz")

    def test_validate_expert_pe_accepts_complete_formal_result(self):
        with TemporaryDirectory() as directory:
            fixture = self._pe_fixture(Path(directory))
            summary, result = build_agent_results.validate_expert_pe(
                expert_id=fixture["expert_id"],
                expected_suite=fixture["expected_suite"],
                summary_path=fixture["expert_summary_path"],
                rows_path=fixture["expert_rows_path"],
                result_path=fixture["expert_result_path"],
            )
        self.assertEqual(summary["judge_model"], "judge")
        self.assertEqual(result["steering"]["selected_alpha"], 10)

    def test_validate_pe_pair_rejects_mismatched_judge_and_suite(self):
        with TemporaryDirectory() as directory:
            fixture = self._pe_fixture(Path(directory))
            expert_summary = json.loads(fixture["expert_summary_path"].read_text())
            expert_summary["judge_model"] = "other"
            fixture["expert_summary_path"].write_text(json.dumps(expert_summary))
            with self.assertRaisesRegex(ValueError, "different PE judges"):
                build_agent_results.validate_pe_pair(**fixture)

            fixture = self._pe_fixture(Path(directory))
            candidate_result = json.loads(fixture["candidate_result_path"].read_text())
            candidate_result["suite"]["path"] = "data/wrong.json"
            fixture["candidate_result_path"].write_text(json.dumps(candidate_result))
            with self.assertRaisesRegex(ValueError, "wrong PE suite"):
                build_agent_results.validate_pe_pair(**fixture)

    def test_validate_pe_pair_rejects_incomplete_or_different_coverage(self):
        with TemporaryDirectory() as directory:
            fixture = self._pe_fixture(Path(directory))
            candidate_rows = fixture["candidate_rows_path"].read_text().splitlines()
            fixture["candidate_rows_path"].write_text("\n".join(candidate_rows[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "do not match its summary"):
                build_agent_results.validate_pe_pair(**fixture)

            fixture = self._pe_fixture(Path(directory))
            expert_rows = fixture["expert_rows_path"].read_text().replace(
                '"case_id": "b"', '"case_id": "c"'
            )
            fixture["expert_rows_path"].write_text(expert_rows)
            with self.assertRaisesRegex(ValueError, "coverage differs"):
                build_agent_results.validate_pe_pair(**fixture)

    def test_validate_pe_pair_rejects_wrong_result_id_stage_and_repeats(self):
        with TemporaryDirectory() as directory:
            fixture = self._pe_fixture(Path(directory))
            candidate_summary = json.loads(fixture["candidate_summary_path"].read_text())
            candidate_summary["result"] = "results/other.json"
            fixture["candidate_summary_path"].write_text(json.dumps(candidate_summary))
            with self.assertRaisesRegex(ValueError, "wrong PE summary result ID"):
                build_agent_results.validate_pe_pair(**fixture)

            fixture = self._pe_fixture(Path(directory))
            candidate_summary = json.loads(fixture["candidate_summary_path"].read_text())
            candidate_summary["stage"] = "steering_screen"
            fixture["candidate_summary_path"].write_text(json.dumps(candidate_summary))
            with self.assertRaisesRegex(ValueError, "PE stage must be formal_steering"):
                build_agent_results.validate_pe_pair(**fixture)

            fixture = self._pe_fixture(Path(directory))
            expert_summary = json.loads(fixture["expert_summary_path"].read_text())
            expert_summary["repeats"] = 1
            fixture["expert_summary_path"].write_text(json.dumps(expert_summary))
            with self.assertRaisesRegex(ValueError, "formal PE repeats must be 2"):
                build_agent_results.validate_pe_pair(**fixture)

    def test_pattern_correlation_uses_per_case_control_adjusted_effect(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.jsonl"
            expert = root / "expert.jsonl"
            candidate.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"case_id": "a", "ratings": {"feature": {"target_relevance": 4}, "baseline": {"target_relevance": 0}, "random": {"target_relevance": 0}}},
                        {"case_id": "b", "ratings": {"feature": {"target_relevance": 2}, "baseline": {"target_relevance": 0}, "random": {"target_relevance": 0}}},
                        {"case_id": "c", "ratings": {"feature": {"target_relevance": 1}, "baseline": {"target_relevance": 1}, "random": {"target_relevance": 0}}},
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            expert.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"case_id": "a", "ratings": {"feature": {"target_relevance": 3}, "baseline": {"target_relevance": 0}, "random": {"target_relevance": 0}}},
                        {"case_id": "b", "ratings": {"feature": {"target_relevance": 2}, "baseline": {"target_relevance": 0}, "random": {"target_relevance": 0}}},
                        {"case_id": "c", "ratings": {"feature": {"target_relevance": 1}, "baseline": {"target_relevance": 1}, "random": {"target_relevance": 0}}},
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            self.assertAlmostEqual(
                build_agent_results.pattern_correlation(candidate, expert), 1.0
            )

    def test_spearman_is_undefined_for_constant_pattern(self):
        self.assertIsNone(build_agent_results.spearman([1, 1, 1], [1, 2, 3]))

    def test_frozen_expert_effect_subtracts_stronger_control(self):
        reference = {
            "protocol": {"alpha": 10},
            "steering": {
                "baseline": {"target_relevance": 0.4},
                "random": {"target_relevance": 0.8},
                "feature": {
                    "target_relevance": 3.2,
                    "target_success_rate": 0.8,
                    "task_preservation": 3.0,
                    "usable_target_rate": 0.7,
                    "degenerate_rate": 0.1,
                },
            },
        }
        row = build_agent_results.frozen_expert_steering(reference)
        self.assertAlmostEqual(row["target_effect"], 0.6)
        self.assertEqual(row["pattern_correlation_to_expert"], 1.0)


if __name__ == "__main__":
    unittest.main()
