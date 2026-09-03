import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_leaderboard.py"
SPEC = importlib.util.spec_from_file_location("build_leaderboard", SCRIPT)
build_leaderboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_leaderboard)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def agent(model: str, **overrides):
    row = {
        "status": "complete",
        "run_id": f"{model}-run",
        "harness": "codex",
        "model": model,
        "reasoning_effort": "high",
        "elapsed_seconds": 120.0,
        "feature_id": 10,
        "exact_match": False,
        "gt_normalized": {"mean_score": 0.25},
        "activation": {
            "positive_mean_rank": 20.0,
            "auroc": 0.75,
            "expert_pattern_spearman": 0.4,
        },
        "steering": {
            "target_effect": 0.1,
            "pe_target_relevance": 3.0,
            "pe_task_preservation": 2.5,
        },
    }
    row.update(overrides)
    return row


class BuildLeaderboardTest(unittest.TestCase):
    def test_score_summary_and_audit_define_run_intersection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            audit = root / "audit.json"
            write_json(
                summary,
                {
                    "eligible_runs": 2,
                    "runs": [
                        {"run_id": "selected", "status": "scored"},
                        {"run_id": "selected-but-failed", "status": "failed"},
                    ],
                },
            )
            write_json(
                audit,
                {
                    "runs": [
                        {"run_id": "selected", "eligible": True},
                        {"run_id": "stale", "eligible": True},
                    ]
                },
            )
            with self.assertRaisesRegex(ValueError, "incomplete or duplicate"):
                build_leaderboard._score_summary_run_ids(summary)

            write_json(
                summary,
                {
                    "eligible_runs": 1,
                    "runs": [{"run_id": "selected", "status": "skipped"}],
                },
            )
            self.assertEqual(
                build_leaderboard._score_summary_run_ids(summary), {"selected"}
            )
            self.assertEqual(
                build_leaderboard._trace_eligible_run_ids(audit),
                {"selected", "stale"},
            )

    def test_collects_only_complete_runs_for_stable_benchmark_tasks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original_root = build_leaderboard.ROOT
            build_leaderboard.ROOT = root
            try:
                write_json(
                    root / "data" / "benchmark_v1.json",
                    {
                        "schema": 1,
                        "tasks": [
                            {
                                "task": "tasks/task_a.json",
                                "concept_id": "alpha",
                                "expert_feature_id": 10,
                                "reference": "data/eval/a_reference.json",
                            },
                            {
                                "task": "tasks/task_b.json",
                                "concept_id": "beta",
                                "expert_feature_id": 20,
                                "reference": "data/eval/b_reference.json",
                            },
                            {
                                "task": "tasks/task_unstable.json",
                                "expert_feature_id": 30,
                                "reference": "data/eval/unstable_reference.json",
                            },
                        ],
                    },
                )
                write_json(root / "data" / "eval" / "a_reference.json", {"status": "stable_reference"})
                write_json(root / "data" / "eval" / "b_reference.json", {"status": "stable_reference"})
                write_json(root / "data" / "eval" / "unstable_reference.json", {"status": "draft"})
                compact = root / "results" / "agent_eval" / "compact.json"
                write_json(
                    compact,
                    {
                        "schema": 1,
                        "task": "tasks/task_a.json",
                        "agents": [
                            agent("strong", exact_match=True, steering={"target_effect": 0.8}, gt_normalized={"mean_score": 0.9}),
                            agent("broken", error="agent crashed"),
                            agent("missing", steering={}),
                        ],
                    },
                )
                write_json(
                    root / "results" / "agent_eval" / "unknown.json",
                    {"schema": 1, "task": "tasks/not_official.json", "agents": [agent("unknown")]},
                )
                write_json(root / "results" / "agent_eval" / "grid.json", {"schema": 1, "feature": {"feature_id": 5}})

                payload = build_leaderboard.collect_leaderboard(
                    sorted((root / "results" / "agent_eval").glob("*.json")),
                    root / "data" / "benchmark_v1.json",
                    {"strong-run", "broken-run", "missing-run"},
                    {"strong-run", "broken-run", "missing-run"},
                )
            finally:
                build_leaderboard.ROOT = original_root

        self.assertEqual(payload["run_counts"]["included"], 1)
        self.assertEqual(payload["task_coverage"]["stable_benchmark_tasks"], 2)
        self.assertEqual(payload["task_coverage"]["covered_tasks"], 1)
        self.assertEqual(payload["task_coverage"]["missing_tasks"], ["tasks/task_b.json"])
        self.assertEqual(
            payload["run_counts"]["skipped_by_reason"],
            {
                "agent_error_present": 1,
                "missing_compact_scores": 1,
                "not_compact_agent_result": 1,
                "task_not_in_stable_benchmark": 1,
            },
        )
        self.assertEqual(payload["configurations"][0]["configuration"], "codex/strong (high)")
        self.assertEqual(payload["configurations"][0]["completed_runs"], 1)
        self.assertEqual(payload["runs"][0]["concept_id"], "alpha")

    def test_deduplicates_runs_recomputes_exact_and_reports_macro_configuration_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original_root = build_leaderboard.ROOT
            build_leaderboard.ROOT = root
            try:
                tasks = []
                for name, expert in (("a", 10), ("b", 20)):
                    reference = f"data/{name}.json"
                    write_json(root / reference, {"status": "stable_reference"})
                    tasks.append(
                        {
                            "task": f"tasks/{name}.json",
                            "concept_id": name,
                            "expert_feature_id": expert,
                            "reference": reference,
                        }
                    )
                benchmark = root / "data" / "benchmark_v2.json"
                write_json(benchmark, {"schema": 1, "tasks": tasks})
                first = agent(
                    "model",
                    run_id="same-run",
                    feature_id=10,
                    exact_match=False,
                    elapsed_seconds=60,
                    gt_normalized={"mean_score": 1.0},
                    steering={
                        "target_effect": 0.8,
                        "causal_stable": True,
                        "usable_steering": True,
                        "pe_target_relevance": 3.0,
                    },
                )
                second = agent(
                    "model",
                    run_id="run-b",
                    feature_id=3,
                    elapsed_seconds=180,
                    gt_normalized={"mean_score": 0.0},
                    steering={"target_effect": 0.0},
                )
                incomplete = agent("model", run_id="incomplete", status="submitted")
                stale = agent("model", run_id="stale")
                ineligible = agent("model", run_id="ineligible")
                paths = []
                for index, payload in enumerate(
                    (
                        {
                            "task": "tasks/a.json",
                            "agents": [first, incomplete, stale, ineligible],
                        },
                        {"task": "tasks/a.json", "agents": [first]},
                        {"task": "tasks/b.json", "agents": [second]},
                    )
                ):
                    path = root / "results" / f"{index}.json"
                    write_json(path, payload)
                    paths.append(path)
                payload = build_leaderboard.collect_leaderboard(
                    paths,
                    benchmark,
                    {"same-run", "run-b", "incomplete", "ineligible"},
                    {"same-run", "run-b", "incomplete", "stale"},
                )
            finally:
                build_leaderboard.ROOT = original_root

        summary = payload["configurations"][0]
        self.assertEqual(summary["completed_tasks"], 2)
        self.assertEqual(summary["coverage_rate"], 1.0)
        self.assertEqual(summary["macro_gt_normalized_activation"], 0.5)
        self.assertEqual(summary["mean_feature_discovery_score"], 0.5)
        self.assertEqual(summary["total_feature_discovery_score"], 1.0)
        self.assertEqual(summary["maximum_feature_discovery_score"], 2)
        self.assertEqual(summary["exact_match_rate"], 0.5)
        self.assertEqual(summary["causal_steering_rate"], 0.5)
        self.assertEqual(summary["usable_steering_rate"], 0.5)
        self.assertEqual(summary["median_elapsed_seconds"], 120.0)
        self.assertEqual(payload["run_counts"]["included"], 2)
        self.assertEqual(
            payload["run_counts"]["skipped_by_reason"],
            {
                "agent_status_not_complete": 1,
                "duplicate_run_id": 1,
                "run_not_in_score_summary": 1,
                "run_not_trace_eligible": 1,
            },
        )
        self.assertTrue(payload["runs"][0]["exact_match"])
        self.assertEqual(payload["runs"][0]["feature_discovery_score"], 1.0)
        self.assertEqual(payload["runs"][0]["pe_target_relevance"], 3.0)

    def test_result_paths_deduplicate_default_absolute_and_same_relative_dir(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results" / "agent_eval"
            write_json(results / "compact.json", {"schema": 1, "agents": []})
            current = Path.cwd()
            try:
                import os

                os.chdir(root)
                paths = build_leaderboard._result_paths(
                    [results.resolve(), Path("results/agent_eval")], []
                )
            finally:
                os.chdir(current)

        self.assertEqual(paths, [(results / "compact.json").resolve()])

    def test_markdown_renders_model_and_run_tables(self):
        payload = {
            "benchmark": "data/benchmark_v1.json",
            "task_coverage": {"stable_benchmark_tasks": 1, "covered_tasks": 1},
            "run_counts": {"included": 1, "skipped": 0},
            "configurations": [
                {
                    "configuration": "codex/model-a (high)",
                    "harness": "codex",
                    "model": "model-a",
                    "reasoning_effort": "high",
                    "benchmark_tasks": 1,
                    "completed_tasks": 1,
                    "completed_runs": 1,
                    "exact_matches": 1,
                    "exact_match_rate": 1.0,
                    "macro_gt_normalized_activation": 0.8,
                    "coverage_rate": 1.0,
                    "causal_steering_rate": 0.0,
                    "usable_steering_rate": 0.0,
                    "median_elapsed_seconds": 120.0,
                    "latency_runs": 1,
                }
            ],
            "runs": [
                {
                    "model": "model-a",
                    "harness": "codex",
                    "reasoning_effort": "high",
                    "concept_id": "alpha",
                    "selected_feature_id": 10,
                    "exact_match": True,
                    "gt_normalized_activation": 0.8,
                    "positive_mean_rank": 20.0,
                    "activation_auroc": 0.9,
                    "expert_spearman": None,
                    "steering_effect": 0.4,
                    "steering_pattern_correlation": None,
                }
            ],
        }
        markdown = build_leaderboard.render_markdown(payload)
        self.assertIn(
            "| codex/model-a (high) | 1/1 | 0.800 | 0.800/1 | 1.000 | 0.000 | 0.000 | 2.0 min |",
            markdown,
        )
        self.assertIn("| codex/model-a (high) | alpha | 10 | yes | 0.800 | 20.0 | 0.900 |  | 0.400 |  |", markdown)


if __name__ == "__main__":
    unittest.main()
