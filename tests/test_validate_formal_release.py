from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_formal_release.py"
SPEC = importlib.util.spec_from_file_location("validate_formal_release", SCRIPT)
validate_formal_release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate_formal_release)


def fixtures() -> tuple[dict, dict]:
    configs = [
        {
            "harness": "codex",
            "model": f"model-{index}",
            "reasoning_effort": "high",
            "completed_tasks": 2,
            "completed_runs": 2,
            "latency_runs": 2,
            "coverage_rate": 1.0,
            "missing_tasks": [],
        }
        for index in range(2)
    ]
    behavior = {
        "status": "complete",
        "coverage": {"expected_cells": 4, "included_cells": 4, "missing_cells": []},
        "selection": {
            "summary_runs": 4,
            "trace_eligible_summary_runs": 4,
            "superseded_attempts": [],
        },
        "configurations": configs,
    }
    runs = [
        {
            "run_id": str(index),
            **{metric: 0.0 for metric in validate_formal_release.REQUIRED_RUN_METRICS},
        }
        for index in range(4)
    ]
    leaderboard = {
        "task_coverage": {"stable_benchmark_tasks": 2, "covered_tasks": 2, "missing_tasks": []},
        "run_counts": {"included": 4, "skipped": 0},
        "configurations": configs,
        "runs": runs,
    }
    return behavior, leaderboard


class ValidateFormalReleaseTest(unittest.TestCase):
    def test_accepts_complete_release(self) -> None:
        behavior, leaderboard = fixtures()
        self.assertEqual(
            validate_formal_release.validate(
                behavior, leaderboard, expected_tasks=2, expected_configurations=2
            ),
            {"tasks": 2, "configurations": 2, "runs": 4},
        )

    def test_rejects_missing_metric(self) -> None:
        behavior, leaderboard = fixtures()
        leaderboard["runs"][0]["steering_effect"] = None
        with self.assertRaisesRegex(ValueError, "missing steering_effect"):
            validate_formal_release.validate(
                behavior, leaderboard, expected_tasks=2, expected_configurations=2
            )


if __name__ == "__main__":
    unittest.main()

