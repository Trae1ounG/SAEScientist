from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_leaderboard", ROOT / "scripts" / "plot_leaderboard.py"
)
plot_leaderboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_leaderboard)


def run(**overrides):
    row = {
        "harness": "codex",
        "model": "model",
        "reasoning_effort": "high",
        "gt_normalized_activation": 0.7,
        "steering_effect": 0.3,
        "pe_target_relevance": 3.0,
        "pe_task_preservation": 2.5,
        "exact_match": False,
        "usable_steering": True,
    }
    row.update(overrides)
    return row


class PlotLeaderboardTest(unittest.TestCase):
    def test_configuration_color_is_fixed(self):
        first = plot_leaderboard.configuration_color("codex/model (high)")
        second = plot_leaderboard.configuration_color("codex/model (high)")
        self.assertEqual(first, second)
        self.assertRegex(first, r"^#[0-9a-f]{6}$")
        self.assertEqual(
            plot_leaderboard.configuration_colors(["b", "a"]),
            plot_leaderboard.configuration_colors(["a", "b"]),
        )
        self.assertNotEqual(
            plot_leaderboard.configuration_colors(["a", "b"])["a"],
            plot_leaderboard.configuration_colors(["a", "b"])["b"],
        )

    def test_collect_points_skips_only_missing_coordinate_pairs(self):
        discovery, behavior = plot_leaderboard.collect_points(
            {
                "runs": [
                    run(),
                    run(gt_normalized_activation=None),
                    run(pe_task_preservation=None),
                    "not-a-row",
                ]
            }
        )
        self.assertEqual(len(discovery), 2)
        self.assertEqual(len(behavior), 2)
        self.assertEqual(discovery[0]["configuration"], "codex/model (high)")

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib is not installed"
    )
    def test_render_writes_png_and_svg_even_when_values_are_missing(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            outputs = plot_leaderboard.render_plots(
                {"runs": [{}, "not-a-row"]}, output_dir
            )
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()

