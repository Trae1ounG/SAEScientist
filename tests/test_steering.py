import unittest
import importlib.util
from pathlib import Path

import torch

from sae_bench.steering import matched_random_direction, steering_hook


SPEC = importlib.util.spec_from_file_location(
    "smoke_qwen3", Path(__file__).parents[1] / "scripts" / "smoke_qwen3.py"
)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class SteeringTest(unittest.TestCase):
    def test_hook_changes_all_tokens_by_default(self):
        output = torch.zeros(2, 3, 4)
        changed = steering_hook(torch.ones(4), 2.0)(None, None, output)
        self.assertTrue(torch.equal(changed, torch.full((2, 3, 4), 2.0)))

    def test_last_position_mode(self):
        output = torch.zeros(2, 3, 4)
        changed = steering_hook(torch.ones(4), 2.0, "last")(None, None, output)
        self.assertTrue(torch.equal(changed[:, :-1], output[:, :-1]))
        self.assertTrue(torch.equal(changed[:, -1], torch.full((2, 4), 2.0)))

    def test_zero_strength_is_noop(self):
        output = (torch.randn(1, 2, 4), "cache")
        changed = steering_hook(torch.randn(4), 0.0)(None, None, output)
        self.assertTrue(torch.equal(changed[0], output[0]))
        self.assertEqual(changed[1], "cache")

    def test_random_control_matches_norm(self):
        direction = torch.arange(1, 9, dtype=torch.float32)
        random_direction = matched_random_direction(direction, seed=7)
        self.assertAlmostEqual(direction.norm().item(), random_direction.norm().item(), places=5)

    def test_repetition_is_degenerate(self):
        self.assertTrue(SMOKE.is_degenerate("...\n" * 20))
        self.assertTrue(SMOKE.is_degenerate("style" * 100))
        self.assertFalse(
            SMOKE.is_degenerate(
                "Sleep consolidates memories and restores attention, helping learners retain new ideas."
            )
        )
        self.assertFalse(
            SMOKE.is_degenerate("睡眠对学习至关重要，因为它有助于巩固记忆并恢复注意力。")
        )

    def test_cjk_ratio(self):
        self.assertEqual(SMOKE.cjk_ratio("中文"), 1.0)
        self.assertEqual(SMOKE.cjk_ratio("abc"), 0.0)


if __name__ == "__main__":
    unittest.main()

