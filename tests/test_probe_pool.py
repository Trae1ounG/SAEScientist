import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_probe_pool.py"
SPEC = importlib.util.spec_from_file_location("run_probe_pool", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProbeRequestTests(unittest.TestCase):
    def test_minimal_request_uses_default_top_k(self):
        request = MODULE.validate_request({"id": "one", "texts": ["a cat"]}, 64)
        self.assertEqual(request["top_k"], 64)
        self.assertEqual(request["feature_ids"], [])

    def test_rejects_oversized_request(self):
        with self.assertRaisesRegex(ValueError, "texts must contain"):
            MODULE.validate_request(
                {"id": "many", "texts": ["x"] * (MODULE.MAX_TEXTS_PER_REQUEST + 1)},
                64,
            )

    def test_rejects_boolean_feature_id(self):
        with self.assertRaisesRegex(ValueError, "feature_ids"):
            MODULE.validate_request(
                {"id": "bad", "texts": ["x"], "feature_ids": [True]}, 64
            )


if __name__ == "__main__":
    unittest.main()

