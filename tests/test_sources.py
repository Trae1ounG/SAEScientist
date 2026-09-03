import unittest

from sae_scientist.sources import (
    GEMMA_SCOPE_9B_IT_RES,
    QWEN3_8B_BASE_L0_50,
    require_official_source,
)


class OfficialSourceTest(unittest.TestCase):
    def test_known_sources_are_allowed(self):
        for source in (QWEN3_8B_BASE_L0_50, GEMMA_SCOPE_9B_IT_RES):
            self.assertEqual(require_official_source(source.repo), source)

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            require_official_source("community/unknown-sae")


if __name__ == "__main__":
    unittest.main()
