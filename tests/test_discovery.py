import unittest

from sae_scientist.discovery import rank_contrast_features


class DiscoveryTest(unittest.TestCase):
    def test_contrast_feature_is_ranked_first(self):
        positive = [{7: 4.0, 9: 1.0}, {7: 2.0}]
        negative = [{9: 1.0}, {9: 1.0}]
        rows = rank_contrast_features(positive, negative)
        self.assertEqual(rows[0]["feature_id"], 7)
        self.assertEqual(rows[0]["positive_rate"], 1.0)

    def test_empty_group_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_contrast_features([], [{}])


if __name__ == "__main__":
    unittest.main()
