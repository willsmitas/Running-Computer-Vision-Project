"""Speed-profile fit and fault-pattern classification."""

import unittest

from runform.speed_profile import classify, fit

ALL = ("easy", "moderate", "fast")


class TestClassify(unittest.TestCase):
    def test_patterns(self):
        self.assertEqual(classify(set(), ALL), "absent")
        self.assertEqual(classify({"easy", "moderate", "fast"}, ALL), "constant")
        self.assertEqual(classify({"fast"}, ALL), "degrades_with_speed")
        self.assertEqual(classify({"moderate", "fast"}, ALL), "degrades_with_speed")
        self.assertEqual(classify({"easy"}, ALL), "slow_only")
        self.assertEqual(classify({"moderate"}, ALL), "inconsistent")

    def test_single_clip_session_makes_no_speed_claim(self):
        self.assertEqual(classify({"easy"}, ("easy",)), "single_speed")
        self.assertEqual(classify(set(), ("easy",)), "absent")

    def test_two_clip_session(self):
        two = ("easy", "fast")
        self.assertEqual(classify({"fast"}, two), "degrades_with_speed")
        self.assertEqual(classify({"easy"}, two), "slow_only")
        self.assertEqual(classify({"easy", "fast"}, two), "constant")


class TestFit(unittest.TestCase):
    def test_linear_recovery(self):
        # Cadence rising ~14 spm per m/s.
        f = fit([(2.5, 160.0), (3.2, 169.8), (4.0, 181.0)])
        self.assertAlmostEqual(f["slope"], 14.0, delta=1.0)
        self.assertGreater(f["r2"], 0.98)
        self.assertEqual(f["n"], 3)

    def test_flat_metric(self):
        f = fit([(2.5, 170.0), (3.2, 170.0), (4.0, 170.0)])
        self.assertAlmostEqual(f["slope"], 0.0, delta=1e-6)
        self.assertEqual(f["r2"], 1.0)

    def test_single_point_returns_none(self):
        self.assertIsNone(fit([(3.0, 170.0)]))


if __name__ == "__main__":
    unittest.main()
