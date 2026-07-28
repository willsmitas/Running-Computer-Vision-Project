"""Confidence gating: n, SD, detection rate, and the hard asymmetry gate."""

import unittest

from runform.gating import grade_clip_metrics
from tests.synthetic import metrics_payload


class TestGating(unittest.TestCase):
    def test_clean_clip_grades_high(self):
        grades = grade_clip_metrics(metrics_payload())
        self.assertEqual(grades["cadence_spm"], "high")
        self.assertEqual(grades["left.knee_angle_at_strike_deg"], "high")
        self.assertEqual(grades["vertical_oscillation_ratio"], "high")

    def test_small_n_is_never_high(self):
        grades = grade_clip_metrics(metrics_payload(n=5, steps=5))
        self.assertEqual(grades["cadence_spm"], "low")
        self.assertEqual(grades["left.overstride_ratio"], "low")

    def test_tiny_n_is_unusable(self):
        grades = grade_clip_metrics(metrics_payload(n=2, steps=2))
        self.assertEqual(grades["cadence_spm"], "unusable")
        self.assertEqual(grades["left.knee_angle_at_strike_deg"], "unusable")

    def test_wide_sd_demotes(self):
        grades = grade_clip_metrics(metrics_payload(knee_sd=15.0))
        self.assertEqual(grades["left.knee_angle_at_strike_deg"], "low")
        grades = grade_clip_metrics(metrics_payload(knee_sd=30.0))
        self.assertEqual(grades["left.knee_angle_at_strike_deg"], "unusable")

    def test_low_detection_rate_caps_at_low(self):
        grades = grade_clip_metrics(metrics_payload(), detection_rate=0.7)
        self.assertNotIn("high", grades.values())

    def test_asymmetry_gated_hardest(self):
        # Even with both sides high-grade, asymmetry never grades high
        # (single-side view can fabricate it — risk register).
        grades = grade_clip_metrics(metrics_payload(asym={"ground_contact_ms": 20.0}))
        self.assertEqual(grades["asymmetry_pct.ground_contact_ms"], "low")
        # And with shaky underlying sides it is unusable outright.
        grades = grade_clip_metrics(
            metrics_payload(contact_sd=100.0, asym={"ground_contact_ms": 20.0})
        )
        self.assertEqual(grades["asymmetry_pct.ground_contact_ms"], "unusable")


if __name__ == "__main__":
    unittest.main()
