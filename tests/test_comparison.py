"""Phase 5 acceptance: synthetic session pairs with known deltas classify
correctly as significant vs. noise; setup drift is flagged."""

import unittest

from runform.comparison import compare_sessions
from runform.session import interpret_session
from tests.synthetic import metrics_payload, session_clips


def assessment(**overrides):
    clips = session_clips({
        band: metrics_payload(**overrides) for band in ("easy", "moderate", "fast")
    })
    return interpret_session(clips)


def find(deltas, metric, side="left", band="moderate"):
    for d in deltas:
        if d["metric"] == metric and d["side"] == side and d["speed_label"] == band:
            return d
    raise AssertionError(f"no delta for {metric}/{side}/{band}")


class TestComparison(unittest.TestCase):
    def test_real_change_is_significant_and_directional(self):
        before = assessment(overstride=0.32, overstride_sd=0.02, cadence=155.0)
        after = assessment(overstride=0.20, overstride_sd=0.02, cadence=166.0)
        cmp = compare_sessions(before, after)

        ov = find(cmp["metric_deltas"], "overstride_ratio")
        self.assertTrue(ov["significant"])  # |−0.12| > 0.02 noise floor
        self.assertEqual(ov["verdict"], "improved_toward_range")

        cad = find(cmp["metric_deltas"], "cadence_spm", side=None)
        self.assertTrue(cad["significant"])  # +11 spm > 3 spm floor
        self.assertEqual(cad["verdict"], "improved_toward_range")

    def test_noise_is_reported_as_no_change(self):
        # +8 ms on ground contact with 15 ms stride-to-stride SD is not a
        # result, and must not be dressed up as one.
        before = assessment(contact=250.0, contact_sd=15.0)
        after = assessment(contact=258.0, contact_sd=15.0)
        cmp = compare_sessions(before, after)
        gc = find(cmp["metric_deltas"], "ground_contact_ms")
        self.assertFalse(gc["significant"])
        self.assertEqual(gc["verdict"], "no_measurable_change")

    def test_regression_reported_honestly(self):
        before = assessment(overstride=0.20, overstride_sd=0.02)
        after = assessment(overstride=0.32, overstride_sd=0.02)
        cmp = compare_sessions(before, after)
        ov = find(cmp["metric_deltas"], "overstride_ratio")
        self.assertTrue(ov["significant"])
        self.assertEqual(ov["verdict"], "moved_away_from_range")

    def test_setup_drift_flagged(self):
        before = assessment(leg_length=0.40)
        after = assessment(leg_length=0.33)  # camera moved ~18%
        cmp = compare_sessions(before, after)
        kinds = {f["kind"] for f in cmp["setup_drift"]}
        self.assertIn("scale_change", kinds)

    def test_matched_setup_not_flagged(self):
        cmp = compare_sessions(assessment(), assessment())
        self.assertEqual(cmp["setup_drift"], [])
        self.assertEqual(cmp["labels_compared"], ["easy", "moderate", "fast"])


if __name__ == "__main__":
    unittest.main()
