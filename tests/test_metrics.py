"""Signal-processing layer against synthetic ground truth: known cadence
and planted overstride must be recovered; a stationary subject must yield
no gait events (the amplitude gate)."""

import os
import tempfile
import unittest

from runform.metrics import compute_metrics
from tests.synthetic import write_gait_csv


def _run(tmpdir, name, **kwargs):
    path = write_gait_csv(os.path.join(tmpdir, name), **kwargs)
    # width == height -> aspect 1.0, so synthetic coordinates pass through
    # aspect correction unchanged and ground truth stays exact.
    return compute_metrics(path, fps=30.0, width=1000, height=1000)


class TestMetricsRecovery(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_known_cadence_recovered(self):
        m = _run(self.tmpdir, "clean.csv", cadence_spm=180.0, seconds=10.0)
        # BUILD_PLAN Phase 0 acceptance: cadence within 3 spm, strike count
        # within +-1 of truth (30 steps in 10 s at 180 spm).
        self.assertAlmostEqual(m["cadence_spm"], 180.0, delta=3.0)
        self.assertAlmostEqual(m["steps_detected"], 30, delta=1)

    def test_direction_detected(self):
        right = _run(self.tmpdir, "right.csv", direction=1)
        left = _run(self.tmpdir, "left.csv", direction=-1)
        self.assertEqual(right["_meta"]["direction"], "right")
        self.assertEqual(left["_meta"]["direction"], "left")

    def test_planted_overstride_recovered(self):
        clean = _run(self.tmpdir, "clean2.csv", swing_amp=0.08)
        planted = _run(self.tmpdir, "overstride.csv", swing_amp=0.14)
        for side in ("left", "right"):
            c = clean["per_side"][side]["overstride_ratio"]["mean"]
            p = planted["per_side"][side]["overstride_ratio"]["mean"]
            # Expected ratio ~ swing_amp / leg_length (~0.40): the planted
            # fault must read clearly larger, and land out of range.
            self.assertGreater(p, c + 0.08, f"{side}: planted fault not recovered")
            self.assertGreater(p, 0.28)
            self.assertLess(c, 0.24)

    def test_stationary_subject_yields_no_gait(self):
        m = _run(self.tmpdir, "still.csv", stationary=True, noise=0.002)
        # The amplitude gate must reject jitter on a standing subject
        # rather than reporting a plausible-looking cadence.
        self.assertEqual(m["steps_detected"], 0)

    def test_vertical_oscillation_recovered(self):
        m = _run(self.tmpdir, "bounce.csv", bounce_p2p=0.03)
        leg = m["_meta"]["leg_length_units"]
        expected = 0.03 * 0.95 / leg  # p5-p95 of a sine spans ~95% of p2p
        self.assertAlmostEqual(m["vertical_oscillation_ratio"], expected, delta=0.02)

    def test_summaries_carry_spread_and_n(self):
        m = _run(self.tmpdir, "spread.csv")
        for side in ("left", "right"):
            for name in ("knee_angle_at_strike_deg", "overstride_ratio"):
                s = m["per_side"][side][name]
                self.assertIsNotNone(s)
                for key in ("mean", "sd", "n"):
                    self.assertIn(key, s)  # never a bare mean (CLAUDE.md)


if __name__ == "__main__":
    unittest.main()
