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

    def test_cadence_resolves_between_whole_frame_intervals(self):
        """Cadence must resolve finer than the whole-frame ladder.

        A median of integer frame intervals can only land on 60*fps/k, so
        at 30 fps it jumps ~180.0 -> 171.4 -> 163.6, steps of 8-10 spm.
        Phase 0 has to hit +-3 spm, so any true cadence between two rungs
        has to be recovered by averaging, not by the median alone.
        """
        for target in (173.0, 168.0, 186.0):
            m = _run(self.tmpdir, f"cad{target}.csv",
                     cadence_spm=target, seconds=14.0)
            self.assertAlmostEqual(
                m["cadence_spm"], target, delta=3.0,
                msg=f"cadence {target} not resolved off the whole-frame ladder",
            )

    def test_contact_time_never_spans_a_stride(self):
        """A dropped toe-off must not pair its strike with a later stride.

        Occluding one leg across a toe-off leaves the surrounding strikes
        intact, so an unbounded "next toe-off" search jumps a whole stride
        and reports ~967 ms of ground contact -- physically impossible and
        far outside any reference range. Nothing should be emitted for
        that stride instead.
        """
        m = _run(self.tmpdir, "occluded.csv", cadence_spm=180.0, seconds=12.0,
                 occlude=("left", 26, 46))
        # Same-foot stride period at 180 spm = 667 ms; contact is a
        # fraction of it. Allow the full stride as a generous ceiling.
        stride_ms = 60_000.0 / (180.0 / 2)
        gc = m["per_side"]["left"]["ground_contact_ms"]
        self.assertIsNotNone(gc)
        self.assertLess(
            gc["mean"], stride_ms,
            "contact time exceeds a full stride -- pairing spanned strides",
        )

    def test_trunk_lean_sign_holds_facing_either_way(self):
        """Forward lean must read positive regardless of facing.

        Every real clip so far is direction="right", so the screen-left
        path is otherwise unexercised -- and a sign error there would
        silently invert lean for half of all future footage.
        """
        for name, direction in (("lean_r.csv", 1), ("lean_l.csv", -1)):
            m = _run(self.tmpdir, name, direction=direction)
            for side in ("left", "right"):
                lean = m["per_side"][side]["trunk_lean_deg"]
                self.assertIsNotNone(lean)
                self.assertGreater(
                    lean["mean"], 0.0,
                    f"direction={direction} {side}: forward lean read as negative",
                )


if __name__ == "__main__":
    unittest.main()
