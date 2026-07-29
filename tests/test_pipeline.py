"""Phase 1 quality gating.

The gate exists to stop degraded tracking from reaching the metrics
silently (CLAUDE.md). These tests pin the two ways averaging defeated it
on real footage: across sides, and across joints within a side.
"""

import os
import tempfile
import unittest

import pandas as pd

from runform.pipeline import (
    MIN_KEY_JOINT_VISIBILITY,
    STRIKE_IMBALANCE_ABS,
    _key_joint_visibility,
)
from tests.synthetic import make_gait_frames


def _csv_with_visibility(tmpdir, name, overrides):
    """Synthetic landmarks CSV with chosen per-joint visibility."""
    df = make_gait_frames(seconds=6.0)
    for joint, vis in overrides.items():
        df[f"{joint}_vis"] = vis
    path = os.path.join(tmpdir, name)
    df.to_csv(path, index=False)
    return path


class TestKeyJointVisibility(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_well_tracked_side_does_not_mask_a_failing_side(self):
        """A near leg at ~0.9 must not carry a far leg at ~0.45.

        This is the real-footage failure: right leg 0.87-0.92, left leg
        0.39-0.53, and the across-sides average (0.76) cleared the 0.6
        threshold so every clip reported zero quality flags.
        """
        path = _csv_with_visibility(self.tmpdir, "onesided.csv", {
            "left_hip": 0.99, "left_knee": 0.39, "left_ankle": 0.45,
            "right_hip": 0.99, "right_knee": 0.89, "right_ankle": 0.87,
        })
        vis = _key_joint_visibility(path)
        self.assertLess(vis["left"]["min"], MIN_KEY_JOINT_VISIBILITY)
        self.assertGreater(vis["right"]["min"], MIN_KEY_JOINT_VISIBILITY)
        # The gate reads the overall min, so the bad side must win.
        self.assertLess(vis["min"], MIN_KEY_JOINT_VISIBILITY)

    def test_reliable_hip_does_not_mask_an_unreliable_knee(self):
        """Within a side, the hip must not carry the distal joints.

        The hip sits at ~1.0 in every real clip -- it is the body centre
        and essentially never occluded -- so averaging it with a 0.39 knee
        produced a 0.61 side score that cleared the threshold. Gait events
        come from the ankle and knee; a reliable hip says nothing about them.
        """
        path = _csv_with_visibility(self.tmpdir, "hipmask.csv", {
            "left_hip": 1.0, "left_knee": 0.39, "left_ankle": 0.46,
            "right_hip": 1.0, "right_knee": 0.90, "right_ankle": 0.88,
        })
        vis = _key_joint_visibility(path)
        mean_of_joints = (1.0 + 0.39 + 0.46) / 3
        self.assertGreater(mean_of_joints, MIN_KEY_JOINT_VISIBILITY)  # would have passed
        self.assertLess(vis["left"]["min"], MIN_KEY_JOINT_VISIBILITY)  # must not

    def test_clean_tracking_raises_no_flag(self):
        path = _csv_with_visibility(self.tmpdir, "clean.csv", {
            "left_hip": 0.99, "left_knee": 0.88, "left_ankle": 0.85,
            "right_hip": 0.99, "right_knee": 0.90, "right_ankle": 0.87,
        })
        vis = _key_joint_visibility(path)
        self.assertGreater(vis["min"], MIN_KEY_JOINT_VISIBILITY)


class TestStrikeImbalance(unittest.TestCase):
    """A runner alternates feet, so left and right strike counts differ by
    at most one. A wider gap means one leg's events are being missed."""

    def _imbalanced(self, n_left, n_right):
        return abs(n_left - n_right) > STRIKE_IMBALANCE_ABS

    def test_balanced_counts_pass(self):
        self.assertFalse(self._imbalanced(21, 22))
        self.assertFalse(self._imbalanced(27, 27))

    def test_the_original_real_footage_defect_is_caught(self):
        # 25/31 and 23/31 are what the per-foot spacing bug produced before
        # it was fixed; 2/30 is what the "heavy" model variant produced.
        # The gate must catch that class of failure on its own.
        self.assertTrue(self._imbalanced(25, 31))
        self.assertTrue(self._imbalanced(23, 31))
        self.assertTrue(self._imbalanced(2, 30))

    def test_tolerance_does_not_widen_with_clip_length(self):
        # The bound is physical, not proportional: a long clip is no more
        # entitled to a wide left/right gap than a short one.
        self.assertTrue(self._imbalanced(100, 106))


if __name__ == "__main__":
    unittest.main()
