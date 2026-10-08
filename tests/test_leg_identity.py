"""Leg-identity tracking against synthetic ground truth.

swap_front_rear() reproduces the real-footage failure (labels follow
front/rear roles, flipping at every crossover). These tests pin that the
tracker undoes it, never introduces swaps into clean data, and never
guesses a label across a blind span. Synthetic only: whether real swaps
are repaired must still be checked on real clips.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from runform.gating import grade_clip_metrics
from runform.leg_identity import STUCK_SIGN_FRACTION, relabel
from runform.metrics import compute_metrics
from runform.pipeline import (
    MAX_PLAUSIBLE_CADENCE_SPM,
    identity_flags,
    relabel_clip,
)
from tests.synthetic import blind, make_gait_frames, metrics_payload, swap_front_rear

# width == height -> aspect 1.0, as in test_metrics.
W = H = 1000


def _identity_accuracy(out, truth):
    """(fraction of frames labeled, fraction of labeled frames where the
    'left' label sits on the true right ankle)."""
    lab = (out["left_ankle_vis"] > 0).to_numpy()
    a = out["left_ankle_x"].to_numpy()[lab]
    t = truth["left_ankle_x"].to_numpy()[lab]
    o = truth["right_ankle_x"].to_numpy()[lab]
    wrong = np.abs(a - o) < np.abs(a - t) - 1e-9
    return float(lab.mean()), float(wrong.mean()) if len(wrong) else 0.0


class TestRelabel(unittest.TestCase):
    def test_front_rear_swap_is_undone_facing_either_way(self):
        for direction in (1, -1):
            truth = make_gait_frames(seconds=8, direction=direction, noise=0.005)
            swapped = swap_front_rear(truth, direction)
            out, rep = relabel(swapped, W, H)
            labeled, wrong = _identity_accuracy(out, truth)
            self.assertGreater(labeled, 0.9, direction)
            self.assertEqual(wrong, 0.0, direction)
            self.assertGreater(rep["sign_stuck_fraction_raw"], 0.9)
            self.assertLess(rep["sign_stuck_fraction"], 0.6)
            self.assertGreater(rep["swaps_corrected"], 5)

    def test_clean_tracking_is_left_alone(self):
        truth = make_gait_frames(seconds=8, noise=0.005)
        out, rep = relabel(truth, W, H)
        _, wrong = _identity_accuracy(out, truth)
        self.assertEqual(wrong, 0.0)
        self.assertEqual(rep["swaps_corrected"], 0)

    def test_swap_seed_flips_identity_everywhere(self):
        """The runner's 'swap red and blue' click."""
        truth = make_gait_frames(seconds=8, noise=0.005)
        out, rep = relabel(truth, W, H, swap_seed=True)
        labeled, wrong = _identity_accuracy(out, truth)
        self.assertTrue(rep["swap_seed"])
        self.assertGreater(wrong, 0.99)

    def test_hidden_foot_is_never_guessed(self):
        """While a foot's dot is invisible its label locks onto nothing,
        and identity is right again once it reappears."""
        truth = make_gait_frames(seconds=8, noise=0.005)
        occluded = swap_front_rear(truth)
        # Hide the physically-left foot, wherever the swapped labels put it.
        lab_left_is_true_left = np.isclose(occluded["left_ankle_x"], truth["left_ankle_x"])
        for f in range(100, 116):
            side = "left" if lab_left_is_true_left[f] else "right"
            occluded = blind(occluded, [side], f, f)
        out, rep = relabel(occluded, W, H)

        self.assertTrue((out.loc[100:115, "left_ankle_vis"] == 0).all())
        self.assertTrue(out.loc[100:115, "left_ankle_x"].isna().all())
        self.assertIn([100, 115], rep["unlabeled_spans"]["left"])
        _, wrong = _identity_accuracy(out, truth)
        self.assertEqual(wrong, 0.0)
        after = (out.loc[116:, "left_ankle_vis"] > 0).mean()
        self.assertGreater(after, 0.9)

    def test_both_feet_blind_too_long_ends_identity(self):
        truth = make_gait_frames(seconds=8, noise=0.005)
        out, rep = relabel(blind(swap_front_rear(truth), ["left", "right"], 100, 130), W, H)
        _, wrong = _identity_accuracy(out, truth)
        self.assertEqual(wrong, 0.0)
        self.assertTrue((out.loc[100:, "left_ankle_vis"] == 0).all())
        self.assertLess(rep["labeled_fraction"]["left"], 0.5)
        self.assertIn("leg_identity_gaps", identity_flags(rep, {"cadence_spm": 180}))

    def test_no_seed_frame_labels_nothing(self):
        truth = blind(make_gait_frames(seconds=4), ["left"], 0, 10_000)
        out, rep = relabel(truth, W, H)
        self.assertIsNone(rep["seed_frame"])
        self.assertTrue((out["right_ankle_vis"] == 0).all())
        self.assertIn("leg_identity_gaps", identity_flags(rep, {"cadence_spm": 180}))


class TestMetricsAfterRelabel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _metrics(self, df, spans=None):
        path = os.path.join(self.tmp, "x.csv")
        df.to_csv(path, index=False)
        return compute_metrics(path, fps=30.0, width=W, height=H, exclude_spans=spans)

    def test_per_side_strikes_recovered(self):
        truth = make_gait_frames(seconds=10, cadence_spm=180, noise=0.003)
        out, rep = relabel(swap_front_rear(truth), W, H)
        m = self._metrics(out, rep["unlabeled_spans"])
        expected = self._metrics(truth)
        self.assertAlmostEqual(m["cadence_spm"], expected["cadence_spm"], delta=3)
        for side in ("left", "right"):
            self.assertAlmostEqual(
                m["per_side"][side]["strikes_detected"],
                expected["per_side"][side]["strikes_detected"], delta=1,
            )
            self.assertAlmostEqual(
                m["per_side"][side]["overstride_ratio"]["mean"],
                expected["per_side"][side]["overstride_ratio"]["mean"], delta=0.02,
            )

    def test_no_event_at_a_blind_span(self):
        truth = make_gait_frames(seconds=10, cadence_spm=180)
        spans = {"left": [[100, 115]], "right": []}
        m_all = self._metrics(truth)
        m = self._metrics(truth, spans)
        self.assertLess(m["per_side"]["left"]["strikes_detected"],
                        m_all["per_side"]["left"]["strikes_detected"])
        self.assertEqual(m["per_side"]["right"]["strikes_detected"],
                         m_all["per_side"]["right"]["strikes_detected"])


class TestIdentityGating(unittest.TestCase):
    def test_stuck_labels_make_per_side_unusable(self):
        grades = grade_clip_metrics(metrics_payload(), 1.0, ["leg_labels_stuck"])
        per_side = {k: g for k, g in grades.items()
                    if k.startswith(("left.", "right.", "asymmetry_pct."))}
        self.assertTrue(per_side)
        self.assertTrue(all(g == "unusable" for g in per_side.values()))
        self.assertEqual(grades["cadence_spm"], "high")

    def test_identity_gaps_cap_per_side_and_kill_asymmetry(self):
        grades = grade_clip_metrics(metrics_payload(), 1.0, ["leg_identity_gaps"])
        for k, g in grades.items():
            if k.startswith(("left.", "right.")):
                self.assertNotEqual(g, "high", k)
            if k.startswith("asymmetry_pct."):
                self.assertEqual(g, "unusable", k)

    def test_implausible_cadence_flag(self):
        rep = {"seed_frame": 0, "labeled_fraction": {"left": 1.0, "right": 1.0},
               "sign_stuck_fraction": 0.5}
        self.assertEqual(identity_flags(rep, {"cadence_spm": 180}), [])
        self.assertIn("implausible_cadence",
                      identity_flags(rep, {"cadence_spm": MAX_PLAUSIBLE_CADENCE_SPM + 1}))
        rep["sign_stuck_fraction"] = STUCK_SIGN_FRACTION + 0.01
        self.assertIn("leg_labels_stuck", identity_flags(rep, {"cadence_spm": 180}))
        grades = grade_clip_metrics(metrics_payload(), 1.0, ["implausible_cadence"])
        self.assertEqual(grades["cadence_spm"], "unusable")


class TestRelabelClip(unittest.TestCase):
    """The web UI's swap button path: no video, no pose model."""

    def test_reseed_rewrites_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "clip_landmarks.csv")
            swap_front_rear(make_gait_frames(seconds=8, noise=0.003)).to_csv(raw, index=False)
            q = os.path.join(tmp, "clip_quality.json")
            with open(q, "w") as fh:
                json.dump({
                    "video_path": os.path.join(tmp, "missing.mp4"),
                    "skeleton_video_path": os.path.join(tmp, "clip_skeleton.mp4"),
                    "landmarks_csv_path": raw, "fps": 30.0, "width": W, "height": H,
                    "frames": 240, "detection_rate": 1.0,
                }, fh)
            r1 = relabel_clip(q, swap_seed=False)
            r2 = relabel_clip(q, swap_seed=True)
            self.assertNotIn("leg_labels_stuck", r1["quality_flags"])
            self.assertTrue(os.path.exists(r1["tracked_landmarks_csv_path"]))
            self.assertIsNone(r1["seed_image_path"])  # no video -> no render
            # Swapping the seed swaps the sides, not the data.
            self.assertEqual(r1["metrics"]["per_side"]["left"]["strikes_detected"],
                             r2["metrics"]["per_side"]["right"]["strikes_detected"])
            with open(q) as fh:
                self.assertTrue(json.load(fh)["leg_identity"]["swap_seed"])


if __name__ == "__main__":
    unittest.main()
