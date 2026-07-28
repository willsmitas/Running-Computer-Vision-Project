"""Phase 2 acceptance criteria (BUILD_PLAN):
  - a deliberately overstriding clip set surfaces overstride as top root
    cause with correct supporting metrics;
  - a clean clip set fires no high-confidence faults.
"""

import unittest

from runform.errors import RunFormError
from runform.session import interpret_session, pain_mentioned
from tests.synthetic import metrics_payload, session_clips


def overstrider_clips():
    # Overstride ratio far out of range, knee too straight at contact,
    # cadence below range — at every speed (an ingrained pattern).
    m = lambda: metrics_payload(overstride=0.32, knee=176.0, cadence=155.0, steps=24)
    return session_clips({"easy": m(), "moderate": m(), "fast": m()})


def clean_clips():
    m = lambda: metrics_payload()
    return session_clips({"easy": m(), "moderate": m(), "fast": m()})


class TestInterpretSession(unittest.TestCase):
    def test_planted_overstride_surfaces_as_top_cause(self):
        out = interpret_session(overstrider_clips())
        causes = out["root_causes"]
        self.assertTrue(causes, "no root causes fired on a planted fault")
        top = causes[0]
        self.assertEqual(top["cause"], "overstride")
        self.assertEqual(top["confidence"], "high")
        self.assertIn("overstride_ratio", top["metrics"])
        self.assertIn("knee_angle_at_strike_deg", top["metrics"])
        self.assertIn("cadence_spm", top["metrics"])
        # Present at all three speeds -> ingrained pattern -> cueing.
        self.assertEqual(top["speed_classification"], "constant")
        # Correlated symptom collapse: low_cadence is absorbed, not listed.
        self.assertNotIn("low_cadence", [c["cause"] for c in causes])

    def test_clean_session_fires_nothing_high_confidence(self):
        out = interpret_session(clean_clips())
        high = [c for c in out["root_causes"] if c["confidence"] == "high"]
        self.assertEqual(high, [])
        self.assertEqual(out["deviations"], [])

    def test_speed_fits_computed(self):
        out = interpret_session(clean_clips())
        self.assertIn("cadence_spm", out["speed_profile"]["fits"])
        self.assertEqual(out["speed_profile"]["fits"]["cadence_spm"]["n"], 3)

    def test_payload_carries_llm_contract_fields(self):
        out = interpret_session(overstrider_clips(), notes="new shoes")
        self.assertIn("allowed_metrics", out)
        self.assertIn("overstride_ratio", out["allowed_metrics"])
        self.assertFalse(out["session"]["pain_flag"])

    def test_pain_in_notes_sets_flag(self):
        out = interpret_session(clean_clips(), notes="slight knee pain after long runs")
        self.assertTrue(out["session"]["pain_flag"])
        self.assertTrue(pain_mentioned("Achilles has been sore"))
        self.assertFalse(pain_mentioned("felt great, new shoes"))

    def test_rejects_bad_sessions(self):
        with self.assertRaises(RunFormError):
            interpret_session([])
        clips = clean_clips()[:2]
        clips[1]["speed_label"] = clips[0]["speed_label"]  # duplicate band
        with self.assertRaises(RunFormError):
            interpret_session(clips)
        with self.assertRaises(RunFormError):
            interpret_session([{"speed_label": "sprint", "metrics": metrics_payload()}])


if __name__ == "__main__":
    unittest.main()
