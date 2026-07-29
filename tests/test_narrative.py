"""Phase 3 constraints, enforced in code and tested with an injected fake
LLM (no Ollama needed): citation post-validation, retry on invalid
output, disclaimer in 100% of outputs, pain routing."""

import json
import unittest

from runform.errors import NarrativeError
from runform.narrative import DISCLAIMER, PAIN_ROUTING, narrate
from runform.session import interpret_session
from tests.synthetic import metrics_payload, session_clips


def make_assessment(notes=""):
    m = lambda: metrics_payload(overstride=0.32, knee=176.0, cadence=155.0)
    clips = session_clips({"easy": m(), "moderate": m(), "fast": m()})
    return interpret_session(clips, notes=notes)


def valid_response():
    return json.dumps({
        "summary": "Solid session overall with one clear pattern to work on.",
        "issues": [{
            "title": "Overstriding",
            "root_cause": "overstride",
            "explanation": "Your foot lands well ahead of your hips "
                           "(overstride_ratio) with a nearly straight knee "
                           "(knee_angle_at_strike_deg), and cadence_spm is low.",
            "metrics_cited": ["overstride_ratio", "knee_angle_at_strike_deg", "cadence_spm"],
        }],
        "next_focus": "Raise cadence a few percent with a metronome.",
    })


class TestNarrative(unittest.TestCase):
    def test_valid_output_accepted_and_disclaimed(self):
        out = narrate(make_assessment(), generate=lambda p: valid_response())
        self.assertEqual(out["attempts"], 1)
        self.assertIn(DISCLAIMER, out["narrative"])
        self.assertIn("overstride_ratio", out["narrative"])
        self.assertTrue(out["prompt_version"])

    def test_invented_metric_rejected_then_retried(self):
        calls = []

        def flaky(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                bad = json.loads(valid_response())
                bad["issues"][0]["metrics_cited"] = ["flight_time_ms"]  # invented
                return json.dumps(bad)
            return valid_response()

        out = narrate(make_assessment(), generate=flaky)
        self.assertEqual(out["attempts"], 2)
        # The retry prompt must tell the model what was wrong.
        self.assertIn("flight_time_ms", calls[1])

    def test_persistent_garbage_raises(self):
        with self.assertRaises(NarrativeError):
            narrate(make_assessment(), generate=lambda p: "not json at all")

    def test_diagnosis_language_rejected(self):
        bad = json.loads(valid_response())
        bad["summary"] = "I diagnose you with runner's knee."
        with self.assertRaises(NarrativeError):
            narrate(make_assessment(), generate=lambda p: json.dumps(bad))

    def test_issue_cap_enforced(self):
        bad = json.loads(valid_response())
        bad["issues"] = bad["issues"] * 4  # 4 > MAX_ISSUES
        with self.assertRaises(NarrativeError):
            narrate(make_assessment(), generate=lambda p: json.dumps(bad))

    def test_pain_routes_to_professional(self):
        out = narrate(
            make_assessment(notes="some shin pain this week"),
            generate=lambda p: valid_response(),
        )
        self.assertTrue(out["narrative"].startswith(PAIN_ROUTING))


if __name__ == "__main__":
    unittest.main()
