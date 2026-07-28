"""Phase 4 acceptance: every plan item traces to a root cause; every plan
has at least one falsifiable success criterion; prerequisites ordered."""

import unittest

from runform.plan import DRILL_LIBRARY, build_plan
from runform.session import interpret_session
from tests.synthetic import metrics_payload, session_clips


def assessment_with(cause_metrics):
    clips = session_clips({
        band: metrics_payload(**cause_metrics) for band in ("easy", "moderate", "fast")
    })
    return interpret_session(clips)


class TestPlan(unittest.TestCase):
    def test_every_item_traces_to_a_root_cause(self):
        a = assessment_with(dict(overstride=0.32, knee=176.0, cadence=155.0))
        plan = build_plan(a, created_at="2026-07-28")
        self.assertTrue(plan["items"])
        cause_ids = {c["cause"] for c in a["root_causes"]}
        for item in plan["items"]:
            self.assertIn(item["targets_root_cause"], cause_ids)

    def test_falsifiable_success_criteria(self):
        a = assessment_with(dict(overstride=0.32, knee=176.0, cadence=155.0))
        plan = build_plan(a, created_at="2026-07-28")
        self.assertTrue(plan["success_criteria"])
        for c in plan["success_criteria"]:
            self.assertIn(c["direction"], ("increase", "decrease"))
            self.assertGreater(c["target_delta"], 0)
            self.assertIsNotNone(c["baseline"])
        self.assertEqual(plan["recheck_due_date"], "2026-08-25")  # +4 weeks

    def test_prerequisites_come_first(self):
        # Limited hip extension pulls in hill_strides, whose prerequisite
        # is hip_flexor_mobility — mobility must appear earlier.
        a = assessment_with(dict(hip_ext=140.0))
        plan = build_plan(a, created_at="2026-07-28")
        order = [i["drill"] for i in plan["items"]]
        self.assertIn("hill_strides", order)
        self.assertIn("hip_flexor_mobility", order)
        self.assertLess(order.index("hip_flexor_mobility"), order.index("hill_strides"))

    def test_clean_session_gets_honest_maintenance_plan(self):
        a = assessment_with({})
        plan = build_plan(a, created_at="2026-07-28")
        self.assertEqual(plan["items"], [])
        self.assertIn("note", plan)
        self.assertTrue(plan["recheck_due_date"])  # re-film still scheduled

    def test_drill_library_is_wellformed(self):
        from runform.root_causes import CAUSES
        for d in DRILL_LIBRARY:
            self.assertTrue(d["targets"], f"{d['id']} targets nothing")
            for cause in d["targets"]:
                self.assertIn(cause, CAUSES, f"{d['id']} targets unknown cause {cause}")


if __name__ == "__main__":
    unittest.main()
