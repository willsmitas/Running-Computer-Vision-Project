"""Phase 4: prescription. Drill library tagged to ROOT CAUSES (never to
symptoms), assembled into a time- and volume-bounded plan with explicit,
falsifiable success criteria and a re-film date.

Cue style: prefer actionable external/rhythm cues ("match the metronome")
over descriptive corrections ("stop overstriding") — better motor-learning
outcomes (BUILD_PLAN).
"""

from datetime import date, timedelta

from . import root_causes

# A plan targets at most this many root causes at once. Motor-learning
# bandwidth is the constraint: a runner reliably holds one or two focus
# points, not five. Mirrors the 2-3 issue cap in the narrative layer.
MAX_ROOT_CAUSES_PER_PLAN = 2

# Success criterion = close this fraction of the measured deviation by
# the recheck date. Provisional until test-retest data establishes the
# real noise floor; asking for 100% in 4 weeks would just manufacture
# failure, and tiny targets would manufacture success.
TARGET_FRACTION_OF_DEVIATION = 0.5

DEFAULT_PLAN_WEEKS = 4

# Each entry: what it is, the cue the runner actually hears, dosage,
# which root causes it addresses, how to progress, and prerequisites
# (mobility before loading, posture before speed).
DRILL_LIBRARY = [
    {
        "id": "metronome_cadence",
        "name": "Metronome cadence runs",
        "targets": ("overstride", "low_cadence"),
        "cue": "Match your foot strikes to a metronome set 5% above your current cadence.",
        "dosage": "3 x 3 min within an easy run, full recovery between",
        "frequency_per_week": 3,
        "weeks": 4,
        "progression_rule": {
            "advance": "held the metronome for two consecutive runs -> raise it 2%",
            "hold": "within 3 spm of the metronome but not consistent -> repeat the week",
            "regress": "cannot hold it without straining -> lower the metronome 2%",
        },
        "prerequisites": (),
    },
    {
        "id": "wall_lean_posture",
        "name": "Wall lean (falling start) drill",
        "targets": ("trunk_posture", "overstride"),
        "cue": "Fall forward from the ankles, one straight line from head to heel, and step out of the fall.",
        "dosage": "2 x 5 reps before each run",
        "frequency_per_week": 3,
        "weeks": 3,
        "progression_rule": {
            "advance": "clean line in all reps -> step out into 10 m accelerations",
            "hold": "line breaks at the hip on some reps -> repeat the week",
            "regress": "line breaks every rep -> shorten the lean, hands on wall",
        },
        "prerequisites": (),
    },
    {
        "id": "hip_flexor_mobility",
        "name": "Half-kneeling hip flexor mobilization",
        "targets": ("limited_hip_extension",),
        "cue": "In half-kneeling, squeeze the back-leg glute and push the hip gently forward — no low-back arch.",
        "dosage": "2 x 45 s per side",
        "frequency_per_week": 5,
        "weeks": 4,
        "progression_rule": {
            "advance": "no stretch sensation at neutral -> add overhead reach",
            "hold": "stretch still strong -> repeat the week",
            "regress": "pinching in the front of the hip -> reduce range; if it persists, see a professional",
        },
        "prerequisites": (),
    },
    {
        "id": "hill_strides",
        "name": "Short uphill strides",
        "targets": ("limited_hip_extension",),
        "cue": "8-10 s uphill, drive the ground away behind you; tall hips.",
        "dosage": "6 x 10 s on a moderate hill, walk-down recovery",
        "frequency_per_week": 2,
        "weeks": 4,
        "progression_rule": {
            "advance": "posture holds for all reps -> add 2 reps, then steepen",
            "hold": "posture degrades on the last reps -> keep the dose",
            "regress": "posture degrades from the first rep -> flatter hill",
        },
        "prerequisites": ("hip_flexor_mobility",),
    },
    {
        "id": "soft_landings",
        "name": "Quiet-feet running",
        "targets": ("excessive_vertical_oscillation",),
        "cue": "Run so quietly you cannot hear your own footfalls.",
        "dosage": "4 x 2 min within an easy run",
        "frequency_per_week": 3,
        "weeks": 3,
        "progression_rule": {
            "advance": "quiet at easy pace -> apply the cue at moderate pace",
            "hold": "quiet only with full attention -> repeat the week",
            "regress": "cannot soften without over-slowing -> pair with metronome work first",
        },
        "prerequisites": (),
    },
    {
        "id": "single_leg_control",
        "name": "Single-leg step-downs",
        "targets": ("left_right_asymmetry",),
        "cue": "Lower slowly off a low step, pelvis level the whole way down.",
        "dosage": "3 x 8 per side",
        "frequency_per_week": 3,
        "weeks": 4,
        "progression_rule": {
            "advance": "level pelvis all reps -> raise the step height",
            "hold": "pelvis drops on late reps -> keep the dose",
            "regress": "pelvis drops immediately -> lower step, hold support",
        },
        "prerequisites": (),
    },
]

_DRILLS_BY_ID = {d["id"]: d for d in DRILL_LIBRARY}


def _drills_for(cause_id):
    return [d for d in DRILL_LIBRARY if cause_id in d["targets"]]


def _ordered_with_prerequisites(selected):
    """Prerequisites first, then dependents; stable otherwise."""
    ordered, placed = [], set()
    pending = list(selected)
    while pending:
        progressed = False
        for d in list(pending):
            if all(p in placed for p in d["prerequisites"]):
                ordered.append(d)
                placed.add(d["id"])
                pending.remove(d)
                progressed = True
        if not progressed:  # unmet prereq outside selection: append as-is
            ordered.extend(pending)
            break
    return ordered


def build_plan(assessment, created_at=None, duration_weeks=DEFAULT_PLAN_WEEKS):
    """Deterministic plan from a session assessment payload.

    Every item traces to a root cause, and every targeted cause gets at
    least one falsifiable success criterion (metric, direction, how much,
    by when) — that is the Phase 4 acceptance bar, enforced by shape.
    """
    created = date.fromisoformat(created_at) if created_at else date.today()
    causes = assessment.get("root_causes", [])[:MAX_ROOT_CAUSES_PER_PLAN]

    items, seen = [], set()
    for cause in causes:
        selected = [d for d in _drills_for(cause["cause"]) if d["id"] not in seen]
        # Pull in prerequisites that target the same work.
        for d in list(selected):
            for pid in d["prerequisites"]:
                if pid not in seen and all(pid != s["id"] for s in selected):
                    selected.append(_DRILLS_BY_ID[pid])
        for d in _ordered_with_prerequisites(selected):
            seen.add(d["id"])
            items.append({
                "drill": d["id"],
                "name": d["name"],
                "cue": d["cue"],
                "dosage": d["dosage"],
                "frequency_per_week": d["frequency_per_week"],
                "weeks": d["weeks"],
                "targets_root_cause": cause["cause"],
                "progression_rule": d["progression_rule"],
                "prerequisites": list(d["prerequisites"]),
            })

    # Falsifiable success criteria: derived from the worst usable
    # deviation on each targeted cause's primary metrics.
    criteria = []
    for cause in causes:
        primary = set(root_causes.CAUSES[cause["cause"]]["primary"])
        devs = [d for d in assessment.get("deviations", []) if d["metric"] in primary]
        if not devs:
            continue
        worst = max(devs, key=lambda d: abs(d["deviation"]))
        criteria.append({
            "root_cause": cause["cause"],
            "metric": worst["metric"],
            "speed_label": worst["speed_label"],
            "baseline": worst["value"],
            "direction": "decrease" if worst["deviation"] > 0 else "increase",
            "target_delta": round(abs(worst["deviation"]) * TARGET_FRACTION_OF_DEVIATION, 4),
            "by_week": duration_weeks,
        })

    plan = {
        "created_at": created.isoformat(),
        "duration_weeks": duration_weeks,
        "items": items,
        "success_criteria": criteria,
        # The longitudinal loop: a plan without a scheduled re-film is
        # just advice. This date drives the Phase 5 comparison.
        "recheck_due_date": (created + timedelta(weeks=duration_weeks)).isoformat(),
    }
    if not causes:
        plan["note"] = (
            "No high-confidence form faults this session. Keep current "
            "training and re-film on the recheck date to build the trend."
        )
    return plan
