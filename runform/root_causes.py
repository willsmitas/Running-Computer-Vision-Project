"""Phase 2: root-cause collapse. Map correlated symptoms to underlying
causes and rank by breadth of effect — all deterministic, all here.

Overstriding, for example, simultaneously produces a large overstride
ratio, a straight knee at contact, long ground contact, and often low
cadence. Reporting those as four findings would read as four problems;
collapsing them into one cause with four supporting metrics is both more
truthful and what makes the downstream prescription coherent.
"""

# Cause definitions. "primary" metrics are the ones whose deviation can
# fire the cause; "supporting" deviations add breadth (score) but cannot
# fire it alone. The mapping itself is provisional coaching knowledge —
# validate against blind coach comparison (BUILD_PLAN).
CAUSES = {
    "overstride": {
        "label": "Overstriding",
        "primary": ("overstride_ratio", "knee_angle_at_strike_deg"),
        "supporting": ("cadence_spm", "ground_contact_ms"),
    },
    "low_cadence": {
        "label": "Low cadence",
        "primary": ("cadence_spm",),
        "supporting": ("overstride_ratio", "vertical_oscillation_ratio"),
    },
    "limited_hip_extension": {
        "label": "Limited hip extension",
        "primary": ("hip_angle_at_toeoff_deg",),
        "supporting": ("trunk_lean_deg",),
    },
    "excessive_vertical_oscillation": {
        "label": "Excessive vertical oscillation",
        "primary": ("vertical_oscillation_ratio",),
        "supporting": ("cadence_spm", "ground_contact_ms"),
    },
    "trunk_posture": {
        "label": "Trunk posture",
        "primary": ("trunk_lean_deg",),
        "supporting": (),
    },
    "left_right_asymmetry": {
        "label": "Left/right asymmetry",
        "primary": ("asymmetry_pct",),
        "supporting": (),
    },
}


def collapse(deviations, classifications=None):
    """Collapse a session's (usable, graded) deviations into ranked causes.

    deviations: deviation dicts (references.clip_deviations shape) with a
    "grade" attached by the session layer. Unusable-grade deviations must
    already be filtered out.

    Ranking: breadth first — a cause explaining more distinct flagged
    metrics outranks a narrower one. A cause whose flagged metrics are
    entirely explained by higher-ranked causes is dropped (that is the
    collapse: its evidence was somebody else's symptom set all along).

    Confidence: "high" only if every flagged primary metric carries at
    least one high-grade deviation; anything resting on low-grade
    evidence is "low". No high-confidence causes should fire on a clean,
    well-tracked clip set — that is the Phase 2 acceptance criterion.
    """
    classifications = classifications or {}

    flagged = {}
    for d in deviations:
        flagged.setdefault(d["metric"], []).append(d)
    if not flagged:
        return []

    grade_for = {
        m: ("high" if any(d.get("grade") == "high" for d in ds) else "low")
        for m, ds in flagged.items()
    }

    candidates = []
    for name, spec in CAUSES.items():
        primary_hit = [m for m in spec["primary"] if m in flagged]
        if not primary_hit:
            continue
        all_hit = [m for m in spec["primary"] + spec["supporting"] if m in flagged]
        confidence = "high" if all(grade_for[m] == "high" for m in primary_hit) else "low"
        candidates.append({
            "cause": name,
            "label": spec["label"],
            "confidence": confidence,
            "score": len(all_hit),
            "primary_metrics": primary_hit,
            "metrics": all_hit,
        })

    candidates.sort(key=lambda c: (-c["score"], -len(c["primary_metrics"]), c["cause"]))

    ranked, explained = [], set()
    for cand in candidates:
        if not set(cand["metrics"]) - explained:
            continue  # fully absorbed by higher-ranked causes
        explained.update(cand["metrics"])
        speed_class = next(
            (classifications[m] for m in cand["primary_metrics"] if m in classifications),
            None,
        )
        cand["speed_classification"] = speed_class
        ranked.append(cand)

    return ranked
