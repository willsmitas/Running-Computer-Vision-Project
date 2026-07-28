"""Phase 2: confidence gating. Grade every metric high / low / unusable
BEFORE interpretation, from n, SD, and tracking quality.

Grade meanings:
    high     — usable for critique and for longitudinal comparison
    low      — reportable with hedging; never the sole basis of a root cause
    unusable — excluded from interpretation entirely (and said so, honestly)

This is why metrics report {mean, sd, n} and never a bare mean
(CLAUDE.md): SD and n are what decide whether the mean is usable at all.
All thresholds provisional until real footage tunes them.
"""

from .references import PER_SIDE_METRICS

GRADES = ("high", "low", "unusable")

# ~8 events means ~4 strides per side of a periodic signal — about the
# minimum for a mean anyone should act on.
MIN_N_HIGH = 8
# Below this the SD itself is meaningless and the mean is an anecdote.
MIN_N_USABLE = 4
# Cadence needs enough strikes that the median interval is stable.
MIN_STEPS_HIGH = 8
# Below this fraction of frames tracked, positions are heavily
# interpolated; cap every grade at "low".
MIN_DETECTION_RATE = 0.8

# Per-metric absolute SD caps: roughly the point where stride-to-stride
# spread rivals the width of the reference range itself. SD above the cap
# demotes to "low"; above 2x the cap the mean is unusable.
SD_CAPS = {
    "knee_angle_at_strike_deg": 12.0,
    "hip_angle_at_toeoff_deg": 12.0,
    "ground_contact_ms": 60.0,
    "overstride_ratio": 0.08,
    "trunk_lean_deg": 6.0,
}


def grade_key(metric, side=None, submetric=None):
    """The key a deviation looks up its grade under."""
    if metric == "asymmetry_pct" and submetric:
        return f"asymmetry_pct.{submetric}"
    if side:
        return f"{side}.{metric}"
    return metric


def _grade_summary(metric, summary):
    if not summary:
        return "unusable"
    n, sd = summary.get("n", 0), summary.get("sd")
    if n < MIN_N_USABLE:
        return "unusable"
    cap = SD_CAPS.get(metric)
    if cap is not None and sd is not None:
        if sd > 2 * cap:
            return "unusable"
        if sd > cap:
            return "low"
    return "high" if n >= MIN_N_HIGH else "low"


def grade_clip_metrics(metrics, detection_rate=None):
    """Grade every metric in one clip's metrics dict.

    Returns {grade_key: grade}. Keys follow grade_key(): scalars by name,
    per-side as "left.metric"/"right.metric", asymmetry as
    "asymmetry_pct.submetric".
    """
    grades = {}

    steps = metrics.get("steps_detected") or 0
    if steps < MIN_N_USABLE:
        grades["cadence_spm"] = "unusable"
    else:
        grades["cadence_spm"] = "high" if steps >= MIN_STEPS_HIGH else "low"

    for side, table in (metrics.get("per_side") or {}).items():
        for name in PER_SIDE_METRICS:
            grades[grade_key(name, side=side)] = _grade_summary(name, table.get(name))

    vo = metrics.get("vertical_oscillation_ratio")
    grades["vertical_oscillation_ratio"] = "unusable" if vo is None else "high"

    # Asymmetry is gated hardest of all metrics (risk register: a
    # single-side camera view can fabricate left/right differences).
    # It is NEVER graded high, and it is usable at all only when both
    # sides of the underlying metric are themselves high-grade.
    for sub in (metrics.get("asymmetry_pct") or {}):
        both_high = all(
            grades.get(grade_key(sub, side=s)) == "high" for s in ("left", "right")
        )
        grades[grade_key("asymmetry_pct", submetric=sub)] = (
            "low" if both_high else "unusable"
        )

    if detection_rate is not None and detection_rate < MIN_DETECTION_RATE:
        grades = {k: ("low" if g == "high" else g) for k, g in grades.items()}

    return grades
