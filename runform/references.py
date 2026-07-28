"""Phase 2: reference ranges per metric, conditioned on pace band, plus
the helper that turns one clip's metrics dict into a list of deviations.

Deciding what is normal happens HERE, in code — never in the LLM
(architectural rule: deterministic first, LLM last).

Every number in this table is an educated guess, untuned against real
footage (CLAUDE.md). Treat as provisional. The validation path is blind
coach comparison and planted-fault clips (BUILD_PLAN, Validation
strategy) — update these ranges from that evidence, not from vibes.

ANGLE CONVENTION: metrics.joint_angle returns the INTERIOR angle
(straight leg = 180°). Sports-science literature reports flexion
= 180 − interior. Every published number must be converted before it
lands in this table, and the ranges below are already in interior form.
"""

# The three-speed protocol (BUILD_PLAN): ranges are conditioned on which
# of the runner's three repeatable speeds a clip was filmed at. Absolute
# speed values matter less than the runner repeating the same three, so
# v1 conditions on the band label. Revisit once real per-pace data exists.
SPEED_BANDS = ("easy", "moderate", "fast")

# metric -> band -> (low, high): the interval a value "should" fall in.
REFERENCE_RANGES = {
    # steps/min. Slower paces legitimately run lower cadence; the classic
    # "180" is a story about ceilings, not a floor for easy running.
    "cadence_spm": {
        "easy": (158.0, 182.0),
        "moderate": (164.0, 186.0),
        "fast": (170.0, 194.0),
    },
    # Interior knee angle at foot strike. Near-180 = landing on a straight
    # leg, the classic overstride signature. Literature flexion ~10-30° at
    # contact -> interior 150-170.
    "knee_angle_at_strike_deg": {
        "easy": (148.0, 172.0),
        "moderate": (148.0, 170.0),
        "fast": (145.0, 168.0),
    },
    # Interior shoulder-hip-knee angle at toe-off. Larger = fuller hip
    # extension; faster paces demand more of it.
    "hip_angle_at_toeoff_deg": {
        "easy": (150.0, 185.0),
        "moderate": (155.0, 185.0),
        "fast": (160.0, 185.0),
    },
    # Ground contact time in ms; drops as pace rises.
    "ground_contact_ms": {
        "easy": (220.0, 330.0),
        "moderate": (200.0, 300.0),
        "fast": (160.0, 260.0),
    },
    # Ankle ahead of hip at strike / leg length. Some forward placement is
    # normal; large values (with a straight knee) indicate overstriding.
    "overstride_ratio": {
        "easy": (0.00, 0.24),
        "moderate": (0.00, 0.22),
        "fast": (0.00, 0.20),
    },
    # Forward trunk lean from vertical, degrees. Slight forward lean is
    # desirable; bolt-upright/backward lean and excessive folding both flag.
    "trunk_lean_deg": {
        "easy": (1.0, 12.0),
        "moderate": (1.0, 12.0),
        "fast": (2.0, 14.0),
    },
    # Hip vertical peak-to-peak / leg length.
    "vertical_oscillation_ratio": {
        "easy": (0.02, 0.11),
        "moderate": (0.02, 0.10),
        "fast": (0.02, 0.10),
    },
}

# Left/right relative difference (%) above which asymmetry flags. This is
# the hardest-gated metric in the system (see gating.py): a single-side
# camera view can fabricate asymmetry (risk register), so the threshold
# is deliberately generous.
ASYMMETRY_MAX_PCT = 12.0

# The per-side summary metrics inside metrics["per_side"][side].
PER_SIDE_METRICS = (
    "knee_angle_at_strike_deg",
    "hip_angle_at_toeoff_deg",
    "ground_contact_ms",
    "overstride_ratio",
    "trunk_lean_deg",
)


def expected_range(metric, band):
    """(low, high) for a metric at a pace band, or None if unknown."""
    return REFERENCE_RANGES.get(metric, {}).get(band)


def iter_metric_values(metrics):
    """Walk a clip metrics dict, yielding (metric, side, value, summary).

    `summary` is the {mean, sd, n} dict for per-side metrics, or None for
    scalar metrics (cadence, vertical oscillation). Skips missing values.
    Used by both deviation detection and Phase 5 comparison so the two
    layers can never disagree about what a "metric" is.
    """
    cadence = metrics.get("cadence_spm")
    if cadence is not None:
        yield "cadence_spm", None, float(cadence), None

    for side, table in (metrics.get("per_side") or {}).items():
        for name in PER_SIDE_METRICS:
            summary = table.get(name)
            if summary:
                yield name, side, float(summary["mean"]), summary

    vo = metrics.get("vertical_oscillation_ratio")
    if vo is not None:
        yield "vertical_oscillation_ratio", None, float(vo), None


def clip_deviations(metrics, band):
    """All out-of-range values in one clip, at the given pace band.

    Each deviation dict:
        metric      base metric name
        side        "left"/"right" for per-side metrics, else None
        submetric   set only for asymmetry entries
        speed_label pace band the range came from
        value       observed value (mean, for per-side metrics)
        expected    [low, high] reference interval
        deviation   signed distance beyond the interval (0 never emitted)
        sd, n       spread/support where available (None for scalars)
    """
    devs = []
    for metric, side, value, summary in iter_metric_values(metrics):
        rng = expected_range(metric, band)
        if rng is None:
            continue
        lo, hi = rng
        if lo <= value <= hi:
            continue
        deviation = value - lo if value < lo else value - hi
        devs.append({
            "metric": metric,
            "side": side,
            "submetric": None,
            "speed_label": band,
            "value": value,
            "expected": [lo, hi],
            "deviation": round(deviation, 4),
            "sd": summary["sd"] if summary else None,
            "n": summary["n"] if summary else None,
        })

    for sub, pct in (metrics.get("asymmetry_pct") or {}).items():
        if pct is None or pct <= ASYMMETRY_MAX_PCT:
            continue
        devs.append({
            "metric": "asymmetry_pct",
            "side": None,
            "submetric": sub,
            "speed_label": band,
            "value": float(pct),
            "expected": [0.0, ASYMMETRY_MAX_PCT],
            "deviation": round(float(pct) - ASYMMETRY_MAX_PCT, 4),
            "sd": None,
            "n": None,
        })

    return devs
