"""Phase 5: session-over-session comparison — the actual product.

Single-camera pose estimation carries systematic error, but that error
largely cancels when the same runner is filmed the same way, so change
detection is more trustworthy than absolute assessment (BUILD_PLAN).

The two honesty mechanisms, both deterministic:
  - Significance gate: within-session SD across strides is the noise
    floor. A delta smaller than it is not a result and is reported as
    "no measurable change" — never dressed up as progress.
  - Setup-drift check: if the runner's on-screen scale or the clip speed
    changed materially between sessions, deltas are flagged as suspect
    before anyone reads meaning into them.
"""

from . import references

# Delta must exceed multiplier x noise floor to count as real. 1.0 is the
# BUILD_PLAN minimum ("a delta smaller than that is not a result"); raise
# it once test-retest clips establish the empirical floor.
SIGNIFICANCE_MULTIPLIER = 1.0

# Scalar metrics have no per-stride SD, so they get fixed floors.
# Cadence: +-3 spm matches the Phase 0 manual-count acceptance tolerance.
CADENCE_MIN_DELTA_SPM = 3.0
# Vertical oscillation ratio floor: provisional, ~10% of the range width.
VO_MIN_DELTA = 0.01

# Relative change in leg_length_units (body scale on screen) above which
# the camera distance/framing must have changed -> setup drift.
LEG_LENGTH_DRIFT_TOL = 0.10

# Same-label clips must be within this relative speed difference,
# otherwise the "comparison" is between different efforts.
SPEED_MATCH_TOL = 0.05


def _range_distance(value, rng):
    """How far outside the reference interval a value sits (0 = inside)."""
    lo, hi = rng
    return max(lo - value, value - hi, 0.0)


def _verdict(before, after, delta, significant, band, metric):
    if not significant:
        return "no_measurable_change"
    rng = references.expected_range(metric, band)
    if rng is None:
        return "changed"
    b, a = _range_distance(before, rng), _range_distance(after, rng)
    if a < b:
        return "improved_toward_range"
    if a > b:
        return "moved_away_from_range"
    return "changed_within_range"


def compare_sessions(before, after):
    """Compare two session assessment payloads (session.interpret_session
    output). Clips are matched by speed_label; per-side metrics compare
    side to side. Returns a JSON-safe comparison dict."""
    labels = [
        band for band in references.SPEED_BANDS
        if band in before["metrics_by_clip"] and band in after["metrics_by_clip"]
    ]

    deltas, drift_flags = [], []

    before_clips = {c["speed_label"]: c for c in before["session"]["clips"]}
    after_clips = {c["speed_label"]: c for c in after["session"]["clips"]}

    for band in labels:
        # --- setup drift gates -------------------------------------------
        bl = (before_clips.get(band) or {}).get("leg_length_units")
        al = (after_clips.get(band) or {}).get("leg_length_units")
        if bl and al and abs(al - bl) / bl > LEG_LENGTH_DRIFT_TOL:
            drift_flags.append({
                "speed_label": band, "kind": "scale_change",
                "before": bl, "after": al,
                "note": "On-screen body scale changed materially — camera "
                        "distance or framing drifted. Deltas at this speed "
                        "are suspect.",
            })
        bs = (before_clips.get(band) or {}).get("speed_value")
        as_ = (after_clips.get(band) or {}).get("speed_value")
        if bs and as_ and abs(as_ - bs) / bs > SPEED_MATCH_TOL:
            drift_flags.append({
                "speed_label": band, "kind": "speed_mismatch",
                "before": bs, "after": as_,
                "note": "Clip speeds differ — this compares different "
                        "efforts, not form change.",
            })

        # --- metric deltas ------------------------------------------------
        b_vals = {
            (m, s): (v, summ)
            for m, s, v, summ in references.iter_metric_values(before["metrics_by_clip"][band])
        }
        a_vals = {
            (m, s): (v, summ)
            for m, s, v, summ in references.iter_metric_values(after["metrics_by_clip"][band])
        }

        for key in sorted(b_vals.keys() & a_vals.keys(), key=str):
            metric, side = key
            (bv, bsum), (av, asum) = b_vals[key], a_vals[key]
            delta = av - bv

            if bsum and asum:
                # Within-session stride-to-stride SD is the noise floor.
                floor = max(bsum["sd"], asum["sd"]) * SIGNIFICANCE_MULTIPLIER
            elif metric == "cadence_spm":
                floor = CADENCE_MIN_DELTA_SPM
            elif metric == "vertical_oscillation_ratio":
                floor = VO_MIN_DELTA
            else:
                floor = None

            significant = floor is not None and abs(delta) > floor
            deltas.append({
                "metric": metric,
                "side": side,
                "speed_label": band,
                "before": bv,
                "after": av,
                "delta": round(delta, 4),
                "noise_floor": round(floor, 4) if floor is not None else None,
                "significant": significant,
                "verdict": _verdict(bv, av, delta, significant, band, metric),
            })

    verdict_counts = {}
    for d in deltas:
        verdict_counts[d["verdict"]] = verdict_counts.get(d["verdict"], 0) + 1

    return {
        "labels_compared": labels,
        "metric_deltas": deltas,
        "setup_drift": drift_flags,
        "verdict_counts": verdict_counts,
        # Self-reported adherence gets attached by the caller when known;
        # progression decisions (advance/hold/regress) need it.
        "plan_adherence": None,
    }
