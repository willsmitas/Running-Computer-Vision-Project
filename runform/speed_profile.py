"""Phase 2: speed-profile derivation across the session's clips.

A single clip gives a point; three clips give a slope. How a fault
responds to pace distinguishes fault classes that look identical at one
speed (BUILD_PLAN):

    constant             -> ingrained motor pattern; responds to cueing
    degrades_with_speed  -> strength/mobility/coordination ceiling;
                            responds to strength work, not cueing
    slow_only            -> often benign, or a low-cadence artifact

Prescriptions differ sharply between those classes, so the classification
lives here in code, deterministically — the LLM only explains it.
"""

import numpy as np

from .references import SPEED_BANDS


def fit(points):
    """Least-squares line through [(speed_value, metric_mean), ...].

    Returns {slope, intercept, r2, n} or None with fewer than 2 points.
    With only 3 clips per session the fit is descriptive, not inferential
    — r2 says how line-like the response is, nothing more.
    """
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        # All y equal: a flat line fits them exactly.
        r2 = 1.0
    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r2": round(r2, 4),
        "n": len(points),
    }


def classify(flagged_bands, present_bands):
    """Classify a metric's fault pattern across pace bands.

    flagged_bands: bands where the metric deviated from reference.
    present_bands: bands the session actually has clips for.
    """
    order = [b for b in SPEED_BANDS if b in present_bands]
    flagged = set(flagged_bands) & set(order)

    if not flagged:
        return "absent"
    if len(order) < 2:
        # One clip cannot support any speed-response claim. Be honest
        # about that rather than defaulting to "constant".
        return "single_speed"
    if flagged == set(order):
        return "constant"
    # Contiguous run of the fastest bands -> worsens as pace rises.
    for k in range(1, len(order)):
        if flagged == set(order[-k:]):
            return "degrades_with_speed"
    if flagged == {order[0]}:
        return "slow_only"
    # e.g. flagged only at moderate: no coherent speed story. Say so.
    return "inconsistent"
