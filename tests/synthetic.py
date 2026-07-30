"""Synthetic ground-truth generators (CLAUDE.md testing strategy).

Two levels:
  - write_gait_csv(): a landmarks CSV with KNOWN cadence, overstride
    amplitude, and bounce, for validating the signal-processing layer
    without any video.
  - metrics_payload(): a metrics dict with chosen values, for unit-testing
    the Phase 2+ layers (gating, root causes, comparison) directly.

Synthetic tests validate logic, NOT real-world behavior — the known open
risk (pose-backend left/right leg swap at crossover) is untouched by any of
this and remains a Phase 0 gate on real footage.
"""

import numpy as np
import pandas as pd

from runform.metrics import REQUIRED


def make_gait_frames(
    fps=30.0,
    seconds=10.0,
    cadence_spm=180.0,
    swing_amp=0.08,      # ankle-vs-hip horizontal amplitude, normalized units
    bounce_p2p=0.03,     # hip vertical peak-to-peak
    direction=1,         # +1 faces screen-right, -1 screen-left
    noise=0.0,
    stationary=False,
    occlude=None,        # (side, start_frame, end_frame): drop that leg's
                         # visibility, as the far leg really does mid-clip
    seed=0,
):
    """Kinematically simple runner: sinusoidal ankle swing about the hip,
    legs in antiphase, slight forward trunk lean, slightly bent knee.

    Ground truth: foot strike = ankle's furthest-forward point, so strikes
    per foot = seconds * cadence/120, and overstride_ratio at strike is
    ~ swing_amp / leg_length (leg length ~0.40 units here).
    """
    rng = np.random.default_rng(seed)
    n = int(fps * seconds)
    t = np.arange(n) / fps

    hip_y0, ankle_y0 = 0.45, 0.85
    step_hz = cadence_spm / 60.0        # both feet
    foot_hz = cadence_spm / 120.0       # one foot

    cols = {"frame": np.arange(n)}

    def put(joint, x, y):
        cols[f"{joint}_x"] = x + rng.normal(0, noise, n) if noise else x * np.ones(n)
        cols[f"{joint}_y"] = y + rng.normal(0, noise, n) if noise else y * np.ones(n)
        cols[f"{joint}_z"] = np.zeros(n)
        cols[f"{joint}_vis"] = np.ones(n)

    hip_x = np.full(n, 0.5)
    hip_y = hip_y0 + (0.0 if stationary else (bounce_p2p / 2) * np.sin(2 * np.pi * step_hz * t))

    put("left_hip", hip_x, hip_y)
    put("right_hip", hip_x, hip_y)
    put("left_shoulder", hip_x + 0.03 * direction, np.full(n, 0.15))
    put("right_shoulder", hip_x + 0.03 * direction, np.full(n, 0.15))

    for side, phase0 in (("left", 0.0), ("right", np.pi)):
        phase = 2 * np.pi * foot_hz * t + phase0
        swing = 0.0 if stationary else swing_amp * np.sin(phase)
        ankle_x = hip_x + swing * direction
        ankle_y = ankle_y0 + (0.0 if stationary else 0.03 * np.cos(phase))
        knee_x = (hip_x + ankle_x) / 2 - 0.03 * direction  # slight bend
        knee_y = (hip_y0 + ankle_y0) / 2 * np.ones(n)
        put(f"{side}_ankle", ankle_x, ankle_y)
        put(f"{side}_knee", knee_x, knee_y)
        put(f"{side}_heel", ankle_x - 0.02 * direction, np.full(n, 0.95))
        put(f"{side}_foot_index", ankle_x + 0.03 * direction, np.full(n, 0.95))

    df = pd.DataFrame(cols)

    # Simulated occlusion: on real footage the leg farther from the camera
    # periodically drops below the visibility threshold, metrics.py masks
    # those samples to NaN, and gait events in that window go undetected.
    # That is what makes a strike pair with a toe-off from a LATER stride
    # unless the pairing is bounded.
    if occlude:
        side, start, end = occlude
        for joint in (f"{side}_ankle", f"{side}_knee", f"{side}_heel",
                      f"{side}_foot_index"):
            col = f"{joint}_vis"
            if col in df.columns:
                df.loc[start:end, col] = 0.2  # below metrics.VIS_THRESHOLD

    missing = [j for j in REQUIRED if f"{j}_x" not in df.columns]
    assert not missing, f"synthetic generator missing joints: {missing}"
    return df


def write_gait_csv(path, **kwargs):
    df = make_gait_frames(**kwargs)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Metric-dict level (Phase 2+ layers)
# ---------------------------------------------------------------------------

def _summary(mean, sd=2.0, n=12):
    return {"mean": round(float(mean), 2), "sd": float(sd), "n": int(n)}


def metrics_payload(
    cadence=172.0,
    steps=24,
    knee=162.0, knee_sd=3.0,
    hip_ext=168.0,
    contact=250.0, contact_sd=15.0,
    overstride=0.15, overstride_sd=0.02,
    lean=6.0,
    vo=0.07,
    n=12,
    asym=None,          # e.g. {"ground_contact_ms": 18.0}
    leg_length=0.40,
):
    """A well-formed metrics dict (metrics.compute_metrics shape) with
    identical left/right sides unless asym overrides are given."""
    def side_table():
        return {
            "knee_angle_at_strike_deg": _summary(knee, knee_sd, n),
            "hip_angle_at_toeoff_deg": _summary(hip_ext, 4.0, n),
            "ground_contact_ms": _summary(contact, contact_sd, n),
            "overstride_ratio": _summary(overstride, overstride_sd, n),
            "trunk_lean_deg": _summary(lean, 1.5, n),
            "strikes_detected": n,
        }

    payload = {
        "cadence_spm": float(cadence),
        "steps_detected": int(steps),
        "per_side": {"left": side_table(), "right": side_table()},
        "vertical_oscillation_ratio": float(vo),
        "asymmetry_pct": dict(asym or {}),
        "_meta": {
            "frames": 300,
            "duration_s": 10.0,
            "fps": 30.0,
            "direction": "right",
            "leg_length_units": float(leg_length),
        },
    }
    return payload


def session_clips(metrics_by_band, speeds=None):
    """[(band, metrics)] or {band: metrics} -> interpret_session clip list."""
    speeds = speeds or {"easy": 2.6, "moderate": 3.3, "fast": 4.1}
    items = metrics_by_band.items() if isinstance(metrics_by_band, dict) else metrics_by_band
    return [
        {"speed_label": band, "speed_value": speeds[band], "metrics": m}
        for band, m in items
    ]
