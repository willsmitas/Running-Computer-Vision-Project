"""Landmarks CSV -> running form metrics.

Pipeline:
    1. Condition   -> aspect correction, gap fill, smoothing
    2. Normalize   -> scale metrics by the runner's own leg length
    3. Gait events -> detect foot strike / toe-off frames
    4. Metrics     -> compute per stride, aggregate with spread

Assumes a SIDE-ON (sagittal) view. Metrics from head-on or angled footage
will be unreliable — especially knee angle and overstride.

Invariants preserved here (CLAUDE.md — do not break):
  - Aspect correction on x is mandatory before any geometry.
  - Gait events use ankle position RELATIVE TO HIP, never absolute.
  - MediaPipe's z is ignored entirely; sagittal-plane 2D only.
  - All distance metrics normalize by leg length.
  - joint_angle returns the INTERIOR angle (straight leg = 180°);
    literature flexion = 180 − interior. Convert before comparing.

Standalone usage (or via the running_metrics.py shim):
    python -m runform.metrics clip_landmarks.csv --fps 30 --width 1920 --height 1080
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from .errors import MetricsError
from .landmarks import LANDMARK_NAMES

# Joints this analysis actually uses. The head/hand/foot-detail landmarks
# in the CSV are ignored here — they're either noisy or irrelevant to
# sagittal gait metrics.
REQUIRED = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Guard the shared-scheme invariant: every joint read here must exist in
# the scheme pose.py writes. Fails at import time if the two drift apart.
assert set(REQUIRED) <= set(LANDMARK_NAMES), "metrics REQUIRED joints not in landmark scheme"

# Landmarks below this visibility are treated as missing rather than trusted.
VIS_THRESHOLD = 0.5

# Fewer frames than this cannot contain even a couple of strides at any
# plausible cadence/fps combination — refuse rather than emit junk.
MIN_FRAMES = 20


# ---------------------------------------------------------------------------
# 1. Signal conditioning
# ---------------------------------------------------------------------------

def load_and_condition(csv_path, aspect, smooth_window):
    """Load CSV, drop low-confidence points, fill gaps, smooth."""
    df = pd.read_csv(csv_path)

    coords = {}
    for joint in REQUIRED:
        for axis in ("x", "y"):
            col, vis_col = f"{joint}_{axis}", f"{joint}_vis"
            if col not in df.columns:
                raise MetricsError(
                    f"Column '{col}' missing from CSV. Was this produced by "
                    f"runform.pose (pose_skeleton_starter)?"
                )
            series = df[col].copy()
            # Mask out points the model wasn't confident about
            if vis_col in df.columns:
                series[df[vis_col] < VIS_THRESHOLD] = np.nan
            coords[f"{joint}_{axis}"] = series

    out = pd.DataFrame(coords)

    # ASPECT CORRECTION: MediaPipe normalizes x by frame width and y by
    # frame height independently. On a 16:9 video, one x-unit spans 1.78x
    # more real-world distance than one y-unit. Scaling x by the aspect
    # ratio puts both axes in the same units so angles and distances are
    # geometrically correct.
    for col in out.columns:
        if col.endswith("_x"):
            out[col] = out[col] * aspect

    # Gap fill: short dropouts (occlusion at limb crossover) interpolate
    # cleanly. Long dropouts stay NaN and get excluded from metrics.
    out = out.interpolate(method="linear", limit=5, limit_direction="both")

    # Smooth: raw pose output is jittery frame to frame. Savitzky-Golay
    # preserves peak shape better than a moving average, which matters
    # because gait event detection depends on peak locations.
    if smooth_window and smooth_window >= 5:
        window = min(smooth_window, len(out) if len(out) % 2 else len(out) - 1)
        if window % 2 == 0:
            window -= 1
        if window >= 5:
            for col in out.columns:
                valid = out[col].notna()
                if valid.sum() > window:
                    smoothed = out[col].copy()
                    smoothed[valid] = savgol_filter(
                        out[col][valid].to_numpy(), window, polyorder=2
                    )
                    out[col] = smoothed

    return out


# ---------------------------------------------------------------------------
# 2. Scale normalization
# ---------------------------------------------------------------------------

def leg_length(df):
    """Median hip-to-ankle distance. Used as the body-scale reference so
    metrics are comparable across videos shot at different distances."""
    lengths = []
    for side in ("left", "right"):
        dx = df[f"{side}_hip_x"] - df[f"{side}_ankle_x"]
        dy = df[f"{side}_hip_y"] - df[f"{side}_ankle_y"]
        lengths.append(np.sqrt(dx ** 2 + dy ** 2))
    return float(np.nanmedian(pd.concat(lengths)))


def midpoint(df, joint):
    """Midpoint between the left and right instance of a joint."""
    x = (df[f"left_{joint}_x"] + df[f"right_{joint}_x"]) / 2
    y = (df[f"left_{joint}_y"] + df[f"right_{joint}_y"]) / 2
    return x, y


# ---------------------------------------------------------------------------
# 3. Gait event detection
# ---------------------------------------------------------------------------

def running_direction(df):
    """+1 if the runner faces screen-right, -1 if screen-left.
    Toes sit forward of the heel, so that offset gives the facing."""
    offsets = []
    for side in ("left", "right"):
        offsets.append(df[f"{side}_foot_index_x"] - df[f"{side}_heel_x"])
    mean_offset = float(np.nanmean(pd.concat(offsets)))
    return 1.0 if mean_offset >= 0 else -1.0


# A real running stride swings the ankle through at least this fraction of
# leg length relative to the hip. Below it, we are looking at tracking
# noise on a near-stationary subject, not gait.
MIN_SWING_RATIO = 0.15


def detect_events(df, side, direction, fps, leg_len):
    """Find foot strike and toe-off frames for one foot.

    Method: track the ankle's horizontal position RELATIVE TO THE HIP.
    Using the relative signal (not absolute ankle position) cancels out
    camera panning and the runner's forward travel, leaving a clean
    oscillation. The foot reaches its furthest-forward point at/near foot
    strike, and its furthest-back point at/near toe-off.
    """
    hip_x, _ = midpoint(df, "hip")
    rel = (df[f"{side}_ankle_x"] - hip_x) * direction
    rel = rel.to_numpy()

    if np.isnan(rel).all():
        return np.array([], dtype=int), np.array([], dtype=int)

    filled = pd.Series(rel).interpolate(limit_direction="both").to_numpy()

    # ABSOLUTE AMPLITUDE GATE. Without this, a clip of someone standing
    # still produces a flat signal whose own noise clears a purely
    # relative threshold, and jitter gets reported as a plausible-looking
    # cadence. Requiring real swing relative to leg length rejects that.
    swing = float(np.percentile(filled, 95) - np.percentile(filled, 5))
    if leg_len <= 0 or swing < MIN_SWING_RATIO * leg_len:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Minimum spacing between successive strikes of the SAME foot.
    # A stride rate above ~240 steps/min is not physically plausible, so
    # this rejects double-detections from residual noise.
    min_gap = max(int(fps * 60 / 240), 2)
    amplitude = np.nanstd(filled)

    strikes, _ = find_peaks(filled, distance=min_gap, prominence=amplitude * 0.3)
    toeoffs, _ = find_peaks(-filled, distance=min_gap, prominence=amplitude * 0.3)
    return strikes, toeoffs


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def joint_angle(df, frame, a, b, c):
    """Interior angle at joint b, formed by segments b->a and b->c.
    Returns degrees, or None if any point is missing."""
    try:
        pa = np.array([df[f"{a}_x"].iloc[frame], df[f"{a}_y"].iloc[frame]])
        pb = np.array([df[f"{b}_x"].iloc[frame], df[f"{b}_y"].iloc[frame]])
        pc = np.array([df[f"{c}_x"].iloc[frame], df[f"{c}_y"].iloc[frame]])
    except IndexError:
        return None
    if np.isnan(np.concatenate([pa, pb, pc])).any():
        return None

    v1, v2 = pa - pb, pc - pb
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def trunk_lean(df, frame, direction):
    """Forward lean of the torso from vertical, in degrees.
    Positive = leaning in the direction of travel."""
    hip_x, hip_y = midpoint(df, "hip")
    sh_x, sh_y = midpoint(df, "shoulder")
    dx = sh_x.iloc[frame] - hip_x.iloc[frame]
    # Image y grows downward, so shoulder-above-hip is a negative dy.
    dy = sh_y.iloc[frame] - hip_y.iloc[frame]
    if np.isnan(dx) or np.isnan(dy):
        return None
    return float(np.degrees(np.arctan2(dx * direction, abs(dy))))


def summarize(values):
    """Mean + spread + n. The spread is the useful part: a tight SD means
    a consistent stride, a wide one means either genuine variability or
    unreliable tracking. Both are worth knowing before trusting the mean."""
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return None
    return {
        "mean": round(float(np.mean(vals)), 2),
        "sd": round(float(np.std(vals)), 2),
        "n": len(vals),
    }


def analyze(df, fps, direction, leg_len):
    events = {}
    for side in ("left", "right"):
        strikes, toeoffs = detect_events(df, side, direction, fps, leg_len)
        events[side] = {"strikes": strikes, "toeoffs": toeoffs}

    results = {}
    duration_s = len(df) / fps

    # --- Cadence -----------------------------------------------------------
    # Derived from the mean interval BETWEEN strikes, not from
    # (step count / clip duration). Counting steps truncates the partial
    # stride at each end of the clip, which biases short clips low by a
    # few percent. Intervals measure the actual step period directly.
    all_strikes = np.sort(
        np.concatenate([events[s]["strikes"] for s in ("left", "right")])
    ) if any(len(events[s]["strikes"]) for s in ("left", "right")) else np.array([])

    total_steps = len(all_strikes)
    if total_steps >= 3:
        intervals = np.diff(all_strikes)
        # Median resists a single missed strike (which would otherwise
        # show up as one double-length interval and drag the mean).
        step_period_frames = float(np.median(intervals))
        results["cadence_spm"] = (
            round(60.0 * fps / step_period_frames, 1) if step_period_frames > 0 else None
        )
    elif duration_s > 0:
        results["cadence_spm"] = round(total_steps / duration_s * 60, 1)
    else:
        results["cadence_spm"] = None
    results["steps_detected"] = total_steps

    # --- Per-side metrics --------------------------------------------------
    per_side = {}
    for side in ("left", "right"):
        strikes = events[side]["strikes"]
        toeoffs = events[side]["toeoffs"]

        knee_angles, overstrides, leans, contact_times = [], [], [], []
        hip_x, _ = midpoint(df, "hip")

        for f in strikes:
            # Knee flexion at foot strike. A very straight leg (angle near
            # 180) at contact is the classic overstriding signature.
            knee_angles.append(
                joint_angle(df, f, f"{side}_hip", f"{side}_knee", f"{side}_ankle")
            )
            leans.append(trunk_lean(df, f, direction))

            # Overstride: how far ahead of the hip the foot lands,
            # as a fraction of leg length.
            ankle_x = df[f"{side}_ankle_x"].iloc[f]
            if not np.isnan(ankle_x) and leg_len > 0:
                overstrides.append(
                    float((ankle_x - hip_x.iloc[f]) * direction / leg_len)
                )

            # Ground contact: this strike to the next toe-off after it.
            later = toeoffs[toeoffs > f]
            if len(later):
                contact_times.append(float((later[0] - f) / fps * 1000))

        # Hip extension at toe-off: how far the leg drives back behind
        # the body. Limited extension often points to hip flexor tightness.
        hip_ext = [
            joint_angle(df, f, f"{side}_shoulder", f"{side}_hip", f"{side}_knee")
            for f in toeoffs
        ]

        per_side[side] = {
            "knee_angle_at_strike_deg": summarize(knee_angles),
            "hip_angle_at_toeoff_deg": summarize(hip_ext),
            "ground_contact_ms": summarize(contact_times),
            "overstride_ratio": summarize(overstrides),
            "trunk_lean_deg": summarize(leans),
            "strikes_detected": len(strikes),
        }

    results["per_side"] = per_side

    # --- Vertical oscillation ---------------------------------------------
    # Peak-to-peak bounce of the hip midpoint, as a fraction of leg length.
    _, hip_y = midpoint(df, "hip")
    hy = hip_y.dropna().to_numpy()
    if len(hy) > 10 and leg_len > 0:
        # 5th-95th percentile rather than min/max: a single tracking
        # glitch would otherwise dominate the range.
        p2p = float(np.percentile(hy, 95) - np.percentile(hy, 5))
        results["vertical_oscillation_ratio"] = round(p2p / leg_len, 4)
    else:
        results["vertical_oscillation_ratio"] = None

    # --- Symmetry ----------------------------------------------------------
    # Left/right asymmetry is one of the more actionable outputs here,
    # since it can flag a compensation pattern from a past injury.
    symmetry = {}
    for metric in ("ground_contact_ms", "knee_angle_at_strike_deg", "overstride_ratio"):
        l, r = per_side["left"][metric], per_side["right"][metric]
        if l and r:
            denom = (abs(l["mean"]) + abs(r["mean"])) / 2
            if denom > 0:
                symmetry[metric] = round(abs(l["mean"] - r["mean"]) / denom * 100, 1)
    results["asymmetry_pct"] = symmetry

    return results


def compute_metrics(csv_path, fps, width, height, smooth=9):
    """The importable entry point: landmarks CSV -> metrics dict.

    Raises MetricsError on anything that would make the numbers junk
    (missing columns, too few frames, no usable body scale) rather than
    emitting metrics silently derived from degraded tracking.
    """
    aspect = width / height
    df = load_and_condition(csv_path, aspect, smooth)

    if len(df) < MIN_FRAMES:
        raise MetricsError(
            f"Too few frames to analyze ({len(df)} < {MIN_FRAMES}). Need a longer clip."
        )

    direction = running_direction(df)
    leg_len = leg_length(df)
    if not np.isfinite(leg_len) or leg_len <= 0:
        raise MetricsError(
            "Could not establish body scale — check pose detection quality."
        )

    results = analyze(df, fps, direction, leg_len)
    results["_meta"] = {
        "frames": len(df),
        "duration_s": round(len(df) / fps, 2),
        "fps": fps,
        "direction": "right" if direction > 0 else "left",
        # Body scale in aspect-corrected normalized units. Phase 5 uses
        # this as the setup-drift check: a material change between
        # sessions means the camera moved, not the runner's form.
        "leg_length_units": round(leg_len, 4),
    }
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Compute running form metrics from a landmarks CSV.")
    p.add_argument("csv", help="landmarks CSV from runform.pose / pose_skeleton_starter.py")
    p.add_argument("--fps", type=float, required=True, help="source video frame rate")
    p.add_argument("--width", type=float, default=1920, help="source video width in px")
    p.add_argument("--height", type=float, default=1080, help="source video height in px")
    p.add_argument("--smooth", type=int, default=9, help="smoothing window in frames (odd, >=5; 0 disables)")
    p.add_argument("--out", default=None, help="write metrics to this JSON file")
    args = p.parse_args(argv)

    try:
        results = compute_metrics(args.csv, args.fps, args.width, args.height, args.smooth)
    except MetricsError as e:
        sys.exit(str(e))

    text = json.dumps(results, indent=2)
    print(text)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"\nWrote {args.out}")

    # Sanity warnings. Metrics that look precise but come from bad event
    # detection are worse than no metrics at all.
    steps = results.get("steps_detected", 0)
    if steps < 6:
        print(
            f"\nWarning: only {steps} foot strikes detected. Either the clip is "
            "too short or gait event detection is failing — check the skeleton "
            "overlay video before trusting these numbers.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
