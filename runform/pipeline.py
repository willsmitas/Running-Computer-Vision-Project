"""Phase 1: one command — video in; skeleton video, landmarks CSV,
metrics JSON, and a quality report out.

fps/width/height are auto-extracted from the video (no flags), errors are
structured (bad video / no pose / too short), and quality flags travel
with the artifacts so downstream layers can gate on them.

Between pose and metrics, leg-identity tracking (runform.leg_identity)
rewrites the landmarks into <clip>_landmarks_tracked.csv; metrics read
that, and the raw CSV is kept for audit.
"""

import json
import os

import pandas as pd

from .errors import PoseQualityError
from .errors import VideoError
from .leg_identity import STUCK_SIGN_FRACTION, relabel_csv
from .metrics import compute_metrics

# Below this fraction of frames tracked, the overlay needs eyeballing
# before the numbers are trusted -> flag.
MIN_DETECTION_RATE = 0.8
# Below this, metrics would be built mostly on interpolated positions.
# Refuse outright (fail loudly) rather than emit degraded metrics.
HARD_MIN_DETECTION_RATE = 0.5
# Mean model confidence on the joints gait metrics actually depend on.
MIN_KEY_JOINT_VISIBILITY = 0.6
# Shorter than this cannot yield enough strides for a stable {mean, sd, n}.
MIN_CLIP_SECONDS = 5.0
# Same threshold metrics.py warns at.
MIN_STRIKES = 6

KEY_JOINTS = (
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

# Each leg must carry a confirmed identity on at least this fraction of
# frames for per-side metrics to be graded above "low". Below it, enough
# of the clip is blind spans that per-side means rest on a minority of
# strides.
MIN_IDENTITY_COVERAGE = 0.8

# No treadmill distance runner sustains this; the MediaPipe leg-swap
# segment emitted 327 spm. Elite distance cadence tops out ~200-210;
# 230 leaves margin for sprint-pace clips without admitting aliasing.
MAX_PLAUSIBLE_CADENCE_SPM = 230

# Running alternates feet, so left and right strike counts can differ by
# at most one -- the clip simply starts and ends mid-stride. This bound is
# physical, NOT proportional: a 60 s clip is no more entitled to a wide
# gap than a 15 s one, so there is deliberately no percentage term here.
# Allow 2 rather than 1 to absorb a single missed event at a clip edge.
# A wider gap means one leg's events are being dropped -- the failure that
# showed up as 25/31 on real footage before the per-foot spacing fix.
STRIKE_IMBALANCE_ABS = 2


def _key_joint_visibility(csv_path):
    """Per-joint, per-side visibility of the gait-critical joints.

    Two levels of averaging had to go, because each one hid a real
    tracking failure on actual footage:
      - across sides: the near leg tracks at ~0.9 and masks a far leg at
        ~0.45, yet every per-side and asymmetry metric needs both legs.
      - across joints within a side: the hip sits at ~1.0 in every clip
        (it is the body centre and essentially never occluded), which
        dragged a 0.39 knee up to a 0.62 side average and over threshold.
    So gate on the WORST joint. Returns per-side dicts of joint -> mean
    visibility plus that side's "min", and a top-level "min" overall.
    """
    df = pd.read_csv(csv_path)
    out = {}
    for side in ("left", "right"):
        joints = {}
        for j in KEY_JOINTS:
            if not j.startswith(side):
                continue
            col = f"{j}_vis"
            if col not in df.columns:
                continue
            val = df[col].mean()
            if not pd.isna(val):
                joints[j[len(side) + 1:]] = round(float(val), 3)
        joints["min"] = min(joints.values()) if joints else None
        out[side] = joints
    mins = [out[s].get("min") for s in ("left", "right") if out[s].get("min") is not None]
    out["min"] = min(mins) if mins else None
    return out


def analyze_clip(video_path, out_dir=None, mode="balanced", device="cpu", smooth=9,
                 progress_cb=None, swap_seed=False):
    """Raw clip -> all artifacts + quality report.

    Returns a dict with artifact paths, video properties, quality flags,
    and the metrics. Raises VideoError / PoseQualityError / MetricsError
    with an informative message instead of producing partial junk.

    progress_cb(frames_done, total_or_None) reports pose-estimation
    progress (the slow stage) — used by the web UI.

    swap_seed: the model's left/right labels on the leg-identity seed
    frame are backwards (see runform.leg_identity). Usually set later via
    relabel_clip, once the runner has looked at the seed frame.
    """
    if not os.path.exists(video_path):
        raise VideoError(f"Video not found: {video_path}")

    # Lazy import: rtmlib/onnxruntime/cv2 are heavy and only this stage
    # needs them.
    from .pose import extract_pose

    ex = extract_pose(video_path, out_dir=out_dir, mode=mode, device=device,
                      progress_cb=progress_cb)

    if ex.detection_rate < HARD_MIN_DETECTION_RATE:
        raise PoseQualityError(
            f"Pose detected in only {ex.detection_rate:.0%} of frames — "
            f"metrics from mostly-interpolated tracking would be junk, so "
            f"none were computed. Common causes: runner too small in frame, "
            f"motion blur, occlusion, poor lighting. Try mode='performance', "
            f"or re-film with the runner filling the frame."
        )

    facts = {
        "video_path": video_path,
        "skeleton_video_path": ex.skeleton_video_path,
        "landmarks_csv_path": ex.landmarks_csv_path,
        "fps": ex.fps,
        "width": ex.width,
        "height": ex.height,
        "frames": ex.frames,
        "detection_rate": round(ex.detection_rate, 3),
    }
    return _interpret_landmarks(facts, smooth=smooth, swap_seed=swap_seed)


def relabel_clip(quality_json_path, swap_seed, smooth=9):
    """Re-run leg-identity tracking + metrics + quality flags on an
    already-analyzed clip with a new seed choice. Seconds, not minutes:
    pose estimation is not repeated. This is what the web UI's
    "swap red/blue" button calls."""
    with open(quality_json_path) as fh:
        q = json.load(fh)
    facts = {k: q[k] for k in (
        "video_path", "skeleton_video_path", "landmarks_csv_path", "fps",
        "width", "height", "frames", "detection_rate",
    )}
    return _interpret_landmarks(facts, smooth=smooth, swap_seed=swap_seed)


def _interpret_landmarks(facts, smooth, swap_seed):
    """Raw landmarks CSV -> tracked CSV, metrics, quality flags, re-rendered
    overlay and seed-frame image, with all JSON artifacts written."""
    raw_csv = facts["landmarks_csv_path"]
    duration_s = facts["frames"] / facts["fps"] if facts["fps"] else 0.0

    stem = os.path.splitext(raw_csv)[0]
    stem = stem[: -len("_landmarks")] if stem.endswith("_landmarks") else stem
    tracked_csv = stem + "_landmarks_tracked.csv"
    metrics_json_path = stem + "_metrics.json"
    quality_json_path = stem + "_quality.json"
    seed_jpg_path = stem + "_seed.jpg"

    identity = relabel_csv(raw_csv, tracked_csv, facts["width"], facts["height"],
                           swap_seed=swap_seed)

    quality_flags = []
    if facts["detection_rate"] < MIN_DETECTION_RATE:
        quality_flags.append("low_detection_rate")
    # Visibility is a property of the pose model's confidence, so it is
    # read from the raw CSV, not the tracked one (whose blind spans carry
    # visibility 0 by construction).
    key_vis = _key_joint_visibility(raw_csv)
    # Gate on the WORSE side, not the average: one unusable leg is enough
    # to invalidate every per-side and asymmetry metric.
    if key_vis["min"] is not None and key_vis["min"] < MIN_KEY_JOINT_VISIBILITY:
        quality_flags.append("low_key_joint_visibility")
    if duration_s < MIN_CLIP_SECONDS:
        quality_flags.append("short_clip")

    metrics = compute_metrics(
        tracked_csv, fps=facts["fps"], width=facts["width"], height=facts["height"],
        smooth=smooth, exclude_spans=identity["unlabeled_spans"],
    )
    if (metrics.get("steps_detected") or 0) < MIN_STRIKES:
        quality_flags.append("few_strikes")

    per_side = metrics.get("per_side") or {}
    n_left = (per_side.get("left") or {}).get("strikes_detected") or 0
    n_right = (per_side.get("right") or {}).get("strikes_detected") or 0
    if abs(n_left - n_right) > STRIKE_IMBALANCE_ABS:
        quality_flags.append("strike_count_imbalance")

    quality_flags += identity_flags(identity, metrics)

    # Re-render the overlay from the tracked labels so the colors on
    # screen are the identities the metrics used, and save the seed frame
    # for the runner to confirm. Needs cv2 + the original video; skipped
    # (not failed) if either is unavailable, since metrics do not depend
    # on it.
    seed_image = None
    if os.path.exists(facts.get("video_path") or ""):
        try:
            from .pose import render_overlay, render_seed_frame
        except ImportError:
            render_overlay = render_seed_frame = None
        if render_overlay is not None:
            render_overlay(facts["video_path"], tracked_csv, facts["skeleton_video_path"])
            if identity["seed_frame"] is not None:
                seed_image = render_seed_frame(
                    facts["video_path"], tracked_csv, identity["seed_frame"], seed_jpg_path,
                )

    result = {
        **facts,
        "tracked_landmarks_csv_path": tracked_csv,
        "metrics_json_path": metrics_json_path,
        "quality_json_path": quality_json_path,
        "seed_image_path": seed_image,
        "duration_s": round(duration_s, 2),
        "key_joint_visibility": key_vis,
        "leg_identity": identity,
        "quality_flags": quality_flags,
        "metrics": metrics,
    }

    with open(metrics_json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(quality_json_path, "w") as fh:
        json.dump({k: v for k, v in result.items() if k != "metrics"}, fh, indent=2)

    return result


def identity_flags(identity, metrics):
    """Quality flags from leg-identity tracking + a cadence sanity check.

    These exist because the leg-swap failure passed every other gate on
    real footage (100% detection, visibility 0.83+): see the addendum in
    scripts/phase0_validation.md.
    """
    flags = []
    covered = identity.get("labeled_fraction") or {}
    if identity.get("seed_frame") is None or min(
        covered.get("left", 0.0), covered.get("right", 0.0)
    ) < MIN_IDENTITY_COVERAGE:
        flags.append("leg_identity_gaps")
    stuck = identity.get("sign_stuck_fraction")
    if stuck is not None and stuck > STUCK_SIGN_FRACTION:
        flags.append("leg_labels_stuck")
    cadence = metrics.get("cadence_spm")
    if cadence is not None and cadence > MAX_PLAUSIBLE_CADENCE_SPM:
        flags.append("implausible_cadence")
    return flags
