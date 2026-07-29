"""Phase 1: one command — video in; skeleton video, landmarks CSV,
metrics JSON, and a quality report out.

fps/width/height are auto-extracted from the video (no flags), errors are
structured (bad video / no pose / too short), and quality flags travel
with the artifacts so downstream layers can gate on them.
"""

import json
import os

import pandas as pd

from .errors import PoseQualityError
from .errors import VideoError
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


def analyze_clip(video_path, out_dir=None, model_variant="lite", smooth=9, frame_mode="image"):
    """Raw clip -> all artifacts + quality report.

    Returns a dict with artifact paths, video properties, quality flags,
    and the metrics. Raises VideoError / PoseQualityError / MetricsError
    with an informative message instead of producing partial junk.
    """
    if not os.path.exists(video_path):
        raise VideoError(f"Video not found: {video_path}")

    # Lazy import: mediapipe/cv2 are heavy and only this stage needs them.
    from .pose import extract_pose

    ex = extract_pose(video_path, out_dir=out_dir, model_variant=model_variant, frame_mode=frame_mode)
    duration_s = ex.frames / ex.fps if ex.fps else 0.0

    if ex.detection_rate < HARD_MIN_DETECTION_RATE:
        raise PoseQualityError(
            f"Pose detected in only {ex.detection_rate:.0%} of frames — "
            f"metrics from mostly-interpolated tracking would be junk, so "
            f"none were computed. Common causes: runner too small in frame, "
            f"motion blur, occlusion, poor lighting. Try model_variant="
            f"'full' or 'heavy', or re-film with the runner filling the frame."
        )

    quality_flags = []
    if ex.detection_rate < MIN_DETECTION_RATE:
        quality_flags.append("low_detection_rate")
    key_vis = _key_joint_visibility(ex.landmarks_csv_path)
    # Gate on the WORSE side, not the average: one unusable leg is enough
    # to invalidate every per-side and asymmetry metric.
    if key_vis["min"] is not None and key_vis["min"] < MIN_KEY_JOINT_VISIBILITY:
        quality_flags.append("low_key_joint_visibility")
    if duration_s < MIN_CLIP_SECONDS:
        quality_flags.append("short_clip")

    metrics = compute_metrics(
        ex.landmarks_csv_path, fps=ex.fps, width=ex.width, height=ex.height,
        smooth=smooth,
    )
    if (metrics.get("steps_detected") or 0) < MIN_STRIKES:
        quality_flags.append("few_strikes")

    per_side = metrics.get("per_side") or {}
    n_left = (per_side.get("left") or {}).get("strikes_detected") or 0
    n_right = (per_side.get("right") or {}).get("strikes_detected") or 0
    if abs(n_left - n_right) > STRIKE_IMBALANCE_ABS:
        quality_flags.append("strike_count_imbalance")

    stem = os.path.splitext(ex.landmarks_csv_path)[0]
    stem = stem[: -len("_landmarks")] if stem.endswith("_landmarks") else stem
    metrics_json_path = stem + "_metrics.json"
    quality_json_path = stem + "_quality.json"

    result = {
        "video_path": video_path,
        "skeleton_video_path": ex.skeleton_video_path,
        "landmarks_csv_path": ex.landmarks_csv_path,
        "metrics_json_path": metrics_json_path,
        "quality_json_path": quality_json_path,
        "fps": ex.fps,
        "width": ex.width,
        "height": ex.height,
        "frames": ex.frames,
        "duration_s": round(duration_s, 2),
        "detection_rate": round(ex.detection_rate, 3),
        "key_joint_visibility": key_vis,
        "quality_flags": quality_flags,
        "metrics": metrics,
    }

    with open(metrics_json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(quality_json_path, "w") as fh:
        json.dump({k: v for k, v in result.items() if k != "metrics"}, fh, indent=2)

    return result
