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


def _key_joint_visibility(csv_path):
    df = pd.read_csv(csv_path)
    cols = [f"{j}_vis" for j in KEY_JOINTS if f"{j}_vis" in df.columns]
    if not cols:
        return None
    val = df[cols].mean().mean()
    return None if pd.isna(val) else round(float(val), 3)


def analyze_clip(video_path, out_dir=None, model_variant="lite", smooth=9):
    """Raw clip -> all artifacts + quality report.

    Returns a dict with artifact paths, video properties, quality flags,
    and the metrics. Raises VideoError / PoseQualityError / MetricsError
    with an informative message instead of producing partial junk.
    """
    if not os.path.exists(video_path):
        raise VideoError(f"Video not found: {video_path}")

    # Lazy import: mediapipe/cv2 are heavy and only this stage needs them.
    from .pose import extract_pose

    ex = extract_pose(video_path, out_dir=out_dir, model_variant=model_variant)
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
    if key_vis is not None and key_vis < MIN_KEY_JOINT_VISIBILITY:
        quality_flags.append("low_key_joint_visibility")
    if duration_s < MIN_CLIP_SECONDS:
        quality_flags.append("short_clip")

    metrics = compute_metrics(
        ex.landmarks_csv_path, fps=ex.fps, width=ex.width, height=ex.height,
        smooth=smooth,
    )
    if (metrics.get("steps_detected") or 0) < MIN_STRIKES:
        quality_flags.append("few_strikes")

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
