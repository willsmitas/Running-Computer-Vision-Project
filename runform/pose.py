"""Video -> skeleton overlay video + landmarks CSV (23-point reduced set).

Backend: RTMPose, via rtmlib's Wholebody solution (COCO-WholeBody, 133
keypoints) on onnxruntime. Chosen over MediaPipe after real-footage
validation: MediaPipe swapped left/right leg identity at essentially every
limb crossover on low-resolution footage while passing every quality gate,
whereas RTMPose held identity perfectly on the same segments — see the
2026-07-30 addendum in scripts/phase0_validation.md. A Wholebody model
(not plain COCO-17 body) is required because metrics.py reads heel and
forefoot points that COCO-17 lacks.

This is the only module that imports rtmlib/cv2. Keep it that way so
the interpretation layers (metrics, session, plan, comparison) stay
runnable and testable without heavy vision dependencies.

Model files download automatically to ~/.cache/rtmlib on first run.
CPU inference runs on the order of 1-5 s/frame — much slower than the
old MediaPipe backend. Identity correctness was judged worth the wait;
use mode="lightweight" when speed matters more than accuracy.

Standalone usage (or via the pose_skeleton_starter.py shim):
    python -m runform.pose input_video.mp4 [--mode balanced]
"""

import argparse
import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np
from rtmlib import Wholebody

from .errors import VideoError
from .landmarks import LANDMARK_NAMES, POSE_CONNECTIONS

# rtmlib Wholebody presets: pose backbone + input resolution tradeoffs.
# "balanced" is the variant validated against real footage in the
# 2026-07 evaluations; "performance" is larger/slower, "lightweight"
# faster and less accurate.
RTM_MODES = ("performance", "balanced", "lightweight")

# COCO-WholeBody-133 index order (from rtmlib's own keypoint table,
# rtmlib/visualization/skeleton/coco133.py — verified empirically during
# the A/B evaluation, not assumed from memory):
#   0-4    face: nose, left/right eye, left/right ear
#   5-16   COCO-17 body: shoulders, elbows, wrists, hips, knees, ankles
#   17-22  feet: left big/small toe, left heel, right big/small toe,
#          right heel
#   23-90  face mesh (68 pts, unused)
#   91-111 left hand: root + 4 points per finger (thumb..pinky)
#   112-132 right hand, same layout

# The averaged "head" point (reduced-scheme index 0) comes from the five
# body-set face points. The nose alone is noisy in profile view; a head
# centroid is a much stabler anchor for the skeleton.
_HEAD_SOURCE = (0, 1, 2, 3, 4)

# LANDMARK_NAMES[1:] -> COCO-WholeBody index. Two mapping notes:
# - foot_index has no exact COCO-WholeBody equivalent; big_toe is the
#   same forefoot point for every downstream use (toe-vs-heel x offset
#   for running_direction(), foot-strike x for overstride).
# - pinky/index/thumb map to the matching hand knuckles (pinky/index
#   MCP, thumb second joint — the same joints MediaPipe's pose points
#   referred to). They are drawn in the overlay but never read by
#   metrics.py.
_BODY_SOURCE = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_pinky": 91 + 17, "right_pinky": 112 + 17,
    "left_index": 91 + 5, "right_index": 112 + 5,
    "left_thumb": 91 + 2, "right_thumb": 112 + 2,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
    "left_heel": 19, "right_heel": 22,
    "left_foot_index": 17, "right_foot_index": 20,
}


class _Point:
    """Minimal landmark with normalized x/y plus the z/visibility fields
    the CSV schema expects, so the averaged head point and every mapped
    keypoint can be treated identically downstream."""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def reduce_landmarks(keypoints, scores, width, height):
    """Map one person's (133, 2) pixel keypoints + (133,) scores into the
    23-point reduced scheme: 1 averaged head point + 22 body landmarks in
    LANDMARK_NAMES order, x/y normalized to 0-1 by frame width/height
    (the CSV convention metrics.py expects — it applies the aspect
    correction itself; do NOT aspect-correct here).

    z is always 0.0: RTMPose is 2D-only, and z was never read from the
    old backend either (single-camera depth is unreliable). Scores are
    clipped to [0, 1] — rtmlib/onnxruntime simcc decoding can emit
    values slightly above 1.
    """
    def vis(idx):
        return float(np.clip(scores[idx], 0.0, 1.0))

    head = _Point(
        x=float(np.mean([keypoints[i][0] for i in _HEAD_SOURCE])) / width,
        y=float(np.mean([keypoints[i][1] for i in _HEAD_SOURCE])) / height,
        z=0.0,
        visibility=float(np.mean([vis(i) for i in _HEAD_SOURCE])),
    )
    reduced = [head]
    for name in LANDMARK_NAMES[1:]:
        idx = _BODY_SOURCE[name]
        reduced.append(_Point(
            x=float(keypoints[idx][0]) / width,
            y=float(keypoints[idx][1]) / height,
            z=0.0,
            visibility=vis(idx),
        ))
    return reduced


# BGR (OpenCV order). Left/right legs get the most visually distinct pair
# (red vs. blue) since limb-crossover leg-swap is the top Phase 0 validation
# risk (BUILD_PLAN.md) -- the overlay needs to make a swap obvious at a
# glance. Arms echo the same warm-left/cool-right convention as a secondary
# cue; torso/head bones are neutral green.
COLOR_HEAD = (255, 255, 255)
COLOR_TORSO = (0, 200, 0)
COLOR_LEFT_ARM = (0, 140, 255)
COLOR_RIGHT_ARM = (200, 0, 160)
COLOR_LEFT_LEG = (0, 0, 255)
COLOR_RIGHT_LEG = (255, 60, 0)

_LEFT_LEG_JOINTS = {"left_knee", "left_ankle", "left_heel", "left_foot_index"}
_RIGHT_LEG_JOINTS = {"right_knee", "right_ankle", "right_heel", "right_foot_index"}
_LEFT_ARM_JOINTS = {"left_elbow", "left_wrist", "left_pinky", "left_index", "left_thumb"}
_RIGHT_ARM_JOINTS = {"right_elbow", "right_wrist", "right_pinky", "right_index", "right_thumb"}

# (start_idx, end_idx) pairs from POSE_CONNECTIONS, grouped by limb so each
# bone segment can be colored independently of its (possibly torso-shared)
# endpoint joints.
_LEFT_LEG_CONN = {(13, 15), (15, 17), (17, 19), (17, 21), (19, 21)}
_RIGHT_LEG_CONN = {(14, 16), (16, 18), (18, 20), (18, 22), (20, 22)}
_LEFT_ARM_CONN = {(1, 3), (3, 5), (5, 7), (5, 9), (5, 11), (7, 9)}
_RIGHT_ARM_CONN = {(2, 4), (4, 6), (6, 8), (6, 10), (6, 12), (8, 10)}


def _joint_color(name):
    if name in _LEFT_LEG_JOINTS:
        return COLOR_LEFT_LEG
    if name in _RIGHT_LEG_JOINTS:
        return COLOR_RIGHT_LEG
    if name in _LEFT_ARM_JOINTS:
        return COLOR_LEFT_ARM
    if name in _RIGHT_ARM_JOINTS:
        return COLOR_RIGHT_ARM
    if name == "head":
        return COLOR_HEAD
    return COLOR_TORSO  # shoulders, hips


def _connection_color(start_idx, end_idx):
    pair = (start_idx, end_idx)
    if pair in _LEFT_LEG_CONN:
        return COLOR_LEFT_LEG
    if pair in _RIGHT_LEG_CONN:
        return COLOR_RIGHT_LEG
    if pair in _LEFT_ARM_CONN:
        return COLOR_LEFT_ARM
    if pair in _RIGHT_ARM_CONN:
        return COLOR_RIGHT_ARM
    return COLOR_TORSO


def draw_landmarks(frame, landmarks, width, height):
    points = []
    for name, lm in zip(LANDMARK_NAMES, landmarks):
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, _joint_color(name), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            color = _connection_color(start_idx, end_idx)
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)


@dataclass
class PoseExtraction:
    """What Phase 1 needs to know about an extraction, without re-opening
    the video: artifact paths plus the video properties (auto-extracted,
    so downstream stages never require --fps/--width/--height flags)."""

    input_path: str
    skeleton_video_path: str
    landmarks_csv_path: str
    fps: float
    width: int
    height: int
    frames: int
    detected_frames: int

    @property
    def detection_rate(self) -> float:
        return self.detected_frames / self.frames if self.frames else 0.0


def extract_pose(
    input_path: str,
    out_dir: str | None = None,
    mode: str = "balanced",
    device: str = "cpu",
    progress_every: int = 50,
) -> PoseExtraction:
    """Run pose estimation over a video. Writes the skeleton overlay video
    and the landmarks CSV next to the input (or into out_dir) and returns
    a PoseExtraction. Raises VideoError on unreadable/empty input.

    Every frame is detected independently — no ROI or identity carried
    over between frames. Per-frame detection was validated (on both
    backends) to avoid the tracker failure mode where a bad lock on one
    leg persists across many frames.

    The person bbox is detected explicitly (wholebody.det_model) rather
    than letting rtmlib's __call__ fall back to a whole-frame bbox when
    it finds nobody — that fallback would make every frame look
    "detected" and silently corrupt the detection rate the quality gates
    depend on. If several bboxes fire (background false positive,
    reflection), the largest is kept, matching the old single-person
    configuration.
    """
    if mode not in RTM_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Options: {', '.join(RTM_MODES)}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise VideoError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = os.path.splitext(os.path.basename(input_path))[0]
    dest = out_dir if out_dir else (os.path.dirname(input_path) or ".")
    os.makedirs(dest, exist_ok=True)
    out_video_path = os.path.join(dest, stem + "_skeleton.mp4")
    out_csv_path = os.path.join(dest, stem + "_landmarks.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    csv_file = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["frame"]
    for name in LANDMARK_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]
    csv_writer.writerow(header)

    wholebody = Wholebody(mode=mode, backend="onnxruntime", device=device)

    frame_idx = 0
    detected_count = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            bboxes = wholebody.det_model(frame)

            row = [frame_idx]
            if len(bboxes) > 0:
                if len(bboxes) > 1:
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
                    bboxes = [bboxes[int(np.argmax(areas))]]
                keypoints, scores = wholebody.pose_model(frame, bboxes=bboxes)
                landmarks = reduce_landmarks(keypoints[0], scores[0], width, height)
                draw_landmarks(frame, landmarks, width, height)
                for lm in landmarks:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                detected_count += 1
            else:
                row += [None] * (4 * len(LANDMARK_NAMES))

            csv_writer.writerow(row)
            writer.write(frame)
            frame_idx += 1
            if progress_every and frame_idx % progress_every == 0:
                print(f"  ...{frame_idx} frames processed", flush=True)
    finally:
        cap.release()
        writer.release()
        csv_file.close()

    if frame_idx == 0:
        raise VideoError(f"Video contained no readable frames: {input_path}")

    return PoseExtraction(
        input_path=input_path,
        skeleton_video_path=out_video_path,
        landmarks_csv_path=out_csv_path,
        fps=float(fps),
        width=width,
        height=height,
        frames=frame_idx,
        detected_frames=detected_count,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run RTMPose (rtmlib Wholebody) on a video; write skeleton overlay + landmarks CSV."
    )
    p.add_argument("video", help="input video file")
    p.add_argument("--mode", default="balanced", choices=RTM_MODES,
                   help="rtmlib accuracy/speed preset (default: balanced)")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"),
                   help="onnxruntime execution device (default: cpu)")
    p.add_argument("--out-dir", default=None, help="write artifacts here instead of next to the video")
    args = p.parse_args(argv)

    ex = extract_pose(
        args.video, out_dir=args.out_dir, mode=args.mode, device=args.device,
    )

    print(f"Processed {ex.frames} frames.")
    print(f"Pose detected in {ex.detected_frames} frames ({ex.detection_rate:.1%}).")
    print(f"Overlay video: {ex.skeleton_video_path}")
    print(f"Landmark CSV:  {ex.landmarks_csv_path}")

    if ex.detection_rate < 0.8:
        print(
            "\nWarning: detection rate is low. Common causes -> runner too small "
            "in frame, motion blur, side-on angle with limb occlusion, or poor "
            "lighting. Try mode='performance', or a clip with the runner more "
            "centered/filling the frame."
        )


if __name__ == "__main__":
    main()
