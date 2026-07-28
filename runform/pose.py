"""Video -> skeleton overlay video + landmarks CSV (23-point reduced set).

Uses MediaPipe's current "Tasks" API (PoseLandmarker), NOT the old
mp.solutions.pose API, which was removed in mediapipe 0.10.30+.

This is the only module that imports mediapipe/cv2. Keep it that way so
the interpretation layers (metrics, session, plan, comparison) stay
runnable and testable without heavy vision dependencies.

First-time setup: needs a model file, downloaded automatically on first
run.

Standalone usage (or via the pose_skeleton_starter.py shim):
    python -m runform.pose input_video.mp4 [--model-variant lite]
"""

import argparse
import csv
import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from .errors import VideoError
from .landmarks import LANDMARK_NAMES, NUM_FACE_LANDMARKS, POSE_CONNECTIONS

# Google's official hosted models. "lite" is fastest and fine for getting
# the pipeline working; "full"/"heavy" trade speed for accuracy — worth
# trying when detection rate is low (see Phase 0 in BUILD_PLAN.md).
MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}


def model_path(variant: str) -> str:
    # "pose_landmarker.task" (no suffix) is the historical filename the
    # lite model was downloaded to; keep it so existing downloads reuse.
    if variant == "lite":
        return "pose_landmarker.task"
    return f"pose_landmarker_{variant}.task"


def ensure_model(variant: str = "lite") -> str:
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant '{variant}'. Options: {sorted(MODEL_URLS)}")
    path = model_path(variant)
    if not os.path.exists(path):
        print(f"Downloading pose model ({variant}) to {path} ...")
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
        print("Done.")
    return path


class _Point:
    """Minimal stand-in with the same x/y/z/visibility fields as a
    MediaPipe landmark, so the averaged head point can be treated
    identically to the raw landmarks everywhere downstream."""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def reduce_landmarks(landmarks):
    """Collapse the 33 raw MediaPipe landmarks into 23: 1 averaged head
    point + the 22 body landmarks (shoulders down), preserving order."""
    face = landmarks[:NUM_FACE_LANDMARKS]
    n = len(face)
    head = _Point(
        x=sum(lm.x for lm in face) / n,
        y=sum(lm.y for lm in face) / n,
        z=sum(lm.z for lm in face) / n,
        visibility=sum(lm.visibility for lm in face) / n,
    )
    return [head] + list(landmarks[NUM_FACE_LANDMARKS:])


def draw_landmarks(frame, landmarks, width, height):
    points = []
    for lm in landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)


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
    model_variant: str = "lite",
    progress_every: int = 100,
) -> PoseExtraction:
    """Run pose estimation over a video. Writes the skeleton overlay video
    and the landmarks CSV next to the input (or into out_dir) and returns
    a PoseExtraction. Raises VideoError on unreadable/empty input."""
    model = ensure_model(model_variant)

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

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    detected_count = 0

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_frame),
                )
                timestamp_ms = int((frame_idx / fps) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                row = [frame_idx]
                if result.pose_landmarks:
                    detected_count += 1
                    raw_landmarks = result.pose_landmarks[0]  # first (only) person
                    landmarks = reduce_landmarks(raw_landmarks)
                    draw_landmarks(frame, landmarks, width, height)
                    for lm in landmarks:
                        row += [lm.x, lm.y, lm.z, lm.visibility]
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
        description="Run MediaPipe pose on a video; write skeleton overlay + landmarks CSV."
    )
    p.add_argument("video", help="input video file")
    p.add_argument("--model-variant", default="lite", choices=sorted(MODEL_URLS),
                   help="pose model accuracy/speed tradeoff (default: lite)")
    p.add_argument("--out-dir", default=None, help="write artifacts here instead of next to the video")
    args = p.parse_args(argv)

    ex = extract_pose(args.video, out_dir=args.out_dir, model_variant=args.model_variant)

    print(f"Processed {ex.frames} frames.")
    print(f"Pose detected in {ex.detected_frames} frames ({ex.detection_rate:.1%}).")
    print(f"Overlay video: {ex.skeleton_video_path}")
    print(f"Landmark CSV:  {ex.landmarks_csv_path}")

    if ex.detection_rate < 0.8:
        print(
            "\nWarning: detection rate is low. Common causes -> runner too small "
            "in frame, motion blur, side-on angle with limb occlusion, or poor "
            "lighting. Try the 'full' or 'heavy' model variant, or a clip with "
            "the runner more centered/filling the frame."
        )


if __name__ == "__main__":
    main()
