"""Prototype: RTMPose (via rtmlib's Wholebody solution) as an A/B alternative
to runform/pose.py's MediaPipe extraction.

Evaluation/prototype only -- see CLAUDE.md and the task brief this script was
written for. Does NOT modify, import-and-monkeypatch, or otherwise touch
runform/pose.py, runform/landmarks.py, runform/metrics.py, runform/pipeline.py,
or runform/cli.py. It only *reads* runform.metrics.compute_metrics and
runform.pipeline._key_joint_visibility (both pure pandas/numpy/scipy, no
mediapipe/cv2 import at module load time) to reuse the project's existing,
unmodified metrics logic for the A/B comparison.

Why a Wholebody (COCO-WholeBody, 133-point) model rather than a plain
17-point COCO body model: runform.metrics.REQUIRED needs left/right heel and
left/right foot_index (used by running_direction() and the overstride calc).
Plain COCO-17 only has ankle -- no heel/toe -- so it cannot satisfy the
contract. COCO-WholeBody adds 6 per-foot keypoints (big toe, small toe, heel
per side) that plain body-only configs drop.

Environment note: this script is meant to be run from a DEDICATED venv
(rtmlib + onnxruntime + numpy/pandas/scipy/opencv-python), NOT the project's
main .venv -- see scripts/rtmpose_ab_comparison.md for why. It adds the repo
root to sys.path itself so `import runform` resolves without installing the
project's own venv-managed dependencies.

Usage:
    <rtm-venv>/python.exe scripts/prototype_rtmpose_pose.py \
        Smitas_5flat.mov Smitas_6flat.mov Smitas_8flat.mov \
        --out-dir out_rtmpose
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

from rtmlib import Wholebody

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Keypoint mapping
# ---------------------------------------------------------------------------
# COCO-WholeBody-133 index order, confirmed empirically by reading
# rtmlib/visualization/skeleton/coco133.py (the library's own keypoint-name
# table), NOT assumed from memory:
#   0-4   face (nose, l/r eye, l/r ear)
#   5-16  COCO-17 body (shoulders, elbows, wrists, hips, knees, ankles)
#   17-22 feet: left_big_toe, left_small_toe, left_heel,
#               right_big_toe, right_small_toe, right_heel
#   23-90 face mesh (68 pts) -- unused here
#   91-132 hands (21 + 21) -- unused here
COCO_WHOLEBODY_INDEX = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
    "left_big_toe": 17, "left_small_toe": 18, "left_heel": 19,
    "right_big_toe": 20, "right_small_toe": 21, "right_heel": 22,
}

# runform.metrics.REQUIRED (order matters for readability/diffing, not for
# correctness -- metrics.py looks columns up by name). MediaPipe's
# `foot_index` landmark sits near the tip of the 2nd/3rd toe; COCO-WholeBody
# has no single equivalent point, so big_toe is used as the closest stand-in
# (both are forefoot points used the same way downstream: toe-vs-heel x
# offset for running_direction(), and overstride's foot-strike x position).
REQUIRED_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
JOINT_SOURCE = {
    "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
    "left_hip": "left_hip", "right_hip": "right_hip",
    "left_knee": "left_knee", "right_knee": "right_knee",
    "left_ankle": "left_ankle", "right_ankle": "right_ankle",
    "left_heel": "left_heel", "right_heel": "right_heel",
    "left_foot_index": "left_big_toe", "right_foot_index": "right_big_toe",
}

# rtmlib/onnxruntime scores have been observed to exceed 1.0 slightly
# (e.g. 1.011) on real frames -- simcc decoding artifact, not a bug we can
# fix here. Clip to a sane [0, 1] confidence range before writing, per the
# task's normalization requirement.
VIS_CLIP_MIN, VIS_CLIP_MAX = 0.0, 1.0


def run_wholebody_on_clip(video_path, wholebody, progress_every=50):
    """Per-frame (no tracking) Wholebody inference over a video.

    Mirrors runform.pose.extract_pose's frame_mode="image": every frame is
    detected independently with no ROI carried over from the previous frame.
    That choice is deliberate, not incidental -- BUILD_PLAN.md/pipeline.py
    document that per-frame detection measurably reduced left/right strike
    imbalance for MediaPipe on this project's real footage versus MediaPipe's
    tracking mode, and the same carry-over failure mode (a bad lock on one
    leg persisting across frames) is a generic risk for any tracker-based
    mode, not MediaPipe-specific. Using per-frame detection here keeps the
    comparison apples-to-apples on that axis.

    Returns: (rows, width, height, fps, frame_count, detected_count)
    rows[i] is either None (no person detected that frame) or a dict
    {joint_name: (x_px, y_px, score)} in PIXEL coordinates (un-normalized;
    normalization to 0-1 happens at CSV-write time).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frame_idx = 0
    detected_count = 0
    t_start = time.time()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Detect the person bbox ourselves (rather than calling
            # wholebody(frame) directly) because rtmlib's RTMPose.__call__
            # silently falls back to running pose estimation on the WHOLE
            # FRAME as a fake bbox when the detector finds nobody (see
            # rtmlib/tools/pose_estimation/rtmpose.py: "if len(bboxes) == 0:
            # bboxes = [[0, 0, w, h]]"). That fallback would make every
            # frame look "detected" and silently defeat the detection-rate
            # comparison this evaluation depends on. Calling det_model
            # explicitly lets a true no-detection frame be recorded as such.
            bboxes = wholebody.det_model(frame)

            if len(bboxes) == 0:
                rows.append(None)
                frame_idx += 1
                continue

            # Solo runner on a treadmill: if more than one bbox is returned
            # (false positive in the background, reflection, etc.), keep the
            # largest one, mirroring MediaPipe's num_poses=1 config used by
            # runform.pose.
            if len(bboxes) > 1:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
                bboxes = [bboxes[int(np.argmax(areas))]]

            keypoints, scores = wholebody.pose_model(frame, bboxes=bboxes)
            kpts = keypoints[0]   # (133, 2) pixel coords
            scs = scores[0]       # (133,)

            row = {}
            for joint_name, idx in COCO_WHOLEBODY_INDEX.items():
                x, y = kpts[idx]
                s = float(np.clip(scs[idx], VIS_CLIP_MIN, VIS_CLIP_MAX))
                row[joint_name] = (float(x), float(y), s)
            rows.append(row)
            detected_count += 1
            frame_idx += 1

            if progress_every and frame_idx % progress_every == 0:
                elapsed = time.time() - t_start
                print(f"  ...{frame_idx} frames processed "
                      f"({elapsed:.1f}s, {elapsed / frame_idx:.2f}s/frame)",
                      flush=True)
    finally:
        cap.release()

    return rows, width, height, fps, frame_idx, detected_count


def write_landmarks_csv(rows, width, height, out_csv_path):
    """Write the CSV in runform.pose's exact column convention:
    frame,{joint}_x,{joint}_y,{joint}_z,{joint}_vis per joint (REQUIRED
    order). x/y normalized to 0-1 by frame width/height (matching
    MediaPipe's own normalized output, which runform.metrics.load_and_condition
    expects and re-scales by aspect ratio itself -- do NOT aspect-correct
    here, that step belongs to metrics.py only).

    z is always written as 0.0: rtmlib/COCO-WholeBody has no z output, and
    z is never read by runform.metrics (CLAUDE.md: "Ignore MediaPipe's z
    coordinate"), so a constant filler keeps the CSV shape identical to the
    real pipeline's output for easy diffing without inventing meaning.

    Per-frame no-detection rows are written fully blank (empty string for
    every column, so pandas.read_csv parses them as NaN) -- this matches
    runform.pose.extract_pose's own behavior of writing `None` for every
    landmark when result.pose_landmarks is empty for that frame.
    """
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame"]
        for name in REQUIRED_JOINTS:
            header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]
        writer.writerow(header)

        for frame_idx, row in enumerate(rows):
            out_row = [frame_idx]
            if row is None:
                out_row += [""] * (4 * len(REQUIRED_JOINTS))
            else:
                for name in REQUIRED_JOINTS:
                    src = JOINT_SOURCE[name]
                    x_px, y_px, vis = row[src]
                    out_row += [x_px / width, y_px / height, 0.0, vis]
            writer.writerow(out_row)


def process_clip(video_path, wholebody, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_csv_path = os.path.join(out_dir, f"{stem}_landmarks.csv")

    print(f"\n=== {video_path} ===")
    t0 = time.time()
    rows, width, height, fps, frames, detected = run_wholebody_on_clip(
        video_path, wholebody
    )
    elapsed = time.time() - t0
    print(f"Processed {frames} frames in {elapsed:.1f}s "
          f"({elapsed / frames:.2f}s/frame avg). "
          f"Detected in {detected}/{frames} frames ({detected / frames:.1%}).")

    write_landmarks_csv(rows, width, height, out_csv_path)
    print(f"Wrote {out_csv_path}")

    return {
        "video_path": video_path,
        "landmarks_csv_path": out_csv_path,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "detected_frames": detected,
        "detection_rate": detected / frames if frames else 0.0,
        "elapsed_s": elapsed,
        "s_per_frame": elapsed / frames if frames else None,
    }


def clip_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": float(fps), "width": width, "height": height, "frames": frames}


# ---------------------------------------------------------------------------
# A/B comparison against the existing MediaPipe pipeline
# ---------------------------------------------------------------------------
# These are the MediaPipe numbers already measured this session (via
# runform.pipeline._key_joint_visibility on the real MediaPipe landmark CSVs
# already sitting in out/), reproduced here so this script can render a
# direct before/after table without needing the old CSVs on hand. If the
# MediaPipe pipeline is re-run and these drift, update this dict --
# it is data, not a derived constant.
BASELINE_MEDIAPIPE = {
    "Smitas_5flat": {
        "speed_mph": 5,
        "left_worst_vis": 0.39, "right_worst_vis": 0.835,
        "left_strikes": 29, "right_strikes": 31,
        "cadence_spm": 185.8,
    },
    "Smitas_6flat": {
        "speed_mph": 6,
        "left_worst_vis": 0.404, "right_worst_vis": 0.845,
        "left_strikes": 27, "right_strikes": 27,
        "cadence_spm": 176.6,
    },
    "Smitas_8flat": {
        "speed_mph": 8,
        "left_worst_vis": 0.43, "right_worst_vis": 0.872,
        "left_strikes": 21, "right_strikes": 22,
        "cadence_spm": 165.7,
    },
}


def run_ab_comparison(clip_stats, out_md_path):
    """clip_stats: list of dicts with at least landmarks_csv_path, fps,
    width, height, detection_rate, elapsed_s, s_per_frame, video_path.

    Feeds each rtmlib-produced CSV through the UNMODIFIED
    runform.metrics.compute_metrics() and runform.pipeline._key_joint_visibility()
    -- imported directly from the installed runform package, never copied or
    reimplemented -- and renders a before/after markdown table against
    BASELINE_MEDIAPIPE.
    """
    # Lazy import so CSV generation (which doesn't need pandas/numpy pinned
    # to the main project's versions) can run even if this venv's versions
    # ever drift from runform's expectations; failures surface here, at
    # comparison time, with a clear traceback instead of at CLI startup.
    from runform.metrics import compute_metrics
    from runform.pipeline import _key_joint_visibility
    from runform.errors import RunFormError

    lines = []
    lines.append("# RTMPose (rtmlib Wholebody) vs MediaPipe -- A/B comparison\n")
    lines.append(
        "Generated by `scripts/prototype_rtmpose_pose.py`. Baseline "
        "(MediaPipe) numbers are the ones already measured this session via "
        "`runform.pipeline._key_joint_visibility` on the real clips; "
        "'after' numbers come from feeding the rtmlib-produced landmark "
        "CSVs through the SAME, unmodified `runform.metrics.compute_metrics` "
        "and `_key_joint_visibility` used by the existing pipeline.\n"
    )

    rows_for_summary = []

    for stat in clip_stats:
        stem = os.path.splitext(os.path.basename(stat["video_path"]))[0]
        baseline = BASELINE_MEDIAPIPE.get(stem)
        lines.append(f"\n## {stem}" + (f" ({baseline['speed_mph']} mph)" if baseline else "") + "\n")

        key_vis = _key_joint_visibility(stat["landmarks_csv_path"])
        left_worst = (key_vis.get("left") or {}).get("min")
        right_worst = (key_vis.get("right") or {}).get("min")

        metrics_error = None
        cadence = None
        left_strikes = right_strikes = None
        try:
            metrics = compute_metrics(
                stat["landmarks_csv_path"], fps=stat["fps"],
                width=stat["width"], height=stat["height"], smooth=9,
            )
            cadence = metrics.get("cadence_spm")
            per_side = metrics.get("per_side") or {}
            left_strikes = (per_side.get("left") or {}).get("strikes_detected")
            right_strikes = (per_side.get("right") or {}).get("strikes_detected")
        except RunFormError as e:
            metrics_error = str(e)

        lines.append("| Metric | MediaPipe (before) | RTMPose/rtmlib (after) |")
        lines.append("|---|---|---|")
        if baseline:
            lines.append(f"| Left worst-joint visibility | {baseline['left_worst_vis']} | "
                          f"{left_worst if left_worst is not None else 'N/A'} |")
            lines.append(f"| Right worst-joint visibility | {baseline['right_worst_vis']} | "
                          f"{right_worst if right_worst is not None else 'N/A'} |")
            lines.append(f"| Left strikes detected | {baseline['left_strikes']} | "
                          f"{left_strikes if left_strikes is not None else 'N/A'} |")
            lines.append(f"| Right strikes detected | {baseline['right_strikes']} | "
                          f"{right_strikes if right_strikes is not None else 'N/A'} |")
            lines.append(f"| Cadence (spm) | {baseline['cadence_spm']} | "
                          f"{cadence if cadence is not None else 'N/A'} |")
        else:
            lines.append(f"| Left worst-joint visibility | -- | {left_worst} |")
            lines.append(f"| Right worst-joint visibility | -- | {right_worst} |")
            lines.append(f"| Left strikes detected | -- | {left_strikes} |")
            lines.append(f"| Right strikes detected | -- | {right_strikes} |")
            lines.append(f"| Cadence (spm) | -- | {cadence} |")
        lines.append(f"| Detection rate (person found at all) | -- | "
                      f"{stat['detection_rate']:.1%} ({stat['detected_frames']}/{stat['frames']} frames) |")
        lines.append(f"| Processing time | -- | "
                      f"{stat['elapsed_s']:.1f}s total, {stat['s_per_frame']:.2f}s/frame |")
        if metrics_error:
            lines.append(f"\n**compute_metrics() raised MetricsError:** {metrics_error}\n")

        rows_for_summary.append({
            "stem": stem,
            "speed_mph": baseline["speed_mph"] if baseline else None,
            "cadence_before": baseline["cadence_spm"] if baseline else None,
            "cadence_after": cadence,
            "left_worst_before": baseline["left_worst_vis"] if baseline else None,
            "left_worst_after": left_worst,
        })

    # Physiological sanity check: does cadence now rise 5 -> 6 -> 8 mph?
    ordered = sorted(
        [r for r in rows_for_summary if r["speed_mph"] is not None],
        key=lambda r: r["speed_mph"],
    )
    lines.append("\n## Cadence-vs-speed sanity check\n")
    lines.append("Cadence should RISE as treadmill speed rises. MediaPipe showed the "
                  "opposite (physiologically backwards, symptomatic of missed strikes "
                  "on the poorly-tracked leg at higher speed).\n")
    lines.append("| Speed (mph) | Cadence before (MediaPipe) | Cadence after (RTMPose) |")
    lines.append("|---|---|---|")
    for r in ordered:
        lines.append(f"| {r['speed_mph']} | {r['cadence_before']} | "
                      f"{r['cadence_after'] if r['cadence_after'] is not None else 'N/A'} |")

    before_vals = [r["cadence_before"] for r in ordered]
    after_vals = [r["cadence_after"] for r in ordered]
    before_monotonic_up = all(b1 < b2 for b1, b2 in zip(before_vals, before_vals[1:]))
    if all(v is not None for v in after_vals):
        after_monotonic_up = all(a1 < a2 for a1, a2 in zip(after_vals, after_vals[1:]))
    else:
        after_monotonic_up = None
    lines.append(f"\nMediaPipe cadence rises monotonically with speed: **{before_monotonic_up}** "
                 f"(expected True for correct tracking).")
    lines.append(f"\nRTMPose cadence rises monotonically with speed: **{after_monotonic_up}**.\n")

    with open(out_md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote comparison report: {out_md_path}")
    return rows_for_summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run rtmlib Wholebody (RTMPose) on video(s); write "
                     "landmarks CSVs in runform's column convention, then "
                     "A/B them against the existing MediaPipe pipeline via "
                     "runform.metrics.compute_metrics / "
                     "runform.pipeline._key_joint_visibility (unmodified)."
    )
    p.add_argument("videos", nargs="+", help="input video file(s)")
    p.add_argument("--out-dir", default="out_rtmpose",
                   help="directory for output CSVs (default: out_rtmpose)")
    p.add_argument("--mode", default="balanced",
                   choices=("performance", "balanced", "lightweight"),
                   help="rtmlib Wholebody preset (default: balanced -- "
                        "same rtmw-x pose backbone as 'performance' but a "
                        "smaller 256x192 input, a reasonable speed/accuracy "
                        "tradeoff on CPU-only onnxruntime)")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"),
                   help="onnxruntime execution device (default: cpu -- see "
                        "scripts/rtmpose_ab_comparison.md for why CUDA "
                        "wasn't used on this machine)")
    p.add_argument("--summary-json", default=None,
                   help="optional path to dump per-clip run stats as JSON "
                        "(frame counts, timing, detection rate) for the "
                        "comparison step to consume")
    p.add_argument("--compare-only", action="store_true",
                   help="skip pose extraction; assume CSVs already exist in "
                        "--out-dir and just run the A/B comparison against "
                        "them (video files are still opened, briefly, to "
                        "read fps/width/height)")
    p.add_argument("--comparison-md", default="scripts/rtmpose_ab_comparison.md",
                   help="where to write the before/after markdown report")
    p.add_argument("--no-compare", action="store_true",
                   help="only generate CSVs, skip the comparison step")
    args = p.parse_args(argv)

    summaries = []
    if args.compare_only:
        # Recover real timing stats from any previous run's --summary-json
        # files sitting in out-dir (glob rather than a single fixed name,
        # since a multi-clip run may have been split across several
        # background invocations, each writing its own summary file).
        import glob
        prior_timing = {}
        for summary_file in glob.glob(os.path.join(args.out_dir, "*_run_summary*.json")):
            try:
                with open(summary_file) as f:
                    for entry in json.load(f):
                        prior_timing[os.path.basename(entry["video_path"])] = entry
            except (OSError, json.JSONDecodeError, KeyError):
                continue

        for video in args.videos:
            props = clip_video_props(video)
            stem = os.path.splitext(os.path.basename(video))[0]
            csv_path = os.path.join(args.out_dir, f"{stem}_landmarks.csv")
            import pandas as pd
            df = pd.read_csv(csv_path)
            detected = int(df["left_hip_x"].notna().sum())
            prior = prior_timing.get(os.path.basename(video), {})
            summaries.append({
                "video_path": video,
                "landmarks_csv_path": csv_path,
                "width": props["width"], "height": props["height"],
                "fps": props["fps"], "frames": props["frames"],
                "detected_frames": detected,
                "detection_rate": detected / props["frames"] if props["frames"] else 0.0,
                "elapsed_s": prior.get("elapsed_s", float("nan")),
                "s_per_frame": prior.get("s_per_frame", float("nan")),
            })
    else:
        wholebody = Wholebody(mode=args.mode, backend="onnxruntime",
                               device=args.device)
        for video in args.videos:
            summaries.append(process_clip(video, wholebody, args.out_dir))

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote run summary: {args.summary_json}")

    if not args.no_compare:
        run_ab_comparison(summaries, args.comparison_md)

    return summaries


if __name__ == "__main__":
    main()
