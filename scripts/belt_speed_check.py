"""Verify a treadmill clip's belt speed from the footage itself.

The belt's surface marks (seam, wear smudges) pass a fixed image band
once per belt revolution. Measuring that period per clip gives relative
belt speed (speed is proportional to 1/period for the same belt and fps),
which catches mislabeled clips before their nominal speeds are fed to
`runform interpret`. This caught the 2026-07 clip set being speed-reversed
relative to its filenames (see phase0_validation.md).

Method: average a thin pixel band on the belt each frame, subtract the
per-column temporal median to remove static content, collapse each frame
to a single "mark is passing" score, then fit passage time against
passage index by linear regression for a sub-frame period estimate.

Usage:
    python scripts/belt_speed_check.py clip.mov --band 1650 1720 140 400

Pick the band on the belt surface, away from where the feet land, from a
frame screenshot. Verify the fit is trustworthy (r^2 near 1, several
passages) before believing the period.

CAUTION — a perfect fit does not prove the period is the belt's. A band
swept by the runner's feet or their shadow measures the STRIDE period
instead, also with r^2 near 1. Always (a) measure 2-3 different bands
and take the consensus, and (b) reject any candidate period within ~5%
of the clip's stride period (60 * fps * 2 / cadence_spm frames) unless
independently confirmed. The belt mark's phase drifts freely relative to
foot strikes; anything phase-locked to the gait is not the belt.
"""
import argparse

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.stats import linregress

# A passage peak must clear this many SDs of the score signal to count.
# Below it, shadow flicker starts producing false passages.
PASSAGE_PROMINENCE_SDS = 1.0

# Reject fits with fewer passages than this: the regression period is
# only sub-frame accurate when it spans many revolutions.
MIN_PASSAGES = 6


def belt_period(video_path, band, start_frame=30, max_frames=1200):
    y0, y1, x0, x1 = band
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows = []
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        strip = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        rows.append(strip.astype(np.float32).mean(axis=0))
    cap.release()
    if len(rows) < 60:
        raise SystemExit(f"Only {len(rows)} frames read — clip too short or band invalid.")

    xt = np.array(rows)
    xt -= np.median(xt, axis=0, keepdims=True)
    score = np.percentile(np.abs(xt), 95, axis=1)
    # Temporal high-pass: mark passages are 1-2 frames wide, while the
    # runner's shadow drifting across the band varies over tens of
    # frames. Removing a rolling median keeps the spikes and drops the
    # shadow, which otherwise out-prominences the real passages.
    score = score - median_filter(score, size=9)

    # Two passes: strong peaks first to learn the approximate period,
    # then a re-detect with a minimum spacing just under it. Without the
    # spacing guard, the runner's shadow drifting over the band produces
    # junk peaks that bias the fitted period while still leaving r^2
    # deceptively high.
    strong, _ = find_peaks(score, prominence=1.5 * np.std(score))
    if len(strong) < MIN_PASSAGES:
        raise SystemExit(
            f"Only {len(strong)} strong mark passages found — band may be "
            "occluded or the belt has no visible mark there. Try a different band."
        )
    approx = float(np.median(np.diff(strong)))
    peaks, _ = find_peaks(score, distance=max(int(0.7 * approx), 1),
                          prominence=PASSAGE_PROMINENCE_SDS * np.std(score))

    # Median spacing anchors the revolution index of each passage, so
    # passages missed under the runner's shadow don't corrupt the fit.
    # Junk spikes IN the shadow region land at arbitrary phases, so the
    # fit is refined by discarding residual outliers and refitting — the
    # true passages agree with each other to well under a frame.
    t = peaks.astype(float)
    idx = np.round((t - t[0]) / approx).astype(int)
    for _ in range(3):
        fit = linregress(idx, t)
        residuals = np.abs(t - (fit.intercept + fit.slope * idx))
        keep = residuals <= 0.15 * approx
        if keep.all():
            break
        t, idx = t[keep], idx[keep]
        if len(t) < MIN_PASSAGES:
            raise SystemExit(
                f"Too few consistent passages left after outlier rejection "
                f"({len(t)}) — band is probably picking up something besides "
                "the belt mark."
            )
    return fit.slope, fit.rvalue ** 2, len(t)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("videos", nargs="+", help="clip(s) to measure")
    p.add_argument("--band", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"),
                   default=[1650, 1720, 140, 400],
                   help="pixel band on the belt surface, away from footfalls")
    p.add_argument("--start", type=int, default=30, help="first frame to use")
    args = p.parse_args()

    results = []
    for v in args.videos:
        period, r2, n = belt_period(v, args.band, args.start)
        results.append((v, period, r2, n))
        print(f"{v}: revolution period {period:.2f} frames "
              f"(r^2={r2:.4f}, {n} passages)")

    if len(results) > 1:
        slowest = max(r[1] for r in results)
        print("\nRelative belt speed (slowest clip = 1.00):")
        for v, period, _, _ in results:
            print(f"  {v}: {slowest / period:.2f}")


if __name__ == "__main__":
    main()
