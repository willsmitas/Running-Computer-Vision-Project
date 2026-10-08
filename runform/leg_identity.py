"""Leg-identity tracking: carry left/right leg labels through a clip by
motion continuity, so a pose backend that names legs by front/rear role
cannot corrupt per-side metrics.

The failure this fixes (scripts/phase0_validation.md addendum): the pose
model finds both legs at the right positions, but at limb crossover it
attaches the "left" label to whichever foot is in front. That is a
LABELING error, not a detection error, so it can be repaired after pose
estimation without touching the model:

  1. Seed. Pick a frame where both legs are clearly visible and well
     apart, and mark one foot red (left) and one blue (right). By default
     the model's own labels on that frame are trusted; the web UI lets the
     runner override with one click (`swap_seed`).
  2. Track. Every other frame, give each label to the foot nearest where
     that labeled foot was heading. At a crossover the two feet overlap in
     position but not in motion: relative to the hip, the planted foot
     moves backward with the belt while the swinging foot moves forward
     fast. A constant-velocity prediction is what carries identity through
     the crossing. The whole leg chain moves as a unit.
  3. Never guess across a blind span. When a foot's dot is not visible,
     its label is assigned to nothing until the dot is visible again, and
     it is re-acquired only when that is unambiguous. Those frames are
     written as unlabeled (NaN, visibility 0) and reported, so metrics can
     drop events at their edges rather than read interpolated positions.

Pure numpy/pandas. Values move between existing CSV columns; the column
order (runform/landmarks.py) is never touched.

All thresholds below are provisional educated guesses, tuned only on
synthetic gait (CLAUDE.md). Validate on real footage before trusting them.
"""

import numpy as np
import pandas as pd

from .errors import MetricsError
from .landmarks import LANDMARK_NAMES
from .metrics import VIS_THRESHOLD  # same cut: below it, "the dot is not visible"


# Joints that swap as one unit when a leg's label is corrected. Hips stay
# put: in a side view they nearly coincide, and swapping them would move
# error into the body centre that every hip-relative signal depends on.
CHAIN_JOINTS = ("knee", "ankle", "heel", "foot_index")
# Joints whose position is matched against the track's prediction. The
# heel duplicates the ankle closely, so it adds weight without adding
# information.
MATCH_JOINTS = ("knee", "ankle", "foot_index")
# Joints that must be visible for a leg to count as "seen" this frame.
VISIBLE_JOINTS = ("knee", "ankle")

# Seed frame needs the ankles at least this far apart (in leg lengths) so
# the starting labels are unambiguous. The ankles of a running stride
# swing roughly +/-0.3-0.5 leg lengths about the hip, so 0.25 rejects the
# crossover band without excluding most of the cycle.
SEED_SEPARATION = 0.25

# Velocity is estimated from the track's observations within this many
# frames. Short enough to follow the foot's reversal at strike and
# toe-off, long enough to average out one frame of pose jitter.
VEL_WINDOW = 3

# A track not observed for more than this many frames is "lost": a
# constant-velocity extrapolation over longer than ~half a step (~5
# frames at 180 spm, 30 fps) points nowhere useful.
MAX_PREDICT_FRAMES = 6

# A candidate further than this (mean per-joint distance, leg lengths)
# from a track's prediction is not that leg. Roughly the distance a swing
# foot can cover in two frames at sprint speed.
MATCH_GATE = 0.35

# Re-acquiring a lost leg requires the other leg's match to be this much
# better (leg lengths) one way than the other. Below it the two feet are
# too close to tell apart, so neither label locks.
AMBIGUITY_MARGIN = 0.05

# Fraction of labeled frames on which the left ankle stays on the same
# side of the right ankle. Genuine alternation sits near 0.5-0.6; labels
# stuck to front/rear roles read 0.9+ (76/80 on the MediaPipe swap clip).
# Above this, per-side data is front-leg-vs-rear-leg, not left-vs-right.
STUCK_SIGN_FRACTION = 0.78

# Unlabeled runs shorter than this are crossover hesitations: one or two
# ambiguous frames that the existing 5-frame gap fill bridges harmlessly.
# Longer runs are blind spans that metrics must exclude events around.
MIN_BLIND_SPAN_FRAMES = 3

_SIDES = ("left", "right")

# Shared-scheme guard, as in metrics.py: every joint moved here must exist
# in the scheme pose.py writes.
assert {f"{s}_{j}" for s in _SIDES for j in CHAIN_JOINTS + ("hip",)} <= set(LANDMARK_NAMES), \
    "leg_identity joints not in landmark scheme"


def _cols(side, joint):
    return [f"{side}_{joint}_{a}" for a in ("x", "y", "z", "vis")]


def _chain_cols(side):
    return [c for j in CHAIN_JOINTS for c in _cols(side, j)]


def _hip_centre(df, aspect):
    hx = df[["left_hip_x", "right_hip_x"]].mean(axis=1).to_numpy() * aspect
    hy = df[["left_hip_y", "right_hip_y"]].mean(axis=1).to_numpy()
    return hx, hy


def _chain_points(df, side, aspect, hx, hy):
    """(n, len(MATCH_JOINTS), 2) hip-relative, aspect-corrected points."""
    pts = []
    for j in MATCH_JOINTS:
        x = df[f"{side}_{j}_x"].to_numpy(dtype=float) * aspect - hx
        y = df[f"{side}_{j}_y"].to_numpy(dtype=float) - hy
        pts.append(np.stack([x, y], axis=1))
    return np.stack(pts, axis=1)


def _chain_visible(df, side, pts):
    vis = np.ones(len(df), dtype=bool)
    for j in VISIBLE_JOINTS:
        v = df[f"{side}_{j}_vis"].to_numpy(dtype=float)
        vis &= np.nan_to_num(v, nan=0.0) >= VIS_THRESHOLD
    vis &= np.isfinite(pts).all(axis=(1, 2))
    return vis


def _leg_length(df, aspect):
    """Median hip->knee->ankle length over frames where a side is visible."""
    lengths = []
    for side in _SIDES:
        ok = np.ones(len(df), dtype=bool)
        for j in ("hip", "knee", "ankle"):
            ok &= np.nan_to_num(df[f"{side}_{j}_vis"].to_numpy(dtype=float)) >= VIS_THRESHOLD

        def p(j):
            return np.stack([df[f"{side}_{j}_x"].to_numpy(dtype=float) * aspect,
                             df[f"{side}_{j}_y"].to_numpy(dtype=float)], axis=1)

        hip, knee, ankle = p("hip"), p("knee"), p("ankle")
        seg = np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1)
        lengths.append(seg[ok & np.isfinite(seg)])
    allv = np.concatenate(lengths)
    return float(np.median(allv)) if len(allv) else float("nan")


class _Track:
    """One labeled leg: recent observations -> constant-velocity prediction."""

    def __init__(self):
        self.obs = []  # (frame, points)

    def observe(self, frame, pts):
        self.obs.append((frame, pts))
        if len(self.obs) > VEL_WINDOW + 1:
            self.obs.pop(0)

    def lost(self, frame):
        return not self.obs or abs(frame - self.obs[-1][0]) > MAX_PREDICT_FRAMES

    def predict(self, frame):
        f_last, p_last = self.obs[-1]
        recent = [(f, p) for f, p in self.obs if abs(f_last - f) <= VEL_WINDOW]
        if len(recent) < 2:
            return p_last
        # Least-squares slope over the window, not an endpoint
        # difference: one jittery frame at either end otherwise swings
        # the prediction enough to flip a crossover (synthetic tests at
        # ~4% leg-length jitter).
        fs = np.array([f for f, _ in recent], dtype=float)
        ps = np.stack([p for _, p in recent])
        fc = fs - fs.mean()
        vel = np.tensordot(fc, ps - ps.mean(axis=0), axes=(0, 0)) / np.sum(fc ** 2)
        return ps.mean(axis=0) + vel * (frame - fs.mean())

    def dist(self, frame, pts, leg_len):
        pred = self.predict(frame)
        return float(np.mean(np.linalg.norm(pts - pred, axis=1))) / leg_len


def _track(order, cand, vis, leg_len, start_map):
    """Run the tracker over `order` (frame indices, seed frame first).

    cand[c][f] -> points of raw chain c (0 = model's left, 1 = model's
    right) at frame f; vis[c][f] -> whether that chain is visible.
    start_map[t] = raw chain carrying track t (0 = red/left, 1 =
    blue/right) on the seed frame.

    Returns {frame: (src_for_track0, src_for_track1)} where src is a raw
    chain index, or None when that track is unlabeled on that frame.
    """
    tracks = [_Track(), _Track()]
    out = {}
    seed = order[0]
    for t in (0, 1):
        tracks[t].observe(seed, cand[start_map[t]][seed])
    out[seed] = (start_map[0], start_map[1])

    for f in order[1:]:
        seen = [c for c in (0, 1) if vis[c][f]]
        active = [t for t in (0, 1) if not tracks[t].lost(f)]
        assign = [None, None]

        if not active:
            # Both legs blind for longer than prediction is meaningful:
            # identity is gone. Labels stay off for the rest of this pass
            # rather than re-seeding on a guess.
            out[f] = (None, None)
            continue

        d = {(t, c): tracks[t].dist(f, cand[c][f], leg_len)
             for t in active for c in seen}

        if len(active) == 2 and len(seen) == 2:
            keep = d[(0, 0)] + d[(1, 1)]
            swap = d[(0, 1)] + d[(1, 0)]
            m = (0, 1) if keep <= swap else (1, 0)
            # Too close to call (feet overlapping mid-crossover): label
            # neither and let both predictions coast through. Committing
            # on a coin flip is exactly how identity gets lost.
            if abs(keep - swap) < AMBIGUITY_MARGIN:
                m = (None, None)
            for t in (0, 1):
                if m[t] is None:
                    continue
                if d[(t, m[t])] <= MATCH_GATE:
                    assign[t] = m[t]
        elif len(active) == 1 and len(seen) == 2:
            # One leg lost. Re-acquire it only by elimination: the leg we
            # still have must clearly match one candidate, and the lost
            # leg takes the other. Ambiguous -> neither label locks.
            t = active[0]
            near, far = sorted(seen, key=lambda c: d[(t, c)])
            if d[(t, near)] <= MATCH_GATE and d[(t, far)] - d[(t, near)] >= AMBIGUITY_MARGIN:
                assign[t] = near
                assign[1 - t] = far
        elif len(seen) == 1:
            # One dot visible. It belongs to a track only if exactly one
            # active track claims it clearly; the hidden leg stays
            # unlabeled until its dot is visible again.
            c = seen[0]
            ranked = sorted(active, key=lambda t: d[(t, c)])
            best = ranked[0]
            clear = len(ranked) == 1 or d[(ranked[1], c)] - d[(best, c)] >= AMBIGUITY_MARGIN
            if d[(best, c)] <= MATCH_GATE and clear:
                assign[best] = c
        # len(seen) == 0: nothing visible, nothing assigned.

        for t in (0, 1):
            if assign[t] is not None:
                tracks[t].observe(f, cand[assign[t]][f])
        out[f] = tuple(assign)
    return out


def _find_seed(vis, cand_ankle_x, leg_len):
    both = vis[0] & vis[1]
    sep = np.abs(cand_ankle_x[0] - cand_ankle_x[1]) / leg_len
    ok = np.flatnonzero(both & (np.nan_to_num(sep) >= SEED_SEPARATION))
    return int(ok[0]) if len(ok) else None


def sign_stuck_fraction(df, mask=None):
    """Fraction of frames where sign(left_ankle_x - right_ankle_x) equals
    its own median sign. ~0.5 for genuine alternation; ~1.0 when labels
    track front/rear roles. None if too few frames to judge.
    """
    dx = (df["left_ankle_x"] - df["right_ankle_x"]).to_numpy(dtype=float)
    ok = np.isfinite(dx) & (dx != 0)
    if mask is not None:
        ok &= mask
    s = np.sign(dx[ok])
    if len(s) < 10:
        return None
    med = 1.0 if np.median(s) >= 0 else -1.0
    return round(float(np.mean(s == med)), 3)


def _spans(mask):
    """[(start, end_inclusive)] runs of True in a boolean array."""
    spans, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def relabel(df, width, height, swap_seed=False):
    """Raw landmarks DataFrame -> (tracked DataFrame, identity report).

    swap_seed=True says the model's labels on the seed frame are backwards
    (the runner's click in the UI): the red/left track starts on the
    model's "right" chain.
    """
    aspect = width / height
    n = len(df)
    out = df.copy()
    report = {
        "seed_frame": None, "swap_seed": bool(swap_seed),
        "frames": n, "swaps_corrected": 0,
        "labeled_fraction": {"left": 0.0, "right": 0.0},
        "unlabeled_spans": {"left": [], "right": []},
        "sign_stuck_fraction_raw": sign_stuck_fraction(df),
        "sign_stuck_fraction": None,
    }
    if n == 0:
        return out, report

    hx, hy = _hip_centre(df, aspect)
    cand = [_chain_points(df, s, aspect, hx, hy) for s in _SIDES]
    vis = [_chain_visible(df, s, cand[i]) for i, s in enumerate(_SIDES)]
    leg_len = _leg_length(df, aspect)
    ankle_x = [cand[i][:, MATCH_JOINTS.index("ankle"), 0] for i in (0, 1)]

    seed = _find_seed(vis, ankle_x, leg_len) if np.isfinite(leg_len) and leg_len > 0 else None
    if seed is None:
        # No frame where both legs are clearly visible and apart: there is
        # nothing to anchor identity to. Leave every leg unlabeled.
        assignment = {f: (None, None) for f in range(n)}
    else:
        report["seed_frame"] = seed
        start_map = (1, 0) if swap_seed else (0, 1)
        fwd = _track(list(range(seed, n)), cand, vis, leg_len, start_map)
        bwd = _track(list(range(seed, -1, -1)), cand, vis, leg_len, start_map)
        assignment = {**bwd, **fwd}

    raw = {s: df[_chain_cols(s)].to_numpy(dtype=float) for s in _SIDES}
    blank = np.full(len(_chain_cols("left")), np.nan)
    vis_idx = [i for i, c in enumerate(_chain_cols("left")) if c.endswith("_vis")]
    new = {s: np.empty_like(raw[s]) for s in _SIDES}
    unlabeled = {s: np.zeros(n, dtype=bool) for s in _SIDES}
    swapped = np.zeros(n, dtype=bool)
    for f in range(n):
        src = assignment[f]
        for t, side in enumerate(_SIDES):
            c = src[t]
            if c is None:
                row = blank.copy()
                row[vis_idx] = 0.0
                unlabeled[side][f] = True
            else:
                row = raw[_SIDES[c]][f]
                if c != t:
                    swapped[f] = True
            new[side][f] = row
    for side in _SIDES:
        out[_chain_cols(side)] = new[side]

    # Count corrections as transitions into a swapped state, not frames.
    report["swaps_corrected"] = int(np.sum(swapped[1:] & ~swapped[:-1]) + swapped[0])
    for side in _SIDES:
        report["labeled_fraction"][side] = round(float(1 - unlabeled[side].mean()), 3)
        report["unlabeled_spans"][side] = [
            [int(a), int(b)] for a, b in _spans(unlabeled[side])
            if b - a + 1 >= MIN_BLIND_SPAN_FRAMES
        ]
    report["sign_stuck_fraction"] = sign_stuck_fraction(
        out, mask=~(unlabeled["left"] | unlabeled["right"])
    )
    return out, report


def relabel_csv(in_csv, out_csv, width, height, swap_seed=False):
    """CSV -> tracked CSV (same columns, same order) + identity report."""
    df = pd.read_csv(in_csv)
    missing = [c for s in _SIDES for c in _chain_cols(s) + _cols(s, "hip")
               if c not in df.columns]
    if missing:
        raise MetricsError(
            f"Landmarks CSV is missing leg columns {missing[:4]}... — was it "
            f"produced by runform.pose?"
        )
    out, report = relabel(df, width, height, swap_seed=swap_seed)
    out.to_csv(out_csv, index=False)
    return report
