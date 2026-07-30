# Phase 0 validation report — real treadmill clips

Date: 2026-07-30. Clips: `Smitas_5flat.mov`, `Smitas_6flat.mov`,
`Smitas_8flat.mov` (portrait 1080x1920, ~30 fps, ~15-20 s each).
Metrics regenerated and byte-identical to `out/*_metrics.json` under the
current code, so all numbers below refer to those artifacts.

Methodology: no video player available, so every check was done by
extracting frames from `out/*_skeleton.mp4` with OpenCV and inspecting
them directly — contact-sheet montages of every detected strike frame,
frame-by-frame sequences around suspect regions and limb crossovers, and
x-t slice analysis of the belt surface for the speed check.

## 1. Clip labels are PACES per mile, and belt timing confirms them

The filenames encode pace per mile — `5flat` = 5:00/mile (12.0 mph),
`6flat` = 6:00/mile (10.0 mph), `8flat` = 8:00/mile (7.5 mph) — NOT
treadmill mph. Under pace semantics the 5 clip is the fastest and the 8
clip the slowest, and the belt measurement below confirms exactly that
ordering and, to within a few percent, the labeled magnitudes.

The treadmill belt has a visible white mark. Its passage period past a
fixed image band was measured per clip (background-subtracted x-t slice,
peak times fitted by linear regression; r² ≥ 0.999, sub-frame precision):

| file | belt revolution period | measured relative speed | expected from pace label | implied speed (6flat = 10.0 mph anchor) |
|---|---|---|---|---|
| `Smitas_5flat` | 16.84-16.89 frames | 1.66 | 1.60 | ~12.4 mph (~4:51/mile) |
| `Smitas_6flat` | 20.86 frames | 1.34 | 1.33 | 10.0 mph (6:00/mile) |
| `Smitas_8flat` | 27.7-28.1 frames | 1.00 | 1.00 | ~7.5 mph (~8:01/mile) |

Same belt, same fps in all clips, so speed ∝ 1/period. The lines are the
belt mark, not the runner's shadow: their period is metronomic, agrees
across 2-3 independent image bands per clip, is far from each clip's
stride period (18.98 / 20.29 / 21.69 frames), and drifts freely in phase
relative to the strike train (phase-lock R = 0.49 / 0.47 / 0.06 per
clip; a foot shadow would be locked near 1.0). Reproduce with
`scripts/belt_speed_check.py` — read its caution about band choice.

Consequences:

- The "cadence falls as speed rises" anomaly was an artifact of reading
  the filenames as mph. Against actual belt speed, cadence RISES
  165.7 → 176.6 → 185.8 spm and right-leg ground contact FALLS
  290 → 232 → 220 ms. Both are textbook-normal speed responses. Not a
  code bug, not a physiology surprise.
- This is also why the RTMPose backend swap (`rtmpose_ab_comparison.md`)
  improved visibility but couldn't move the cadence pattern — the pattern
  was never a pose defect.
- When feeding these clips to `python -m runform interpret`, convert the
  pace labels to actual speeds (e.g. m/s: ~5.5 / 4.5 / 3.35), fastest
  clip = `5flat`. Passing "5/6/8" as speed values would reverse every
  speed-profile slope.
- The 8:00 and 6:00 clips match their labels to ~1 s/mile; the 5:00 clip
  measures ~3-4% fast (~4:51/mile), which is either a runner slightly
  ahead of the belt setting or treadmill calibration error. Filming
  notes should record the console readout at capture time to settle
  such gaps in future sessions.

## 2. Manual ground-truth strike count (acceptance test)

Every detected strike frame was extracted and visually confirmed; the
whole clip was then scanned for strike-shaped moments the detector
missed (any combined-train gap > ~1.5x the step period, plus clip
edges).

| file | steps_detected | manual count | Δ | bar (±1) | cadence_spm | manual cadence | bar (±3) |
|---|---|---|---|---|---|---|---|
| `Smitas_5flat` | 60 | ~62 | 2 | **FAIL** | 185.8 | ~184.3 | pass |
| `Smitas_6flat` | 54 | 54 | 0 | pass | 176.6 | 176.6 | pass |
| `Smitas_8flat` | 43 | 43 | 0 | pass | 165.7 | 165.7 | pass |

For 6flat and 8flat every detection is a real strike, none are missed,
and L/R alternation is physically consistent across the full clip, so
the manual interval count equals the detector's.

The 5flat failure is localized to the last ~3 s of the clip, where pose
detection visibly degrades (the leg overlay disappears entirely around
frames 594-599). In that window the detector: missed left strikes at
~frames 521, 581, 601; fired one left strike ~8 frames early (L534 for a
real strike at ~542); and fired a spurious left strike at frame 592
coincident with the genuine right strike at 592 (far-leg landmarks had
collapsed onto the near leg). Net: 60 detected vs ~62 real. Left-count
uncertainty note: the far leg is often motion-blurred and sometimes
barely visible, so the manual left count carries ±1 uncertainty of its
own; the right-leg count is certain.

Cadence passes on all three clips (the interval estimator's in-band
filter rejects exactly the corrupted end-of-clip intervals).

## 3. Left/right leg swap at crossover — none found

Checked 6 crossover events per clip (18 sequences, ~130 frames), spread
across each clip, stepping through ±3 frames around each frame where the
two ankles' x-positions are closest, plus full-resolution spot checks.

- The blue (right/near-leg) chain never leaves the physically-near leg.
- No identity swap was observed in any sampled crossover.
- The actual failure mode is different: when the far (left) leg is
  occluded or blurred, its landmarks **collapse onto the near leg**
  rather than swapping with it. This corrupts left-side metric values
  (not identities) and is exactly what the `low_key_joint_visibility`
  flag catches (left knee/ankle visibility 0.39-0.46 in all clips).
- Detection runs in per-frame IMAGE mode, so there is no tracking state
  that could hold a swap across frames.

The BUILD_PLAN "stop and fix first" gate on leg swap therefore does not
trigger.

## 4. Verdict: re-shoot required for a full Phase 0 pass

What this footage CAN support: cadence, step count, and right-leg (near
side) metrics — validated above on 2/3 clips, with the third failing
only in a degraded 3-second tail.

What it CANNOT support: any left-side or asymmetry metric. Left knee /
ankle visibility is 0.39-0.46 in every clip, below the 0.5 usability
threshold; left ground-contact SD is up to 107 ms vs ~28 ms on the
right. No threshold tuning fixes a leg the camera can barely see.

Required for the re-shoot:

1. Log the console speed per clip in filming notes at capture time (the
   labels checked out this time, but the ~4% gap on the 5:00 clip was
   only resolvable because the belt mark happened to be measurable).
2. Light the far side of the body, or accept right-leg-only metrics.
   More light + faster shutter would also reduce the motion blur that
   degrades the fast clip's tail.
3. Keep the runner fully in frame for the whole clip (the 5flat tail
   degradation coincides with drift toward the frame edge / shadow).

Threshold note (`MIN_SWING_RATIO`, `PROMINENCE_SD_RATIO`, `--smooth`):
the current values produced exact strike counts on the two clean clips
and correct events everywhere tracking was usable, so no retuning was
done. They remain provisional for degraded footage.

---

# Addendum 2026-07-30: `british_guy_treadmill.mp4` — leg-identity swap found

New validation clip: different runner, treadmill, and gym. 534x574 px,
30 fps, 177 frames (5.9 s) — much smaller and shorter than the Smitas
clips; likely a social-media re-encode. Speed unknown; the runner's pace
was estimated at ~4:40/mile by eye, and nothing in this analysis could
verify or refute that (see belt check below). Clean sagittal framing,
well lit on both sides, runner fully in frame, facing screen-LEFT (the
Smitas clips face right).

## Jump cut and segmentation

The clip contains one hard splice at frame 79->80 (frame-diff 6.4x the
median; visually confirmed — camera reframes and the gait phase is
discontinuous). No other cuts (secondary diff spikes at 74-77 and
167-168 are continuous motion). Gait intervals must never span a splice,
so the clip was split and each segment analyzed independently:
`british_guy_segA.mp4` = frames 0-79 (2.67 s), `british_guy_segB.mp4` =
frames 80-176 (3.23 s). Both are under `MIN_CLIP_SECONDS`, so both carry
the `short_clip` flag by design.

## Surface quality is the best yet — which makes the failure worse

Both segments: 100% detection rate, worst-key-joint visibility left
0.93 / 0.92 and right 0.84 / 0.83 — both sides clear the 0.6 gate that
the Smitas far leg failed. On paper this is the first footage on which
per-side and asymmetry metrics are validatable. They are not, because:

## 1. MediaPipe swaps left/right leg identity at essentially every crossover

This is the exact failure mode the BUILD_PLAN risk register calls
"stop and fix that first", observed for the first time. On the Smitas
clips the far-leg failure mode was landmark collapse; here the labels
detach from the physical legs entirely.

Evidence (reproducible from `out/british_guy_seg{A,B}_landmarks.csv`):

- **Sign test.** In genuine gait, `left_ankle_x - right_ankle_x` flips
  sign once per step (~ every 9.8 frames here) and spends ~50% of frames
  each side of zero. Measured: the "left" ankle is the FRONT ankle on
  76/80 frames (segA) and 93/97 (segB); the separation dips to near zero
  every ~9.8 frames and bounces back without crossing. The anatomical
  labels are tracking the spatial roles front-leg / rear-leg, flipping
  identity at each crossover.
- **Visual confirmation** (skeleton overlay, segA frames 2-13): the foot
  planted at the front carries the red (left) chain; after it rotates
  under the body to the rear it carries the blue (right) chain, while
  the airborne foot crossing to the front picks up red. Same physical
  foot, both colors, within 6 frames.

Consequences observed in the emitted metrics:

- segA `cadence_spm` = **327.3** — physically impossible, and the purest
  symptom. The two per-side strike trains are step-periodic aliases
  offset by ~5 frames, so the combined train has paired intervals
  (5,14,6,13,...) and the in-band mean lands on the pair spacing. (True
  cadence, from crossover timing and manual count: ~183 spm.)
- All per-side values are front-leg-vs-rear-leg, not left-vs-right:
  e.g. segA "left" overstride 0.63 vs "right" -0.30 is the geometry of
  a foot at touchdown vs a foot behind the hip, not asymmetry. The
  `asymmetry_pct` block (200% on overstride) is meaningless.
- segB cadence (187.0) came out plausible **by luck** of interval
  distribution — a corrupted clip can still emit a sane-looking number.

**No existing quality gate catches this.** Detection 100%, visibility
0.83+, no flags beyond `short_clip`. The pipeline currently has no
defense against a clip whose per-side data is garbage while every
quality signal reads excellent.

## 2. Acceptance test: manual ground-truth strike count

Every frame of both skeleton videos was inspected via contact-sheet
montages (same methodology as the main report). Manual touchdown frames:

- segA: ~10, ~20, ~30, ~40, ~49, ~59, ~69, ~79 (a foot is already
  planted at frame 0 — clip-edge stance, strike predates the segment).
  8 touchdowns, mean step period 9.86 frames -> **182.6 spm**.
- segB: ~4, ~14, ~24, ~33, ~43, ~53, ~62, ~72, ~81, ~91.
  10 touchdowns, mean step period 9.67 frames -> **186.2 spm**.

| segment | steps_detected | manual | bar (±1) | cadence_spm | manual | bar (±3) |
|---|---|---|---|---|---|---|
| segA | 8 | 8 | pass | 327.3 | ~182.6 | **FAIL (by ~145)** |
| segB | 9 | 10 | pass | 187.0 | ~186.2 | pass (by luck — see above) |

The total step count survives the swap (each physical touchdown is
captured once by whichever labeled train claims it), which means
`steps_detected` alone cannot reveal the corruption either.

## 3. Model/mode variants do not fix the swap

Sign-test on segA landmarks per variant (`pos/neg` = frames with left
ankle behind/in front; genuine alternation would be ~40/40 with ~8
sustained flips):

| variant | dx sign pos/neg | sign changes | cadence_spm |
|---|---|---|---|
| lite / image (default) | 4 / 76 | 6 | 327.3 |
| heavy / image | 13 / 67 | 15 | 182.6 |
| lite / video | 1 / 79 | 2 | 327.3 |
| heavy / video | 21 / 55 | 16 | 183.1 |

Heavy reduces swap frequency and its cadence happens to land near truth,
but the labels still sit on the front leg ~3/4 of the time — per-side
metrics remain garbage in every variant.

**RTMPose (rtmlib Wholebody) holds identity where MediaPipe cannot.**
The prototype backend (`scripts/prototype_rtmpose_pose.py`, run in a
throwaway venv; landmarks in `out_rtmpose_bgt/`) was fed the same two
segments and its CSVs pushed through the same unmodified metrics code:

| segment | dx sign pos/neg | sign changes | cadence_spm | manual cadence |
|---|---|---|---|---|
| segA | **40 / 40** | **8** | 183.1 | ~182.6 |
| segB | 51 / 46 | 11 | 195.2 | ~186.2 |

That is textbook-genuine alternation: balanced sign split with one
sustained flip per step (8 flips = 8 touchdowns on segA). segA cadence
lands within 1 spm of manual. Two honest caveats: segB cadence reads
9 spm high (interval-filter artifact on a 10-step sample — small-n, not
identity corruption), and RTMPose's ground-contact times (~275-300 ms)
run far longer than MediaPipe's near-leg values on comparable footage,
so its event *timing* would need its own validation before adoption.
The earlier "DO NOT promote RTMPose" verdict (`rtmpose_ab_comparison.md`)
was based on the Smitas clips, where MediaPipe held identity; on
footage where MediaPipe swaps, that verdict deserves re-examination —
identity integrity is a harder requirement than per-joint visibility.

## 4. What did validate on this footage

- **Direction detection**: `"left"` on both segments — first validation
  of the left-facing case (Smitas clips are all right-facing).
- **Trunk lean sign**: positive (~10-11 deg toward travel) with the
  runner visibly leaning forward — the trunk-lean sign fix holds for
  left-facing footage. (Values are front/rear-contaminated per above;
  only the sign is being credited here.)
- **Jump-cut handling by segmentation**: no detected event interval
  spans the splice, by construction.

## 5. Belt speed check: no measurable mark

Six candidate bands on the belt surface were tried with
`belt_speed_check.py` on both segments. Every band either found too few
consistent passages or locked onto a period of 9.6-9.7 frames — within
2.5% of the step period (9.8), i.e. the runner's own gait/shadow, which
the script's caution note explicitly says to reject. This treadmill
shows no usable belt mark at this resolution, so the ~4:40/mile estimate
stays **unverified**. The measured cadence (~183-186 spm) is consistent
with a trained runner at that pace but does not confirm it.

## 6. Verdict and required follow-up

Phase 0 on this clip: **FAIL — the leg-swap gate triggers.** The Smitas
finding ("no swap at 18 sampled crossovers") was real but does not
generalize: swap behavior is footage-dependent, and the failure arrived
on the *highest*-quality footage yet by every existing quality signal.
Candidate contributing factors (unverified): low resolution, left-facing
runner, and both legs being equally well lit — the symmetric appearance
removes the cue that kept labels anchored on the Smitas clips.

Required before per-side/asymmetry metrics can be trusted from any
footage:

1. **Add an automated leg-swap flag to Phase 1** — the sign test above
   is cheap and decisive: fraction of frames on which
   `sign(left_ankle_x - right_ankle_x)` equals its own median sign.
   Genuine gait ~50-60%; swapped tracking >90%. Flag, and refuse
   per-side/asymmetry output, above ~75-80%. A cadence sanity ceiling
   (impossible `cadence_spm`) would have caught segA independently.
2. Re-evaluate the pose backend against THIS clip class (small,
   re-encoded, left-facing), not only the Smitas clips.
3. Until then, treat per-side and asymmetry output from any new footage
   as unvalidated regardless of visibility scores.
