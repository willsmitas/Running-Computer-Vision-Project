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
