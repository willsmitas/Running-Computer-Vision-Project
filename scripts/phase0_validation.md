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

## 1. Headline finding: the clip filenames are speed-REVERSED

The treadmill belt has a visible white mark. Its passage period past a
fixed image band was measured per clip (background-subtracted x-t slice,
peak times fitted by linear regression; r² ≥ 0.999, sub-frame precision):

| file | belt revolution period | relative belt speed | actual speed if slowest = 5.0 mph |
|---|---|---|---|
| `Smitas_5flat` | 16.84-16.89 frames | 1.66 | **~8.1 mph (fastest)** |
| `Smitas_6flat` | 20.86 frames | 1.34 | **~6.6 mph** |
| `Smitas_8flat` | 27.7-28.1 frames | 1.00 | **~5.0 mph (slowest)** |

Same belt, same fps in all clips, so speed ∝ 1/period. The lines are the
belt mark, not the runner's shadow: their period is metronomic, agrees
across 2-3 independent image bands per clip, is far from each clip's
stride period (18.98 / 20.29 / 21.69 frames), and drifts freely in phase
relative to the strike train (phase-lock R = 0.49 / 0.47 / 0.06 per
clip; a foot shadow would be locked near 1.0). Reproduce with
`scripts/belt_speed_check.py` — read its caution about band choice.

Consequences:

- The "cadence falls as speed rises" anomaly (185.8 → 176.6 → 165.7 spm
  across files named 5 → 6 → 8) is **not** a code bug and not
  physiologically backwards. Against actual belt speed, cadence RISES
  165.7 → 176.6 → 185.8 spm and right-leg ground contact FALLS
  290 → 232 → 220 ms. Both are textbook-normal speed responses.
- This is also why the RTMPose backend swap (`rtmpose_ab_comparison.md`)
  improved visibility but couldn't move the cadence pattern — the pattern
  was never a pose defect.
- **Do not pass the filename speeds to `python -m runform interpret`.**
  Until the clips are re-filmed or re-labeled with verified speeds, any
  speed-profile slope computed from these labels has the wrong sign.
- The middle clip measures ~6.5-6.7 mph against a 5.0 anchor, so either
  it was run slightly above 6.0 or the treadmill's calibration differs
  from true speed by a few percent. Filming notes should record the
  console speed at capture time to settle this.

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

1. Record and verify the actual belt speed per clip (film the console,
   or log it in filming notes at capture time). Name files after
   verified speeds.
2. Light the far side of the body, or accept right-leg-only metrics.
   More light + faster shutter would also reduce the motion blur that
   degrades the fast clip's tail.
3. Keep the runner fully in frame for the whole clip (the 5flat tail
   degradation coincides with drift toward the frame edge / shadow).

Threshold note (`MIN_SWING_RATIO`, `PROMINENCE_SD_RATIO`, `--smooth`):
the current values produced exact strike counts on the two clean clips
and correct events everywhere tracking was usable, so no retuning was
done. They remain provisional for degraded footage.
