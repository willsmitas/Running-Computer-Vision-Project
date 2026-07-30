# Sports2D evaluation

Evaluation only, per CLAUDE.md's working-style rules (ask before refactoring,
no new deps in `requirements.txt`) -- nothing under `runform/` was touched or
imported-and-modified. `sports2d` was installed into a standalone venv
(`Sports2D` 0.8.34, BSD-3-Clause) and run against `Smitas_8flat.mov`, the
best-tracked of the 3 real clips.

Two operational gotchas cost most of the wall-clock time on this evaluation
and are worth recording so a future attempt doesn't repeat them:

1. **Windows `MAX_PATH` (260 char) DLL-load failure.** Installing into a
   venv nested under this session's default scratchpad path (`...\AppData\
   Local\Temp\claude\...\scratchpad\venv_sports2d\...`) made `statsmodels`'
   compiled `_smoothers_lowess` extension's absolute path too long to load
   (`ImportError: DLL load failed ... The filename or extension is too
   long`). Sports2D pulls in a much heavier dependency tree than rtmlib
   alone (Pose2Sim, OpenSim, OpenVINO, PySide6, statsmodels, matplotlib --
   full install is ~1.3 GB), so it hits this ceiling where rtmlib didn't.
   Fix: install at a short path (used `C:\s2dvenv` here).
2. **Default `person_ordering_method` is `'on_click'`.** With more than one
   candidate detection (or even one, depending on internals), Sports2D opens
   a window and blocks waiting for a mouse click to pick which detected
   person to analyze -- it hangs indefinitely in any non-interactive/
   automated run with no error or timeout. Fix: pass
   `--person_ordering_method largest_size` (or `highest_likelihood`) for
   headless use. This is a real footgun for anyone scripting Sports2D
   against a batch of clips.

Command used, once both were worked around:

```
sports2d -i Smitas_8flat.mov -r out_sports2d \
  --show_realtime_results false --show_graphs false --save_graphs false \
  --person_ordering_method largest_size --backend onnxruntime --device cpu
```

Output: `out_sports2d/Smitas_8flat_Sports2D/` (skeleton-overlay .mp4,
per-frame .png images, `_px_person00.trc` / `_m_person00.trc` pose in pixels
and meters, `.c3d`, `_angles_person00.mot`, `_calib.toml`, `logs.txt`).
Total wall time for the full run (pose + filtering + angles + video/image
export), per its own log: **188.35 s** for 469 frames on CPU onnxruntime --
pose estimation itself was ~130 s of that (`469/469 [02:10<00:00]`), the
rest was Butterworth/Hampel filtering, angle computation, and re-encoding
overlay video + per-frame PNGs.

## What Sports2D computes vs. what `runform/metrics.py` computes

These two tools overlap much less than "both do running biomechanics from
video" suggests -- they sit at different layers:

| | `runform/metrics.py` (this project) | Sports2D |
|---|---|---|
| Pose source | MediaPipe PoseLandmarker (Tasks API) | RTMPose family via `rtmlib` (RTMPose-m/x, HALPE-26 keypoints for the default `body_with_feet` model) |
| Gait events (foot strike / toe-off) | Yes -- core of the module (`detect_events()`, ankle-relative-to-hip peak detection) | **No**, not in the main pipeline. Pose2Sim (Sports2D's dependency) ships a separate, disconnected CLI utility (`Pose2Sim.Utilities.trc_gaitevents`) that can derive on/off events from a saved `.trc` file by one of three methods (forward-coordinates, height-threshold, forward-velocity), but it is not wired into `sports2d`'s own pipeline or its saved outputs, and it stops at event timestamps -- no cadence, contact time, or asymmetry rollup. |
| Cadence / contact time / overstride / asymmetry | Yes, all four, with `{mean, sd, n}` and confidence gating | No |
| Joint/segment angles | Yes: hip/knee/ankle interior angle, `joint_angle()` (returns INTERIOR angle, 180 deg = straight leg, matching this project's own convention) | Yes, and considerably more of them: default joint angles are ankle/knee/hip/shoulder/elbow (both sides), default segment angles add foot/shank/thigh/pelvis/trunk/shoulders/head/arm/forearm. Sports2D's angle convention differs from this project's (its own sign/flexion convention, not necessarily "interior angle" -- would need reconciling before reusing numbers directly). |
| Inverse kinematics / kinetics (OpenSim) | No | Optional (`--do_ik`, off by default) -- can scale an OpenSim model and estimate joint torques/forces given a participant mass. Well beyond this project's current scope. |
| Multi-session / longitudinal comparison | Yes (`comparison.py`, `session.py`) | No |
| LLM narrative / training plan | Yes (`narrative.py`, `plan.py`) | No |
| Output format | Single landmarks CSV + one metrics JSON | `.trc` (pixels and meters), `.c3d`, `.mot` (angles), skeleton overlay video + per-frame PNGs, calibration `.toml` |

**Bottom line on scope overlap:** Sports2D is a pose-to-angle-and-optionally-IK
tool; this project's `metrics.py` is a pose-to-gait-event-and-spatiotemporal-
metric tool. The apparent overlap ("both do running biomechanics") is
real only at the joint-angle layer. The actual problem this investigation
is chasing -- backwards cadence vs. speed -- is entirely outside what
Sports2D computes out of the box. Adopting Sports2D would not, by itself,
give this project cadence/contact-time/overstride/asymmetry; those would
still need to be built (as they already have been, in `metrics.py`) on top
of whatever pose source is chosen, or wired up via the disconnected
`trc_gaitevents` utility and then extended significantly.

## Approach to gaps / occlusion

Sports2D's pipeline (`Sports2D/process.py`, confirmed by reading the
post-processing block and the run's own log output):
1. **Interpolation** -- gaps shorter than a threshold (`interp_gap_smaller_
   than`, default 10 frames per the CLI docs, though the observed run log
   says "Interpolating missing sequences if they are smaller than 100
   frames" -- the effective default differs from the documented CLI default,
   worth independent verification if this is ever pursued further) are
   linearly interpolated; frames past that are filled with the last valid
   value (`fill_large_gaps_with='last_value'` by default) rather than left
   NaN.
2. **Outlier rejection** -- a Hampel filter runs before smoothing
   (`reject_outliers=true` by default).
3. **Smoothing** -- Butterworth low-pass by default (4th order, 6 Hz cutoff
   in this run), with several alternate filter choices available (Kalman,
   One Euro, GCV spline, LOESS, median).

This is a meaningfully different philosophy from `runform/metrics.py`'s
`load_and_condition()`: this project masks anything below `VIS_THRESHOLD`
to NaN and only gap-fills SHORT dropouts (`limit=5` frames), deliberately
leaving long dropouts as NaN so degraded stretches get EXCLUDED from metrics
rather than papered over (CLAUDE.md: "fail loudly ... never emit metrics
silently derived from degraded tracking"). Sports2D's "fill large gaps with
last_value" default is closer to "always produce a continuous curve" than
to this project's "refuse to guess" convention -- worth flagging since it's
a real philosophical mismatch, not just an implementation detail, if this
project's pose/gap-handling logic were ever rebuilt around Sports2D's
outputs.

## Confirmed license terms

Checked the actual bundled `LICENSE` file in each installed package's
`dist-info/licenses/`, not just the PyPI classifier string:

- **Sports2D** 0.8.34 -- BSD 3-Clause License, Copyright (c) 2022,
  perfanalytics. Matches the PyPI `License-Expression: BSD-3-Clause` field.
- **Pose2Sim** 0.10.49 (Sports2D's core, non-optional dependency) -- also
  BSD 3-Clause License, Copyright (c) 2022, perfanalytics.
- **rtmlib** 0.0.15 (the pose backend both Sports2D and this project's own
  Task 2 prototype use) -- Apache License 2.0 (bundled `LICENSE` file
  confirmed directly; PyPI's own `License` field is blank, so this
  needed checking the actual file, not just metadata).

All three are permissive and compatible with this project's existing
BSD-3-Clause `Sports2D` dependency chain and zero-cost-inference constraint.
No GPL/AGPL/commercial-license concerns found anywhere in the dependency
tree that was installed (OpenSim, OpenVINO, PySide6, etc. are all
permissively licensed as well, though none of those were exercised here
since `--do_ik` stayed off).

## Does Sports2D's underlying pose backend show better left-leg tracking?

Sports2D's own saved outputs (`.trc`, `.c3d`, `.mot`) do **not** include a
per-keypoint confidence/likelihood column -- TRC is a plain X/Y/Z marker
format, and Sports2D only uses the RTMPose confidence scores internally (for
its own outlier rejection) without persisting them. So there is no way to
pull a left/right visibility number directly out of Sports2D's saved files.

Instead of guessing from architecture family alone, `logs.txt` from the run
identifies the exact backend Sports2D used for its default `body_with_feet`
/ `balanced` mode:

```
Using model HALPE_26 for body_with_feet pose estimation in balanced mode.
load yolox_m_8xb8-300e_humanart-c2c7a14a.onnx with onnxruntime backend
load rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx with onnxruntime backend
```

This is *exactly* `rtmlib.BodyWithFeet(mode='balanced')` (confirmed by
reading `rtmlib/tools/solution/body_with_feet.py`'s `MODE` dict directly --
same detector and pose-model URLs, character for character). So this
backend was run directly (bypassing Sports2D's CLI, using the same
det-then-pose call pattern as the Task 2 prototype) on `Smitas_8flat.mov`
and its raw HALPE-26 hip/knee/ankle confidence scores were measured --
giving a real number instead of an inference:

| | MediaPipe (baseline, 8 mph) | This project's rtmlib prototype (Wholebody-133, `rtmw-x`, 8 mph) | Sports2D's exact default backend (HALPE-26, `rtmpose-m`, 8 mph) |
|---|---|---|---|
| Left (far leg) worst-joint vis | 0.43 | 0.721 | 0.688 |
| Right (near leg) worst-joint vis | 0.872 | 0.749 | 0.742 |
| Near/far gap | 0.442 | 0.028 | 0.054 |

**Yes** -- Sports2D's default pose backend shows the same large improvement
in far-leg visibility and near-symmetric left/right confidence that the
larger Wholebody model did in Task 2 (see `scripts/rtmpose_ab_comparison.md`
for the full 3-clip table and the cadence-vs-speed sanity check). This is a
robust finding, not an artifact of picking one particular RTMPose model
size: both a 133-keypoint and a 26-keypoint RTMPose variant fix the same
tracking-confidence asymmetry MediaPipe shows on this footage. It does NOT,
however, mean Sports2D would fix the backwards-cadence problem -- see the
Task 2 report's verdict: the cadence trend was unchanged even with the much
better-balanced Wholebody backend, and Sports2D doesn't compute cadence at
all out of the box regardless.

## Other observations worth flagging

- Sports2D's automatic view-direction detection (`visible_side='auto'` by
  default) classified this side-on treadmill footage as **"Seen from the
  front"** (`logs.txt`: "- Person 0: Seen from the front."), not
  left/right-facing. That's a real misclassification for genuinely sagittal
  footage and would silently affect the sign/convention of any joint or
  segment angle Sports2D reports for these clips if used directly, though
  it does not affect the raw keypoint pixel positions themselves. Would
  need to be pinned manually (`--visible_side left` or `right`) for this
  project's treadmill setup rather than trusted to auto-detect.
- Sports2D's pose-only throughput (rtmpose-m/HALPE-26, `balanced`, CPU
  onnxruntime, `det_frequency` default of "every 4th frame" for the
  detector) was noticeably faster than this project's Task 2 prototype
  (~0.28 s/frame vs. ~1.0-1.06 s/frame) -- expected, since it both uses a
  smaller pose model and re-runs the (slower) person detector far less
  often, trading a little tracking robustness for speed. Not a criticism of
  either choice, just a real tradeoff to weigh if throughput ever becomes a
  constraint.

## Recommendation

Do not adopt Sports2D as a component of this project. Its pose backend
(rtmlib/RTMPose) is worth having validated -- see `scripts/
rtmpose_ab_comparison.md` for the full verdict -- but Sports2D itself is a
different-shaped tool (angle/IK-focused, no gait-event or spatiotemporal
layer) sitting on top of a much heavier dependency tree (OpenSim, OpenVINO,
PySide6, statsmodels, ~1.3 GB installed) than this project needs for what it
actually computes. If RTMPose is ever promoted into this project, doing it
directly via `rtmlib` (as prototyped in `scripts/prototype_rtmpose_pose.py`)
keeps the dependency footprint close to what CLAUDE.md already asks for
("only needs numpy/opencv-python/onnxruntime") rather than inheriting
Sports2D/Pose2Sim's much larger surface for functionality (gait events,
cadence, LLM narrative, longitudinal comparison) this project already owns
and would have to keep anyway.
