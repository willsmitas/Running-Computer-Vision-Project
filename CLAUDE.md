# Project: Running Form Analysis

Analyzes treadmill running video via pose estimation, produces biomechanics
metrics, and generates form critique + a training plan. Longitudinal:
tracks the same runner across sessions.

Full plan in `BUILD_PLAN.md`. Read it before proposing architecture changes.

## Stack

- Python 3.11
- MediaPipe Tasks API (`PoseLandmarker`) — NOT the removed `mp.solutions` API
- OpenCV, numpy, pandas, scipy
- Local LLM inference via Ollama (8B-class instruct model)
- No paid inference APIs. Cost-per-use must stay at zero.

## Code layout

- `runform/` — the package; `python -m runform <command>` is the entry point.
  - `landmarks.py` — the shared **23-point** reduced landmark scheme
    (one averaged `head` point + 22 body landmarks). Single source of
    truth for the index-order invariant below.
  - `pose.py` — video → skeleton overlay + landmarks CSV. Only module
    that imports mediapipe/cv2.
  - `metrics.py` — landmarks CSV → metrics JSON. Gait events, joint
    angles, cadence, contact time, overstride, asymmetry.
  - `pipeline.py` — Phase 1: video → all artifacts + quality flags.
  - `references.py`, `gating.py`, `speed_profile.py`, `root_causes.py`,
    `session.py` — Phase 2 deterministic interpretation.
  - `narrative.py` — Phase 3 LLM layer (Ollama, schema + citation checks).
  - `plan.py` — Phase 4 drill library and plan builder.
  - `comparison.py` — Phase 5 longitudinal deltas + significance gate.
  - `models.py`, `storage.py` — data model + JSON persistence.
- `pose_skeleton_starter.py`, `running_metrics.py` — thin CLI shims for
  the original workflows; all logic lives in the package.
- `tests/` — stdlib unittest against synthetic ground truth:
  `python -m unittest discover -s tests -t .`

## Non-obvious invariants — do not break these

- **Aspect correction is mandatory.** MediaPipe normalizes x by frame width
  and y by frame height independently. All x values must be scaled by
  (width/height) before any geometry. Skipping this changed overstride by
  74% in testing.
- **Gait events use ankle position RELATIVE TO HIP**, never absolute. The
  relative signal cancels camera pan and forward travel.
- **Ignore MediaPipe's z coordinate.** Single-camera depth is unreliable.
  Sagittal-plane 2D only.
- **All distance metrics normalize by leg length** so results are comparable
  across filming distances.
- **`joint_angle` returns interior angle** (straight leg = 180°). Sports
  science literature reports flexion = 180 − interior. Convert before
  comparing to published ranges.
- Landmark index order is fixed and shared between both scripts. Changing it
  in one silently corrupts the other.

## Architectural rule: deterministic first, LLM last

The LLM writes explanation and prescription. It does NOT:
- decide what is normal
- do arithmetic on metrics
- detect faults
- invent metrics

Reference ranges, confidence gating, and root-cause collapse are plain
Python. This is what keeps a small open-weight model viable. When in doubt,
push logic into code rather than into the prompt.

## Conventions

- Metrics report `{mean, sd, n}` — never a bare mean. SD and n determine
  whether the mean is usable.
- Fail loudly on bad input; never emit metrics silently derived from
  degraded tracking.
- Thresholds go in named module-level constants with a comment explaining
  the reasoning, not inline magic numbers.
- Every threshold currently in the codebase is an educated guess, untuned
  against real footage. Treat as provisional.

## Testing

- Synthetic-data tests: generate landmark CSVs with known ground truth
  (known cadence, known planted fault) and assert recovery. Validates
  signal-processing logic without needing video.
- Synthetic tests do NOT validate real-world behavior. Known open risk:
  MediaPipe may swap left/right leg assignment at limb crossover, which
  would corrupt all per-side and asymmetry metrics. Unverified.
- LLM layer: maintain a regression set of metric payloads with expected
  findings. Run against every prompt or model change.

## Safety constraints — non-negotiable

- No injury diagnosis. Ever.
- Injury risk framed as association from literature, never prediction.
- Any user mention of pain routes to a licensed professional.
- Disclaimer present in every assessment output.

## Working style

- Prefer editing existing files over creating new ones.
- Do not add dependencies without asking.
- Ask before refactoring across multiple files.
- When a task is ambiguous, ask rather than guessing at scope.
