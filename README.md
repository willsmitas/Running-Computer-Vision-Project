# Running Form Analysis

Treadmill running video → pose estimation → biomechanics metrics →
deterministic fault interpretation → local-LLM narrative → training plan,
tracked for the same runner across sessions.

Read `BUILD_PLAN.md` for the full design and phase gates, and `CLAUDE.md`
for invariants and conventions before changing anything.

## Setup

```
.venv\Scripts\activate          # existing venv (mediapipe, opencv, numpy, pandas, scipy)
pip install -r requirements.txt # only needed on a fresh environment
```

The narrative layer needs [Ollama](https://ollama.com) running locally:
`ollama pull llama3.1:8b`. Everything upstream of it runs without.

## Usage

One command per build phase, all via the package CLI:

```
python -m runform analyze clip.mp4                  # video -> skeleton video, landmarks CSV, metrics + quality report
python -m runform interpret --clip easy_metrics.json easy 2.6 ^
                            --clip mod_metrics.json moderate 3.3 ^
                            --clip fast_metrics.json fast 4.1 ^
                            --notes "..." --out assessment.json
python -m runform narrate assessment.json           # needs Ollama
python -m runform plan assessment.json
python -m runform compare before.json after.json
```

`pose_skeleton_starter.py` and `running_metrics.py` remain as thin shims
for the original single-script workflows.

## Tests

Synthetic ground-truth suite (no video, no Ollama needed):

```
python -m unittest discover -s tests -t .
```

## Status

- Phases 1–5 are scaffolded and unit-tested against synthetic data.
- **Phase 0 (validation on real treadmill clips) has been run — partial
  pass, blocked on footage.** Full results in `scripts/phase0_validation.md`.
  Manual ground-truth strike counts match `steps_detected` exactly on 2 of
  3 clips (the third, the fastest clip, misses by 2 in a degraded final
  3 s) and cadence is within 3 spm on all 3; no left/right leg swap was
  found at limb crossover.
  - The clip filenames are paces per mile (`5flat` = 5:00/mile ≈ 12 mph,
    fastest; `8flat` = 8:00/mile = 7.5 mph, slowest), and belt-mark
    timing confirms the labels to within a few percent. Convert pace to
    actual speed before feeding `interpret` — passing "5/6/8" as speed
    values would reverse every speed-profile slope. With true speeds,
    cadence rises with speed normally; the earlier "cadence falls with
    speed" anomaly was a misreading of the labels as mph.
  - Hard blocker: the far (left) leg tracks below the visibility
    threshold in every clip, so left-side and asymmetry metrics are
    unvalidated and untrustworthy on this footage. A re-shoot with light
    on the far side is required before Phase 0 can pass.
- Phase 6 (UI) is deliberately last; everything is exercisable via CLI.
