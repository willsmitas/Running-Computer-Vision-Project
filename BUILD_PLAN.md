# Running Form Analysis App — Build Plan

## Product thesis

Most consumer running-form tools give a one-off verdict from a single clip.
The defensible product here is **longitudinal**: same runner, same setup,
tracked over weeks, with a prescription that adapts to measured change.

Single-camera pose estimation carries meaningful systematic error. That error
largely **cancels** when comparing the same runner filmed the same way across
sessions. So change detection is far more trustworthy than absolute
assessment — and it is also the more compelling product.

Design for the repeat-visit case from day one, even if v1 only handles one
session.

---

## Two protocol decisions that carry a lot of weight

### Treadmill, side view

This is not just a convenience. It removes most of the error sources that
would otherwise wreck longitudinal comparison:

- Runner stays at a fixed position in frame → no camera panning, so the
  hip-relative signal in `detect_events` stays clean
- Constant distance from camera → consistent scale between sessions
- Controlled, known speed → the single most important piece of context
- Repeatable lighting and background → stable landmark confidence

Treadmill biomechanics differ slightly from overground. Irrelevant here,
because the comparison is always treadmill-to-treadmill.

**Consequence:** setup consistency is a first-class feature, not an
instruction buried in help text. Store camera height, distance, and phone
orientation from session 1 and replay them to the user at every subsequent
session. Drift in setup will masquerade as drift in form.

### Three speeds, not one

The three-speed protocol is the strongest analytical idea in this design and
should be treated as core, not as "more data."

A single clip gives a point. Three clips give a **slope** — how each metric
responds to increasing pace. That distinguishes fault classes that look
identical at one speed:

- **Constant across speeds** → ingrained motor pattern; responds to cueing
  and drills
- **Degrades as speed rises** → strength, mobility, or coordination ceiling;
  responds to strength work, not cueing
- **Only present at slow speed** → often benign, or a low-cadence artifact

Prescriptions should differ sharply between those cases. A cueing plan for
what is actually a strength limitation will fail, and the runner will
conclude the app does not work.

Suggested speeds: easy, moderate, near-threshold. Actual values matter less
than the user repeating **the same three** every session. Store them and
pre-fill on return visits.

---

## Architectural principle: deterministic first, LLM last

The LLM does **explanation and prescription only**. It never does detection,
never decides what is normal, never does arithmetic on metrics.

Everything upstream — reference comparison, confidence gating, root-cause
collapse — is plain code: testable, debuggable, free to run, and identical
every time.

This is not only a correctness argument. It is what makes a **small
open-weight model viable**. If the model must judge whether 176° knee angle
is abnormal, you need a large, expensive model and you will still get
hallucinated norms. If the model receives "knee angle 176°, expected 155–170
at this pace, deviation +6°, root cause: overstride, confidence: high" then
the remaining task is narrative writing — well within a 7–8B model.

**Every hour spent moving logic out of the prompt and into code reduces the
model size you need.**

---

## Data model

Design for longitudinal use from the start. Rough shape:

```
User
  id, height_cm, experience_level, created_at
  baseline_setup { camera_distance_m, camera_height_cm, orientation }

Session
  id, user_id, recorded_at, session_number
  setup_confirmed (bool)
  notes (injury status, fatigue, shoes, surface)

Clip
  id, session_id, speed_label (easy|moderate|fast)
  speed_value, speed_unit
  video_path, skeleton_video_path, landmarks_csv_path
  fps, width, height
  detection_rate, quality_flags[]

Metrics
  clip_id
  cadence_spm, per_side{...}, vertical_oscillation_ratio, asymmetry_pct
  n_strides, confidence_grade

SpeedProfile          # derived across the 3 clips in a session
  session_id
  metric_name -> { slope, intercept, r2, classification }

Assessment
  session_id
  root_causes[] (ranked, with supporting metrics)
  narrative (LLM output)
  model_name, model_version, prompt_version

Plan
  id, session_id, created_at, duration_weeks
  items[] { drill, cue, sets, frequency, targets_root_cause, progression_rule }
  recheck_due_date
  success_criteria[] { metric, direction, target_delta }

Comparison            # session N vs session N-1
  from_session_id, to_session_id
  metric_deltas[], significance_flags[]
  plan_adherence (self-reported)
```

Two fields that are easy to omit and painful to add later:
`prompt_version` on Assessment (so you can tell whether output changed
because the runner changed or because you edited the prompt), and
`success_criteria` on Plan (so progress is falsifiable rather than vibes).

---

## Build phases

Each phase ends with something runnable. Do not start a phase before its
predecessor's acceptance criteria pass.

### Phase 0 — Validate what already exists

**Nothing downstream is worth building until this passes.** The existing
scripts have only been tested against synthetic data.

- [ ] Run `pose_skeleton_starter.py` on 3 real treadmill clips
- [ ] Watch each skeleton overlay in full; check limb-crossover frames
      specifically for left/right leg swapping
- [ ] Run `running_metrics.py`; manually count foot strikes in the video and
      compare to `steps_detected`
- [ ] Verify `direction` is correct
- [ ] Sanity-check cadence against a manual count (count strikes over 30s)
- [ ] Tune `MIN_SWING_RATIO`, `prominence`, and `--smooth` against real data;
      current values are guesses
- [ ] Apply the trunk-lean sign fix

**Acceptance:** detected strike count within ±1 of manual count on all 3
clips; cadence within 3 spm of manual count.

**If left/right leg assignment swaps at crossover, stop and fix that first.**
It silently corrupts every per-side metric and every asymmetry number, and no
amount of good downstream design survives it.

### Phase 1 — Pipeline orchestration

One command: video in → skeleton video, landmarks CSV, metrics JSON out.

- [ ] Wrap both scripts behind a single entry point
- [ ] Auto-extract fps/width/height from the video (stop requiring flags)
- [ ] Structured error handling: bad video, no pose detected, too short
- [ ] Emit quality flags (detection rate, mean visibility on key joints)

**Acceptance:** single command on a raw clip produces all three artifacts
plus a quality report, and fails informatively on a deliberately bad clip.

### Phase 2 — Session layer and interpretation (all deterministic)

The analytical core. No LLM yet.

- [ ] Session model: 3 clips + speeds + context
- [ ] Reference ranges per metric, **conditioned on pace**
- [ ] Confidence gating: n_strides, SD, detection rate, visibility
      → grade each metric (high / low / unusable)
- [ ] Speed-profile derivation: fit each metric against speed, classify as
      constant / degrades-with-speed / slow-only
- [ ] Root-cause collapse: map correlated symptoms to underlying causes,
      rank by breadth of effect
- [ ] Output: ranked root causes, each with supporting metrics and
      confidence

**Acceptance:** feed it a clip set from a runner deliberately overstriding;
overstride surfaces as top root cause with correct supporting metrics.
Feed it a clean clip set; no high-confidence faults fire.

This phase is fully unit-testable with synthetic metric dicts — no video
needed. Write those tests; they are cheap and they protect the layer that
everything else depends on.

### Phase 3 — LLM narrative layer

- [ ] Local inference via Ollama (see model strategy below)
- [ ] Prompt takes: metrics, reference deltas, confidence grades, speed
      classifications, ranked root causes, runner context
- [ ] Structured JSON output, schema-validated, with retry on invalid
- [ ] Hard constraints in prompt:
      - every claim cites the metric that triggered it
      - max 2–3 issues
      - no diagnosis; any mention of pain routes to a professional
      - no invented metrics
- [ ] Post-validation: reject output referencing metrics not supplied
- [ ] Injury-risk framing as **association, not prediction** ("patterns like
      this are associated with X in the literature"), never "you will get X"

**Acceptance:** 20 runs on the same input produce consistent findings; zero
outputs cite a metric that was not provided; disclaimer present in 100%.

### Phase 4 — Prescription and plan

- [ ] Drill library, each entry tagged to root causes it addresses
- [ ] Drills map to **root cause**, not symptom
- [ ] Prefer internal cues ("run at 180 steps per minute") over descriptive
      ones ("stop overstriding") — better motor-learning outcomes
- [ ] Plan is time- and volume-bounded: weeks, frequency, duration
- [ ] Ordered, with prerequisites respected
- [ ] Explicit success criteria: which metric, which direction, how much,
      by when
- [ ] Schedule re-film prompt at plan end

**Acceptance:** every plan item traces to a root cause; every plan has at
least one falsifiable success criterion.

### Phase 5 — Longitudinal comparison

The actual product.

- [ ] Session-over-session metric deltas
- [ ] **Significance gate:** distinguish real change from measurement noise.
      Use the within-session SD across strides as the noise floor — a delta
      smaller than that is not a result. Do not report it as one.
- [ ] Setup-drift check: flag if camera distance or scale changed materially
- [ ] Progression rules per drill: advance / hold / regress based on whether
      target metrics moved
- [ ] Progress visualization across sessions
- [ ] Handle the awkward cases honestly: no change, regression, partial
      adherence

**Acceptance:** synthetic session pairs with known deltas classify correctly
as significant vs. noise.

The temptation here is to show progress that is not there. Resist it. A tool
that says "no measurable change yet, keep going" builds more trust than one
that manufactures a 2 spm improvement.

### Phase 6 — UI

Deliberately last. Everything above is testable via CLI.

- [ ] Guided capture with setup replay from baseline
- [ ] Speed entry, pre-filled from previous session
- [ ] Processing progress (pose estimation is slow; show it)
- [ ] Results: skeleton video, metrics, narrative, plan
- [ ] Progress view
- [ ] Re-film reminder

---

## Open-weight model strategy

**"No cost per use" is achievable, but be clear about where the cost moves.**

Three deployment options, honestly:

1. **Local inference (Ollama)** — genuinely free, fine for you and teammates.
   The right choice for Phases 3–5. Not a distribution strategy.
2. **Self-hosted GPU** — fixed monthly cost regardless of usage. Predictable,
   not free. Sane once you have real users.
3. **On-device mobile** — genuinely free at any scale, but constrains you to
   ~3B-class models and adds real engineering. Viable *only* because your
   deterministic layer does the heavy lifting.

Free tiers on hosted inference providers exist but will rate-limit; fine for
development, not for a launch.

**Model sizing:** start with an 8B-class instruct model. Test whether it holds
the constraints — cites metrics, respects the issue cap, keeps the disclaimer,
emits valid JSON. If it drifts, first try tightening the prompt and
constraining output schema harder; only then reach for a larger model.

**Build a regression set early.** Ten to fifteen representative metric
payloads with expected findings. Run them against any model or prompt change.
This is what lets you swap models without re-reading every output by hand,
and it is the single highest-leverage testing investment in the project.

**Structured output is where small models fail first.** Budget for retry
logic and schema validation from the start rather than discovering it later.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Left/right leg swap at limb crossover | Corrupts all per-side and asymmetry output | Phase 0 gate; consider tracking continuity check |
| Reference ranges wrong or not pace-conditioned | Confident, wrong critique | Validate against coach assessment |
| Setup drift between sessions read as form change | False progress signals | Store and replay baseline setup; flag scale changes |
| Small model hallucinates norms | Plausible nonsense | Keep judgment in code; post-validate citations |
| Asymmetry artifact from single-side view | Fake injury-risk flags | Gate asymmetry hardest; consider both-sides filming |
| Measurement noise reported as progress | Erodes trust | Significance gate against within-session SD |

---

## Validation strategy

You have an advantage most people building this do not: access to actual
D1 coaches and teammates.

- **Blind coach comparison** — coach assesses the same clips independently;
  compare to system output. Disagreements localize to one of three layers:
  metric extraction, reference ranges, or reasoning. Different fixes.
- **Planted faults** — have a teammate deliberately overstride, or run with
  excessive vertical oscillation. Check the system finds what you planted.
  Cheap, and catches whole classes of failure.
- **Test-retest** — film the same runner twice in one session. Any metric
  that differs materially between two back-to-back clips cannot support
  week-over-week claims.

That last one is worth doing early. It empirically establishes your noise
floor, which Phase 5 needs anyway.

---

## Out of scope for v1

- Injury diagnosis of any kind
- Overground / outdoor footage (revisit after treadmill works)
- Multi-angle capture
- Real-time analysis
- Multi-user / social features
- Absolute-units output where ratios suffice

---

## Suggested first three sessions of work

1. Phase 0 validation on real clips + trunk-lean fix
2. Phase 1 pipeline wrapper
3. Phase 2 reference layer + confidence gating (deterministic, unit-tested)

Do not touch the LLM layer until Phase 2 output is something you would
trust a coach to read.
