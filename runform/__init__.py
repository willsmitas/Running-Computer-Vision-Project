"""Running form analysis: pose estimation -> metrics -> deterministic
interpretation -> LLM narrative -> training plan, tracked longitudinally.

Layering rule (see BUILD_PLAN.md): everything that decides what is normal,
detects faults, or does arithmetic on metrics is plain Python. The LLM
layer (`narrative`) only turns precomputed findings into prose.

Module map, by build phase:

    landmarks       shared 23-point landmark scheme (single source of truth)
    pose            Phase 0/1  video -> skeleton overlay + landmarks CSV
    metrics         Phase 0/1  landmarks CSV -> metrics JSON
    pipeline        Phase 1    one command: video -> all artifacts + quality
    references      Phase 2    reference ranges conditioned on pace
    gating          Phase 2    confidence grading (high / low / unusable)
    speed_profile   Phase 2    metric-vs-speed fit + classification
    root_causes     Phase 2    collapse correlated symptoms, rank causes
    session         Phase 2    orchestrates the deterministic interpretation
    narrative       Phase 3    Ollama narrative with schema/citation checks
    plan            Phase 4    drill library -> bounded, falsifiable plan
    comparison      Phase 5    session-over-session deltas + significance
    models          data model (BUILD_PLAN shape)
    storage         JSON persistence for longitudinal use
    cli             `python -m runform <command>`
"""

__version__ = "0.1.0"
