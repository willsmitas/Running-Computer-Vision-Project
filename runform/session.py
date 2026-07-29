"""Phase 2 orchestrator: clips + speeds + context -> deterministic
interpretation payload.

The output of interpret_session() is the analytical core of the product:
deviations from pace-conditioned reference ranges, confidence grades,
speed-profile classifications, and ranked root causes. It must be
something a coach could read and trust BEFORE any narrative is written on
top of it — the LLM (Phase 3) receives this payload and adds prose only.
"""

from .errors import RunFormError
from . import gating, references, root_causes, speed_profile

# Any of these in the session notes routes the runner to a licensed
# professional (safety constraint, CLAUDE.md). Deliberately over-broad:
# normal training soreness will trip it, and routing a sore-but-fine
# runner to a professional is the failure mode we prefer.
PAIN_KEYWORDS = (
    "pain", "painful", "hurt", "hurts", "hurting",
    "ache", "aches", "aching", "sore", "soreness",
    "injur", "sprain", "strain", "tendon",
)


def pain_mentioned(notes):
    text = (notes or "").lower()
    return any(kw in text for kw in PAIN_KEYWORDS)


def _validate_clips(clips):
    if not clips or len(clips) > len(references.SPEED_BANDS):
        raise RunFormError(
            f"Expected 1-{len(references.SPEED_BANDS)} clips, got {len(clips) if clips else 0}."
        )
    labels = [c.get("speed_label") for c in clips]
    for label in labels:
        if label not in references.SPEED_BANDS:
            raise RunFormError(
                f"Unknown speed_label '{label}'. Expected one of {references.SPEED_BANDS}."
            )
    if len(set(labels)) != len(labels):
        raise RunFormError(f"Duplicate speed labels in session: {labels}")
    for c in clips:
        if not isinstance(c.get("metrics"), dict):
            raise RunFormError(f"Clip '{c.get('speed_label')}' has no metrics dict.")


def interpret_session(clips, notes=""):
    """clips: list of dicts, one per pace band:
        {
          "speed_label": "easy" | "moderate" | "fast",
          "speed_value": float,          # treadmill speed, user-entered
          "metrics": {...},              # metrics.compute_metrics() output
          "detection_rate": float|None,  # from the Phase 1 pipeline
          "quality_flags": [...],        # from the Phase 1 pipeline
        }

    Returns the deterministic assessment payload (plain dict, JSON-safe).
    """
    _validate_clips(clips)
    clips = sorted(clips, key=lambda c: references.SPEED_BANDS.index(c["speed_label"]))
    present_bands = [c["speed_label"] for c in clips]

    all_devs, excluded, clip_summaries, metrics_by_clip = [], [], [], {}
    points = {}  # metric -> [(speed_value, side-averaged mean)] for fits

    for clip in clips:
        band = clip["speed_label"]
        metrics = clip["metrics"]
        detection_rate = clip.get("detection_rate")
        metrics_by_clip[band] = metrics

        grades = gating.grade_clip_metrics(metrics, detection_rate)

        for dev in references.clip_deviations(metrics, band):
            key = gating.grade_key(dev["metric"], dev["side"], dev["submetric"])
            dev["grade"] = grades.get(key, "low")
            if dev["grade"] == "unusable":
                # Honest exclusion beats silent use of junk evidence.
                excluded.append({**dev, "reason": "graded unusable"})
            else:
                all_devs.append(dev)

        # Metrics graded unusable are also excluded from "looks clean"
        # claims — record them so the payload never implies they passed.
        for key, grade in grades.items():
            if grade == "unusable":
                excluded.append({
                    "metric": key, "speed_label": band,
                    "reason": "graded unusable (insufficient n, spread, or tracking)",
                })

        speed_value = clip.get("speed_value")
        if speed_value is not None:
            values = {}
            for metric, _side, value, _summary in references.iter_metric_values(metrics):
                values.setdefault(metric, []).append(value)
            for metric, vals in values.items():
                # Sides averaged: the fit tracks the metric's overall
                # response to pace, not per-side detail.
                points.setdefault(metric, []).append(
                    (float(speed_value), sum(vals) / len(vals))
                )

        meta = metrics.get("_meta", {})
        clip_summaries.append({
            "speed_label": band,
            "speed_value": speed_value,
            "cadence_spm": metrics.get("cadence_spm"),
            "steps_detected": metrics.get("steps_detected"),
            "leg_length_units": meta.get("leg_length_units"),
            "duration_s": meta.get("duration_s"),
            "direction": meta.get("direction"),
            "detection_rate": detection_rate,
            "quality_flags": clip.get("quality_flags", []),
        })

    fits = {
        metric: fitted
        for metric, pts in points.items()
        if (fitted := speed_profile.fit(pts)) is not None
    }

    flagged_bands = {}
    for dev in all_devs:
        flagged_bands.setdefault(dev["metric"], set()).add(dev["speed_label"])
    classifications = {
        metric: speed_profile.classify(bands, present_bands)
        for metric, bands in flagged_bands.items()
    }

    causes = root_causes.collapse(all_devs, classifications)

    allowed = sorted(
        {m for m, _s, _v, _sum in references.iter_metric_values(clips[0]["metrics"])}
        | {"asymmetry_pct"}
    )

    return {
        "session": {
            "notes": notes or "",
            "pain_flag": pain_mentioned(notes),
            "speeds": {c["speed_label"]: c.get("speed_value") for c in clips},
            "clips": clip_summaries,
        },
        "metrics_by_clip": metrics_by_clip,
        "deviations": all_devs,
        "excluded_metrics": excluded,
        "speed_profile": {"fits": fits, "classifications": classifications},
        "root_causes": causes,
        "allowed_metrics": allowed,
    }
