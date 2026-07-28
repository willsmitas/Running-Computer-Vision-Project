"""Shared exceptions.

Convention (CLAUDE.md): fail loudly on bad input; never emit metrics
silently derived from degraded tracking. Raising one of these with an
informative message IS the failure mode — callers should not catch and
paper over them.
"""


class RunFormError(Exception):
    """Base class for all pipeline errors."""


class VideoError(RunFormError):
    """Video missing, unreadable, or empty."""


class PoseQualityError(RunFormError):
    """Pose was extracted but tracking is too poor to support metrics."""


class MetricsError(RunFormError):
    """Landmark data cannot support metric computation."""


class NarrativeError(RunFormError):
    """LLM output failed validation after retries, or Ollama unreachable."""
