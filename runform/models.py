"""Data model, shaped for longitudinal use from day one (BUILD_PLAN).

These dataclasses are the canonical record shapes for storage and the
future UI. The analysis layers themselves pass plain dicts (JSON-safe);
convert with to_dict() at the storage boundary.

Two fields that are easy to omit and painful to add later, kept on
purpose: Assessment.prompt_version (so you can tell whether output
changed because the runner changed or because the prompt did) and
TrainingPlan.success_criteria (so progress is falsifiable, not vibes).
"""

from dataclasses import asdict, dataclass, field


@dataclass
class BaselineSetup:
    """Captured at session 1 and REPLAYED to the user at every subsequent
    session — setup drift masquerades as form drift, so consistency is a
    first-class feature, not a help-text footnote."""
    camera_distance_m: float
    camera_height_cm: float
    orientation: str  # "landscape" | "portrait"


@dataclass
class UserProfile:
    id: str
    height_cm: float | None = None
    experience_level: str | None = None  # "beginner" | "recreational" | "competitive"
    created_at: str = ""
    baseline_setup: BaselineSetup | None = None
    # The three treadmill speeds, stored so return visits pre-fill the
    # same ones; repeating the exact speeds is what makes the speed
    # profile comparable across sessions.
    speeds: dict | None = None  # {"easy": 2.8, "moderate": 3.4, "fast": 4.2}
    speed_unit: str = "m/s"


@dataclass
class Clip:
    id: str
    session_id: str
    speed_label: str  # "easy" | "moderate" | "fast"
    speed_value: float | None = None
    speed_unit: str = "m/s"
    video_path: str = ""
    skeleton_video_path: str = ""
    landmarks_csv_path: str = ""
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    detection_rate: float | None = None
    quality_flags: list = field(default_factory=list)


@dataclass
class Session:
    id: str
    user_id: str
    recorded_at: str
    session_number: int
    setup_confirmed: bool = False
    notes: str = ""  # injury status, fatigue, shoes, surface
    clips: list = field(default_factory=list)  # [Clip]


@dataclass
class SpeedProfileEntry:
    metric: str
    slope: float
    intercept: float
    r2: float
    classification: str  # speed_profile.classify() output


@dataclass
class Assessment:
    session_id: str
    root_causes: list = field(default_factory=list)  # ranked, with metrics
    narrative: str = ""
    model_name: str = ""
    model_version: str = ""
    prompt_version: str = ""


@dataclass
class SuccessCriterion:
    root_cause: str
    metric: str
    direction: str  # "increase" | "decrease"
    target_delta: float
    baseline: float | None = None
    speed_label: str | None = None
    by_week: int | None = None


@dataclass
class PlanItem:
    drill: str
    cue: str
    dosage: str
    frequency_per_week: int
    targets_root_cause: str
    progression_rule: dict = field(default_factory=dict)


@dataclass
class TrainingPlan:
    id: str
    session_id: str
    created_at: str
    duration_weeks: int
    items: list = field(default_factory=list)  # [PlanItem]
    success_criteria: list = field(default_factory=list)  # [SuccessCriterion]
    recheck_due_date: str = ""


@dataclass
class Comparison:
    from_session_id: str
    to_session_id: str
    metric_deltas: list = field(default_factory=list)
    setup_drift: list = field(default_factory=list)
    plan_adherence: str | None = None  # self-reported


def to_dict(obj):
    return asdict(obj)
