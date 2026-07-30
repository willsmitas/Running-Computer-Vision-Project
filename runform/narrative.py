"""Phase 3: the LLM narrative layer, via local Ollama (zero cost per use).

The LLM writes explanation and prescription narrative ONLY. It never
decides what is normal, never does arithmetic, never detects faults,
never invents metrics — all of that arrived precomputed in the
assessment payload from session.interpret_session().

Small open-weight models fail first at structured output, so schema
validation + retry is built in from the start, and every hard constraint
is ALSO enforced in code after the fact:
  - citations restricted to supplied metrics (post-validated)
  - issue cap (post-validated)
  - no diagnosis language (post-validated, crude but deterministic)
  - disclaimer appended by code, never trusted to the model
  - pain routing decided by code from the session payload
"""

import json
import urllib.error
import urllib.request

from .errors import NarrativeError

# Bump on ANY prompt change and store with every Assessment, so you can
# tell whether output changed because the runner changed or because the
# prompt did (BUILD_PLAN data model).
PROMPT_VERSION = "0.2.0"

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://localhost:11434"

# Max issues in one assessment. More than 2-3 is unactionable for the
# runner and invites the model to pad.
MAX_ISSUES = 3

# 1 first attempt + 2 retries with the validation errors fed back.
MAX_ATTEMPTS = 3

REQUEST_TIMEOUT_S = 180

# Appended by code to 100% of assessments (safety constraint, CLAUDE.md).
DISCLAIMER = (
    "Automated analysis from 2D video pose estimation — not medical "
    "advice, and not an injury diagnosis. Single-camera measurements "
    "carry systematic error; trends across sessions are more reliable "
    "than any absolute number."
)

# Prepended by code whenever the session notes mention pain.
PAIN_ROUTING = (
    "You mentioned pain in your session notes. This tool cannot and does "
    "not assess pain or injury — please see a licensed sports-medicine "
    "professional or physical therapist before acting on anything below."
)

_SCHEMA_DESCRIPTION = """{
  "summary": "2-3 sentence plain-language overview of this session",
  "issues": [
    {
      "title": "short issue name",
      "root_cause": "one of the provided root-cause ids",
      "explanation": "what the numbers show and why it matters, citing metrics by name",
      "metrics_cited": ["metric names from allowed_metrics that this issue rests on"]
    }
  ],
  "next_focus": "one sentence on the single most useful thing to work on"
}"""


def build_prompt(assessment):
    """Assessment payload -> full prompt. The model gets ranked findings,
    not raw data, so its remaining job is narrative writing — which is
    what keeps an 8B-class model viable."""
    payload = {
        "runner_context": {
            "notes": assessment["session"]["notes"],
            "speeds": assessment["session"]["speeds"],
        },
        "ranked_root_causes": assessment["root_causes"],
        "deviations": assessment["deviations"],
        "speed_classifications": assessment["speed_profile"]["classifications"],
        "excluded_metrics": assessment["excluded_metrics"],
        "allowed_metrics": assessment["allowed_metrics"],
    }
    return f"""You are a running-form coach writing a short, encouraging assessment
from a treadmill gait analysis. All detection and judgment has already
been done; the findings below are final. Your job is ONLY to explain
them clearly.

Hard rules — violating any of them invalidates your response:
1. Use only the findings provided. Never invent, estimate, or recompute
   a number, and never mention a metric that is not in allowed_metrics.
2. Cover at most {MAX_ISSUES} issues, in the given root-cause order. Do not
   re-rank, merge, or add issues. If ranked_root_causes is EMPTY, that
   means no fault was found — this is a good outcome, not missing data.
   You MUST return "issues": [] in that case. Do not invent a root cause,
   and do not cite a metric name as if it were one. Say in the summary
   that the session looked clean.
3. Every issue must cite the metric names it rests on in metrics_cited.
4. Never diagnose an injury or name a medical condition. If pain is
   mentioned in the runner context, say only that a licensed
   professional should evaluate it.
5. If you mention injury risk at all, frame it strictly as an
   association reported in research literature, never a prediction.
6. If a finding's confidence is "low", say so plainly and hedge.
7. Respond with valid JSON only, no prose outside JSON, matching:
{_SCHEMA_DESCRIPTION}

Findings:
{json.dumps(payload, indent=1)}
"""


def ollama_generate(prompt, model=DEFAULT_MODEL, host=DEFAULT_HOST):
    """One completion from a local Ollama server. No paid APIs, ever."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read())["response"]
    except (urllib.error.URLError, OSError) as e:
        raise NarrativeError(
            f"Could not reach Ollama at {host} ({e}). Is it running? "
            f"Start it with `ollama serve` and pull the model with "
            f"`ollama pull {model}`."
        ) from e


def validate_output(parsed, assessment):
    """Deterministic post-validation. Returns a list of problems (empty =
    valid). This, not the prompt, is the actual enforcement mechanism."""
    problems = []
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]

    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("missing or empty 'summary'")

    allowed = set(assessment["allowed_metrics"])
    cause_ids = {c["cause"] for c in assessment["root_causes"]}

    issues = parsed.get("issues")
    if not isinstance(issues, list):
        problems.append("'issues' must be a list")
        issues = []
    if len(issues) > MAX_ISSUES:
        problems.append(f"{len(issues)} issues given; max is {MAX_ISSUES}")
    # Symmetric to the empty-cause_ids check below: an empty issues list
    # is only valid when there is nothing to report. If root causes WERE
    # identified, "issues: []" is silent under-reporting, not a safe
    # default -- without this, a model that fails once can retry into
    # dropping every real finding rather than fixing the actual problem,
    # and that emptied response has always passed validation unchecked.
    if cause_ids and not issues:
        problems.append(
            f"root causes {sorted(cause_ids)} were identified but no "
            f"issues were reported"
        )
    for i, issue in enumerate(issues):
        if not isinstance(issue, dict):
            problems.append(f"issue {i} is not an object")
            continue
        if not str(issue.get("title", "")).strip():
            problems.append(f"issue {i} missing 'title'")
        if not str(issue.get("explanation", "")).strip():
            problems.append(f"issue {i} missing 'explanation'")
        cited = issue.get("metrics_cited")
        if not isinstance(cited, list) or not cited:
            problems.append(f"issue {i} cites no metrics")
        else:
            invented = [m for m in cited if m not in allowed]
            if invented:
                problems.append(
                    f"issue {i} cites metrics that were not supplied: {invented}"
                )
        # No "cause_ids and" guard: when cause_ids is empty (a clean
        # session, no root causes identified), nothing is a member of it,
        # so ANY issue correctly fails here. That's intentional -- a clean
        # session has nothing to report, and the model fabricating a
        # critique anyway must be rejected, not waved through because
        # there happened to be zero real causes to check against.
        if issue.get("root_cause") not in cause_ids:
            problems.append(
                f"issue {i} root_cause '{issue.get('root_cause')}' is not one "
                f"of the provided causes {sorted(cause_ids)}"
            )

    # Crude but deterministic diagnosis guard; the prompt does the nuance,
    # this catches the bright line.
    for text in _all_strings(parsed):
        if "diagnos" in text.lower():
            problems.append("output contains diagnosis language")
            break

    return problems


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)


def render(parsed, assessment):
    """Validated structured output -> final text. Safety text is added
    here, in code — never left to the model."""
    lines = []
    if assessment["session"]["pain_flag"]:
        lines += [PAIN_ROUTING, ""]
    lines.append(parsed["summary"].strip())
    for issue in parsed.get("issues", []):
        cited = ", ".join(issue.get("metrics_cited", []))
        lines += ["", f"• {issue['title']}: {issue['explanation'].strip()} "
                      f"(metrics: {cited})"]
    next_focus = str(parsed.get("next_focus", "")).strip()
    if next_focus:
        lines += ["", f"Focus next: {next_focus}"]
    lines += ["", "—", DISCLAIMER]
    return "\n".join(lines)


def narrate(assessment, model=DEFAULT_MODEL, host=DEFAULT_HOST,
            generate=None, max_attempts=MAX_ATTEMPTS):
    """Assessment payload -> validated narrative.

    `generate` (prompt -> raw text) is injectable so tests and the
    regression set can run without a live Ollama server.
    """
    gen = generate or (lambda p: ollama_generate(p, model=model, host=host))
    base_prompt = build_prompt(assessment)

    prompt, problems = base_prompt, []
    for attempt in range(1, max_attempts + 1):
        raw = gen(prompt)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            problems = ["response was not valid JSON"]
        else:
            problems = validate_output(parsed, assessment)
        if not problems:
            return {
                "narrative": render(parsed, assessment),
                "structured": parsed,
                "model_name": model,
                "prompt_version": PROMPT_VERSION,
                "attempts": attempt,
            }
        prompt = (
            base_prompt
            + "\nYour previous response was rejected for these reasons:\n- "
            + "\n- ".join(problems)
            + "\nRespond again, fixing every problem. JSON only."
        )

    raise NarrativeError(
        f"LLM output failed validation after {max_attempts} attempts. "
        f"Last problems: {problems}"
    )
