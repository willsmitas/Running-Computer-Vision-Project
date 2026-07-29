"""Command-line entry point: python -m runform <command>

Commands mirror the build phases so each layer stays exercisable from
the terminal long before any UI exists (Phase 6 is deliberately last):

    analyze    Phase 1  video -> skeleton video, landmarks CSV, metrics, quality
    interpret  Phase 2  1-3 clip metrics + speeds -> deterministic assessment
    narrate    Phase 3  assessment -> LLM narrative (local Ollama)
    plan       Phase 4  assessment -> drill plan with success criteria
    compare    Phase 5  two assessments -> deltas with significance gate
"""

import argparse
import json
import sys

from .errors import RunFormError


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _emit(obj, out_path=None):
    text = json.dumps(obj, indent=2)
    # Write before printing: the artifact must survive even if stdout is
    # a pipe that goes away mid-print.
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text)
        print(f"Wrote {out_path}", file=sys.stderr)
    print(text)


def cmd_analyze(args):
    from .pipeline import analyze_clip  # lazy: pulls in mediapipe

    result = analyze_clip(
        args.video, out_dir=args.out_dir,
        model_variant=args.model_variant, smooth=args.smooth,
        frame_mode=args.frame_mode,
    )
    print(f"Frames:            {result['frames']} ({result['duration_s']} s @ {result['fps']:.6g} fps)")
    print(f"Detection rate:    {result['detection_rate']:.1%}")
    kv = result["key_joint_visibility"]
    if kv and kv.get("min") is not None:
        # Show the worst joint on each side: gating acts on it, and
        # seeing the two side by side makes a one-legged tracking failure
        # obvious in a way a single averaged number never did.
        print(
            f"Key-joint vis:     left {kv['left']['min']} (worst joint)  "
            f"right {kv['right']['min']} (worst joint)"
        )
    print(f"Steps detected:    {result['metrics'].get('steps_detected')}")
    print(f"Quality flags:     {result['quality_flags'] or 'none'}")
    print(f"Skeleton video:    {result['skeleton_video_path']}")
    print(f"Landmarks CSV:     {result['landmarks_csv_path']}")
    print(f"Metrics JSON:      {result['metrics_json_path']}")
    print(f"Quality JSON:      {result['quality_json_path']}")


def cmd_interpret(args):
    from .session import interpret_session

    clips = []
    for path, label, speed in args.clip:
        loaded = _load_json(path)
        # Accept either a bare metrics JSON or a Phase 1 quality/analysis
        # JSON that embeds "metrics".
        if "metrics" in loaded and isinstance(loaded["metrics"], dict):
            clips.append({
                "speed_label": label,
                "speed_value": float(speed),
                "metrics": loaded["metrics"],
                "detection_rate": loaded.get("detection_rate"),
                "quality_flags": loaded.get("quality_flags", []),
            })
        else:
            clips.append({
                "speed_label": label,
                "speed_value": float(speed),
                "metrics": loaded,
            })
    assessment = interpret_session(clips, notes=args.notes)
    _emit(assessment, args.out)


def cmd_narrate(args):
    from .narrative import narrate

    assessment = _load_json(args.assessment)
    result = narrate(assessment, model=args.model, host=args.host)
    print(result["narrative"])
    print(f"\n[model={result['model_name']} prompt_version={result['prompt_version']} "
          f"attempts={result['attempts']}]", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote {args.out}", file=sys.stderr)


def cmd_plan(args):
    from .plan import build_plan

    assessment = _load_json(args.assessment)
    _emit(build_plan(assessment, created_at=args.created_at), args.out)


def cmd_compare(args):
    from .comparison import compare_sessions

    result = compare_sessions(_load_json(args.before), _load_json(args.after))
    _emit(result, args.out)


def build_parser():
    p = argparse.ArgumentParser(
        prog="runform",
        description="Treadmill running form analysis (see BUILD_PLAN.md).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="video -> skeleton video, landmarks CSV, metrics + quality")
    a.add_argument("video")
    a.add_argument("--out-dir", default=None)
    a.add_argument("--model-variant", default="lite", choices=("lite", "full", "heavy"))
    a.add_argument("--frame-mode", default="image", choices=("video", "image"))
    a.add_argument("--smooth", type=int, default=9)
    a.set_defaults(func=cmd_analyze)

    i = sub.add_parser("interpret", help="clip metrics + speeds -> deterministic assessment")
    i.add_argument(
        "--clip", nargs=3, action="append", required=True,
        metavar=("METRICS_JSON", "LABEL", "SPEED"),
        help="repeat up to 3x, e.g. --clip easy_metrics.json easy 2.8",
    )
    i.add_argument("--notes", default="", help="session context: fatigue, shoes, injury status")
    i.add_argument("--out", default=None)
    i.set_defaults(func=cmd_interpret)

    n = sub.add_parser("narrate", help="assessment -> LLM narrative via local Ollama")
    n.add_argument("assessment")
    n.add_argument("--model", default="llama3.1:8b")
    n.add_argument("--host", default="http://localhost:11434")
    n.add_argument("--out", default=None)
    n.set_defaults(func=cmd_narrate)

    pl = sub.add_parser("plan", help="assessment -> training plan with success criteria")
    pl.add_argument("assessment")
    pl.add_argument("--created-at", default=None, help="ISO date; defaults to today")
    pl.add_argument("--out", default=None)
    pl.set_defaults(func=cmd_plan)

    c = sub.add_parser("compare", help="two assessments -> session-over-session deltas")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except RunFormError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
