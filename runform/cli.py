"""Command-line entry point: python -m runform <command>

Commands mirror the build phases so each layer stays exercisable from
the terminal long before any UI exists (Phase 6 is deliberately last):

    analyze    Phase 1  video -> skeleton video, landmarks CSV, metrics, quality
    relabel    Phase 1  re-run leg-identity tracking with the seed swapped (no pose rerun)
    interpret  Phase 2  1-3 clip metrics + speeds -> deterministic assessment
    narrate    Phase 3  assessment -> LLM narrative (local Ollama)
    plan       Phase 4  assessment -> drill plan with success criteria
    compare    Phase 5  two assessments -> deltas with significance gate
    ui         Phase 6  local web UI over all of the above
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
    from .pipeline import analyze_clip  # lazy: pulls in rtmlib/onnxruntime

    result = analyze_clip(
        args.video, out_dir=args.out_dir,
        mode=args.mode, device=args.device, smooth=args.smooth,
        swap_seed=args.swap_seed,
    )
    _print_analysis(result)


def cmd_relabel(args):
    from .pipeline import relabel_clip

    _print_analysis(relabel_clip(args.quality_json, swap_seed=args.swap_seed,
                                 smooth=args.smooth))


def _print_analysis(result):
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
    ident = result["leg_identity"]
    if ident["seed_frame"] is None:
        print("Leg identity:      no seed frame (legs never clearly visible and apart)")
    else:
        cov = ident["labeled_fraction"]
        print(
            f"Leg identity:      seed frame {ident['seed_frame']}"
            f"{' (swapped)' if ident['swap_seed'] else ''}, "
            f"{ident['swaps_corrected']} label swaps corrected, "
            f"labeled L {cov['left']:.0%} / R {cov['right']:.0%}"
        )
    print(f"Quality flags:     {result['quality_flags'] or 'none'}")
    print(f"Skeleton video:    {result['skeleton_video_path']}")
    print(f"Landmarks CSV:     {result['landmarks_csv_path']}")
    print(f"Tracked CSV:       {result['tracked_landmarks_csv_path']}")
    if result.get("seed_image_path"):
        print(f"Seed frame image:  {result['seed_image_path']}  "
              f"(red dot = LEFT; if wrong, run: relabel <quality json> --swap-seed)")
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


def cmd_ui(args):
    from .webui import run

    run(host=args.host, port=args.port, root=args.data_root,
        open_browser=not args.no_browser)


def build_parser():
    p = argparse.ArgumentParser(
        prog="runform",
        description="Treadmill running form analysis (see BUILD_PLAN.md).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="video -> skeleton video, landmarks CSV, metrics + quality")
    a.add_argument("video")
    a.add_argument("--out-dir", default=None)
    a.add_argument("--mode", default="balanced",
                   choices=("performance", "balanced", "lightweight"),
                   help="rtmlib accuracy/speed preset (default: balanced)")
    a.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"),
                   help="onnxruntime execution device (default: cpu)")
    a.add_argument("--smooth", type=int, default=9)
    a.add_argument("--swap-seed", action="store_true",
                   help="the model's left/right labels on the seed frame are backwards")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("relabel", help="re-run leg-identity tracking + metrics on an analyzed clip")
    r.add_argument("quality_json", help="the <clip>_quality.json written by analyze")
    r.add_argument("--swap-seed", action="store_true",
                   help="red dot on the seed frame image is on the RIGHT foot")
    r.add_argument("--smooth", type=int, default=9)
    r.set_defaults(func=cmd_relabel)

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

    u = sub.add_parser("ui", help="local web UI: capture, results, narrative, plan, progress")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8177)
    u.add_argument("--data-root", default="data", help="storage root (default: ./data)")
    u.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    u.set_defaults(func=cmd_ui)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except RunFormError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
