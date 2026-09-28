"""Command line.

    python -m dms analyze trip.mp4                 # full analysis
    python -m dms analyze trip.mp4 --cache obs.json   # reuse / save the model outputs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .behaviour import LABELS
from .pipeline import Options, analyze


def cmd_analyze(args) -> int:
    opts = Options(analysis_fps=args.fps, driver_side=args.driver_side, pose_model=args.pose_model,
                   object_model=args.object_model)
    out = Path(args.out or Path("runs") / Path(args.video).stem)
    obs = None
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        obs = json.loads(cache.read_text())
        print(f"using cached detections from {cache}")

    last = [0.0]

    def progress(p):
        if time.time() - last[0] > 1:
            last[0] = time.time()
            extra = f"  {p['fps']} fps" if "fps" in p else ""
            sys.stdout.write(f"\r  {p['stage']:<10} {p['progress'] * 100:5.1f}%{extra}   ")
            sys.stdout.flush()

    if obs is None and cache:
        from .pipeline import observe

        obs = observe(args.video, opts, on_progress=progress)
        cache.write_text(json.dumps(obs))
    report = analyze(args.video, out, opts, obs=obs, on_progress=progress, render_video=not args.no_video)
    print()
    for c in report["drivers"]:
        kinds = ", ".join(f"{LABELS[b].lower()} x{v['events']}" for b, v in c["by_behaviour"].items()) or "none"
        print(f"  driver {c['driver']}: score {c['score']} ({c['grade']}), {c['events']} events, "
              f"distracted {c['distracted_pct']}% of {c['driving_s']:.0f}s  [{kinds}]")
    print(f"  results in {out}/")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dms", description="Driver monitoring from cabin video")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse a video file")
    a.add_argument("video")
    a.add_argument("--out")
    a.add_argument("--fps", type=float, default=5.0, help="frames analysed per second")
    a.add_argument("--driver-side", choices=["left", "right"], default="right",
                   help="side of the image the driver sits on")
    a.add_argument("--pose-model", default="yolo11n-pose.pt")
    a.add_argument("--object-model", default="yolo11s.pt")
    a.add_argument("--cache", help="JSON file to save/reuse the model outputs")
    a.add_argument("--no-video", action="store_true", help="skip rendering the annotated video")
    a.set_defaults(func=cmd_analyze)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
