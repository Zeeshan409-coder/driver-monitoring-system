"""Run stage 1 (the models) over a video and cache the observations as JSON.

Stage 1 is the slow part. With a cache, the behaviour rules can be tuned and
re-run in seconds:

    python scripts/extract.py samples/driver-action-recognition.mp4 --out runs/trip.obs.json
    python -m dms analyze samples/driver-action-recognition.mp4 --cache runs/trip.obs.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms.pipeline import Options, observe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--out")
    args = ap.parse_args()

    started, last = time.time(), [0.0]

    def progress(p):
        if time.time() - last[0] > 20:
            last[0] = time.time()
            print(f"{p['progress'] * 100:5.1f}%  {p['fps']} fps", flush=True)

    obs = observe(args.video, Options(analysis_fps=args.fps, width=args.width), on_progress=progress)
    out = Path(args.out or Path(args.video).with_suffix(".obs.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obs))
    print(f"wrote {out} ({len(obs['frames'])} frames, {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
