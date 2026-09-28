"""End-to-end: video in, annotated video + events + per-driver scorecards out."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import render
from .behaviour import BEHAVIOURS, BehaviourAnalyzer, Config, Event
from .identity import assign_drivers, upper_body_signature
from .scoring import scorecard
from .shots import ShotDetector
from .video import cut_clip, ffmpeg_path, to_h264

log = logging.getLogger(__name__)

DRIVER_COLORS = {1: (230, 160, 60), 2: (80, 200, 240), 3: (190, 110, 240), 4: (120, 220, 120), 5: (60, 120, 240)}


@dataclass
class Options:
    analysis_fps: float = 5.0
    width: int = 1280
    pose_model: str = "yolo11n-pose.pt"
    object_model: str = "yolo11s.pt"
    imgsz: int = 960
    driver_side: str = "right"
    clip_pad_s: float = 1.5


class Cancelled(Exception):
    pass


def _frames(video: str, fps: float, width: int):
    """Yield (t, frame) at roughly `fps`, resized to `width`, skipping frames without decoding."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(src_fps / fps))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / src_fps
            scale = width / frame.shape[1]
            if scale != 1:
                frame = cv2.resize(frame, (width, int(round(frame.shape[0] * scale / 2)) * 2),
                                   interpolation=cv2.INTER_AREA)
            yield t, frame
            for _ in range(stride - 1):
                if not cap.grab():
                    return
            i += stride
    finally:
        cap.release()


def video_info(video: str) -> dict:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"fps": fps, "frames": n, "duration": n / fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def observe(video: str, opts: Options, perceiver=None, on_progress=None, should_stop=None) -> dict:
    """Stage 1: run the models over the video. Returns the observations (cacheable)."""
    if perceiver is None:
        from .perception import Perceiver

        perceiver = Perceiver(opts.pose_model, opts.object_model, opts.imgsz)
    info = video_info(video)
    shots = ShotDetector()
    frames, started, size = [], time.time(), None
    for t, frame in _frames(video, opts.analysis_fps, opts.width):
        if should_stop and should_stop():
            raise Cancelled()
        size = [frame.shape[1], frame.shape[0]]
        shot, _ = shots.update(frame, t)
        obs = perceiver(frame)
        for p in obs["people"]:
            p["sig"] = upper_body_signature(frame, p)
        frames.append({"t": round(t, 3), "shot": shot, **obs})
        if on_progress:
            elapsed = time.time() - started
            on_progress({"stage": "detecting", "progress": min(1.0, t / info["duration"]) if info["duration"] else 0,
                         "fps": round(len(frames) / elapsed, 1) if elapsed else 0})
    return {"video": Path(video).name, "fps": opts.analysis_fps, "size": size, "frames": frames}


def backfill_signatures(video: str, obs: dict, opts: Options) -> None:
    """Add clothing signatures to observations cached before signatures existed."""
    if all("sig" in p for f in obs["frames"] for p in f["people"]):
        return
    it = iter(obs["frames"])
    current = next(it, None)
    for t, frame in _frames(video, opts.analysis_fps, opts.width):
        if current is None:
            break
        if abs(t - current["t"]) < 1e-3:
            for p in current["people"]:
                p["sig"] = upper_body_signature(frame, p)
            current = next(it, None)


def analyze(video: str, out_dir: str | Path, opts: Options | None = None, obs: dict | None = None,
            perceiver=None, on_progress: Callable[[dict], None] | None = None,
            should_stop: Callable[[], bool] | None = None, render_video: bool = True) -> dict:
    opts = opts or Options()
    out = Path(out_dir)
    for sub in ("clips", "snapshots"):  # a re-run replaces the old results
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).iterdir():
            old.unlink()
    started = time.time()

    if obs is None:
        obs = observe(video, opts, perceiver, on_progress, should_stop)
    else:
        backfill_signatures(video, obs, opts)
    frames = obs["frames"]
    if not frames:
        raise ValueError("the video contains no readable frames")
    width, height = obs["size"]
    detect_s = time.time() - started

    # stage 2
    analyzer = BehaviourAnalyzer(width, height, Config(driver_side=opts.driver_side))
    states, events = analyzer.analyse(frames)
    fixtures = analyzer.fixtures

    # who is driving
    sigs = [s.driver.get("sig") if s.driver else None for s in states]
    shot_driver = assign_drivers([s.shot for s in states], sigs)
    for e in events:
        shot = next(s.shot for s in states if s.t >= e.start - 1e-6)
        e.driver = shot_driver[shot]

    info = video_info(video)
    duration = info["duration"]
    dt = 1.0 / opts.analysis_fps
    spans = []  # (driver, start, end)
    for s in states:
        d = shot_driver[s.shot]
        if spans and spans[-1][0] == d:
            spans[-1][2] = s.t + dt
        else:
            spans.append([d, s.t, s.t + dt])
    driving = defaultdict(float)
    for d, a, b in spans:
        driving[d] += b - a
    cards = [scorecard(d, driving[d], events) for d in sorted(driving)]

    # stage 3: render
    video_out = None
    if render_video:
        video_out = _render(video, out, opts, frames, states, events, fixtures, shot_driver, spans, duration,
                            on_progress, should_stop)
    for i, e in enumerate(events, start=1):
        e_dict_clip = None
        if video_out is not None:
            start = max(0.0, e.start - opts.clip_pad_s)
            end = min(duration, e.end + opts.clip_pad_s, start + 25)
            name = f"clips/event_{i:03d}.mp4"
            if cut_clip(video_out, out / name, start, end - start):
                e_dict_clip = name
        e._clip = e_dict_clip  # type: ignore[attr-defined]

    report = {
        "video": obs["video"],
        "duration_s": round(duration, 2),
        "analysis_fps": opts.analysis_fps,
        "frames_analysed": len(frames),
        "processing_s": round(time.time() - started, 1),
        "detection_s": round(detect_s, 1),
        "drivers": cards,
        "driver_spans": [{"driver": d, "start": round(a, 2), "end": round(b, 2)} for d, a, b in spans],
        "events": [{**e.to_dict(), "id": i, "clip": e._clip,  # type: ignore[attr-defined]
                    "snapshot": f"snapshots/event_{i:03d}.jpg"} for i, e in enumerate(events, start=1)],
        "shots": len(set(s.shot for s in states)),
        "options": asdict(opts),
        "behaviours": list(BEHAVIOURS),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with (out / "events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "driver", "behaviour", "start_s", "end_s", "duration_s", "confidence", "evidence"])
        for ev in report["events"]:
            w.writerow([ev["id"], ev["driver"], ev["behaviour"], ev["start"], ev["end"], ev["duration"],
                        ev["confidence"], ev["evidence"]])
    return report


def _all_frames(video: str, width: int):
    """Every frame of the video, resized to `width`: (t, frame)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width:
                h = int(round(frame.shape[0] * width / frame.shape[1] / 2)) * 2
                frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
            yield i / fps, frame
            i += 1
    finally:
        cap.release()


def _blend(a: dict, b: dict, w: float) -> dict:
    """Driver pose part-way between two analysed frames, so the skeleton moves smoothly."""
    out = dict(a)
    out["box"] = [x + (y - x) * w for x, y in zip(a["box"], b["box"], strict=True)]
    if a["kpts"] is not None and b["kpts"] is not None:
        out["kpts"] = [[ka[0] + (kb[0] - ka[0]) * w, ka[1] + (kb[1] - ka[1]) * w, min(ka[2], kb[2])]
                       for ka, kb in zip(a["kpts"], b["kpts"], strict=True)]
    return out


def _render(video, out: Path, opts: Options, frames, states, events: list[Event], fixtures, shot_driver, spans,
            duration, on_progress, should_stop):
    """Draws the overlays on every frame of the video, so the result plays at normal speed and
    at the original frame rate. Between two analysed frames the driver's skeleton is interpolated."""
    import bisect
    from dataclasses import replace

    raw = out / "annotated_raw.mp4"
    h, w = render.CANVAS
    src_fps = video_info(video)["fps"]
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
    times = [f["t"] for f in frames]
    kept = [[o for o, fx in zip(f["objects"], m, strict=True) if fx] for f, m in zip(frames, fixtures, strict=True)]
    peaks = {i: e.peak_t for i, e in enumerate(events, start=1)}
    panel = np.zeros((render.VIDEO_H, render.PANEL_W, 3), np.uint8)
    strip = np.zeros((render.STRIP_H, w, 3), np.uint8)
    driver_time = defaultdict(float)
    span_list = [(d, a, b) for d, a, b in spans]
    dt = 1.0 / src_fps
    last_progress = 0.0
    for t, frame in _all_frames(video, render.VIDEO_W):
        if should_stop and should_stop():
            writer.release()
            raise Cancelled()
        i = max(0, bisect.bisect_right(times, t + 1e-6) - 1)
        f, st = frames[i], states[i]
        if st.driver is not None and i + 1 < len(frames) and states[i + 1].driver is not None \
                and states[i + 1].shot == st.shot and times[i + 1] > times[i]:
            wgt = min(1.0, (t - times[i]) / (times[i + 1] - times[i]))
            blended = _blend(st.driver, states[i + 1].driver, wgt)
            f = {**f, "people": [blended if p is st.driver else p for p in f["people"]]}
            st = replace(st, driver=blended)
        img = frame if frame.shape[:2] == (render.VIDEO_H, render.VIDEO_W) else \
            cv2.resize(frame, (render.VIDEO_W, render.VIDEO_H))
        driver = shot_driver[st.shot] if st.driver is not None else None
        if driver is not None:
            driver_time[driver] += dt
        active_ev = [e for e in events if e.start <= t < e.end and e.driver == shot_driver[st.shot]]
        active = [e.behaviour for e in active_ev]
        render.draw_frame_overlays(img, f, st, active, kept[i])
        render.draw_alert(img, active_ev, t)
        started = [e for e in events if e.start <= t]
        render.draw_panel(panel, driver, t, started, driver_time.get(driver, 0.0), active, DRIVER_COLORS)
        render.draw_strip(strip, t, duration, events, span_list, DRIVER_COLORS)
        canvas = render.compose(img, panel, strip)
        writer.write(canvas)
        for k, pt in peaks.items():
            if abs(pt - t) < dt / 2:
                cv2.imwrite(str(out / "snapshots" / f"event_{k:03d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if on_progress and t - last_progress >= 1.0:
            last_progress = t
            on_progress({"stage": "rendering", "progress": min(1.0, t / duration) if duration else 0})
    writer.release()
    final = out / "annotated.mp4"
    to_h264(raw, final)
    return final if ffmpeg_path() else None
