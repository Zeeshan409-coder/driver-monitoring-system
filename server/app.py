"""Fleet dashboard: upload a trip video, get per-driver scorecards and incidents.

    uvicorn server.app:app --port 8000
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .jobs import JobManager

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DMS_DATA", ROOT / "data"))
VIDEOS = DATA / "videos"
SAMPLES = ROOT / "samples"
STATIC = Path(__file__).resolve().parent / "static"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_MB = int(os.environ.get("DMS_MAX_UPLOAD_MB", "1000"))

VIDEOS.mkdir(parents=True, exist_ok=True)
jobs = JobManager(DATA / "runs")
app = FastAPI(title="Driver Monitoring")


def _probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("not a readable video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, _ = cap.read()
    info = {"width": int(cap.get(3)), "height": int(cap.get(4)), "fps": round(fps, 2),
            "duration_s": round(frames / fps, 1) if fps else None}
    cap.release()
    if not ok:
        raise ValueError("could not read any frames")
    return info


def _video_path(video_id: str) -> Path:
    if not re.fullmatch(r"(sample:)?[\w.\-]+", video_id):
        raise HTTPException(404, "video not found")
    path = SAMPLES / video_id.split(":", 1)[1] if video_id.startswith("sample:") else VIDEOS / video_id
    if not path.is_file():
        raise HTTPException(404, "video not found")
    return path


@app.get("/api/videos")
def list_videos():
    items = []
    for folder, prefix in ((SAMPLES, "sample:"), (VIDEOS, "")):
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in VIDEO_EXT:
                continue
            meta = p.with_suffix(p.suffix + ".json")
            try:
                info = json.loads(meta.read_text()) if meta.exists() else {**_probe(p), "name": p.name}
            except ValueError:
                continue
            items.append({"id": prefix + p.name, "sample": bool(prefix), **info})
    return items


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(415, f"unsupported file type {ext or '?'}")
    vid = uuid.uuid4().hex[:10] + ext
    dest = VIDEOS / vid
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    try:
        info = _probe(dest)
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"could not read video: {e}") from e
    info["name"] = file.filename
    dest.with_suffix(ext + ".json").write_text(json.dumps(info))
    return {"id": vid, "sample": False, **info}


@app.get("/api/videos/{video_id}/thumb.jpg")
def thumb(video_id: str):
    cap = cv2.VideoCapture(str(_video_path(video_id)))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(n // 10, int(5 * (cap.get(cv2.CAP_PROP_FPS) or 25))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "could not read frame")
    frame = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), media_type="image/jpeg")


class RunRequest(BaseModel):
    video_id: str
    options: dict = {}


@app.post("/api/runs", status_code=202)
def start_run(req: RunRequest):
    path = _video_path(req.video_id)
    name = path.name
    meta = path.with_suffix(path.suffix + ".json")
    if meta.exists():
        name = json.loads(meta.read_text()).get("name", name)
    return {"id": jobs.submit(path, name, req.options)}


@app.get("/api/runs")
def list_runs():
    return jobs.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = jobs.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    return {"cancelled": jobs.cancel(run_id)}


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str):
    if not re.fullmatch(r"[\w\-]+", run_id) or not jobs.delete(run_id):
        raise HTTPException(404, "run not found")


app.mount("/files", StaticFiles(directory=DATA / "runs"), name="files")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
