import importlib
import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def fake_analyze(video, out_dir, opts, on_progress=None, should_stop=None):
    on_progress({"stage": "detecting", "progress": 0.5})
    report = {
        "video": "x.mp4", "duration_s": 10, "drivers": [{"driver": 1, "score": 88, "grade": "B"}],
        "events": [{"id": 1, "behaviour": "texting"}], "driver_spans": [], "behaviours": [],
    }
    (out_dir / "report.json").write_text(json.dumps(report))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DMS_DATA", str(tmp_path))
    import server.jobs

    monkeypatch.setattr(server.jobs, "analyze", fake_analyze)
    import server.app

    app_module = importlib.reload(server.app)
    return TestClient(app_module.app), tmp_path


def make_video(path):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(30):
        w.write(np.full((240, 320, 3), i * 8, np.uint8))
    w.release()


def wait_done(c, run_id):
    for _ in range(100):
        run = c.get(f"/api/runs/{run_id}").json()
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_upload_run_and_report(client, tmp_path):
    c, data = client
    make_video(tmp_path / "clip.mp4")
    with open(tmp_path / "clip.mp4", "rb") as f:
        r = c.post("/api/videos", files={"file": ("cab.mp4", f, "video/mp4")})
    assert r.status_code == 200
    vid = r.json()["id"]
    assert r.json()["name"] == "cab.mp4" and r.json()["width"] == 320
    assert any(v["id"] == vid for v in c.get("/api/videos").json())
    assert c.get(f"/api/videos/{vid}/thumb.jpg").headers["content-type"] == "image/jpeg"

    run_id = c.post("/api/runs", json={"video_id": vid}).json()["id"]
    run = wait_done(c, run_id)
    assert run["status"] == "done"
    assert run["report"]["drivers"][0]["score"] == 88
    listed = c.get("/api/runs").json()[0]
    assert listed["events"] == 1 and listed["scores"] == [{"driver": 1, "score": 88, "grade": "B"}]
    assert c.get(f"/files/{run_id}/report.json").status_code == 200

    assert c.delete(f"/api/runs/{run_id}").status_code == 204
    assert c.get(f"/api/runs/{run_id}").status_code == 404


def test_rejects_non_video(client):
    c, _ = client
    r = c.post("/api/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = c.post("/api/videos", files={"file": ("broken.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400


def test_unknown_ids(client):
    c, _ = client
    assert c.post("/api/runs", json={"video_id": "../../etc/passwd"}).status_code == 404
    assert c.get("/api/runs/nope").status_code == 404
    assert c.delete("/api/runs/nope").status_code == 404


def test_index_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "Driver Monitoring" in r.text
