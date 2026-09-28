"""Runs analyses one at a time in a background thread.

Analysis is CPU/GPU heavy, so running two at once would only make both slower.
Each run lives in its own folder under data/runs/<id>/ with a run.json holding
its status, so the list of runs survives a server restart.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from dms.pipeline import Cancelled, Options, analyze

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._live: dict[str, dict] = {}      # progress of queued/running jobs
        self._cancel: set[str] = set()
        self._mark_interrupted()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- public -----------------------------------------------------------------
    def submit(self, video_path: Path, video_name: str, options: dict) -> str:
        opts = Options(**{k: v for k, v in options.items() if k in Options.__dataclass_fields__})
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        run_dir = self.runs_dir / run_id
        run_dir.mkdir()
        meta = {
            "id": run_id,
            "status": "queued",
            "video": str(video_path),
            "video_name": video_name,
            "options": asdict(opts),
            "created": time.time(),
            "error": None,
        }
        self._save(run_id, meta)
        with self._lock:
            self._live[run_id] = {"progress": 0.0}
        self._queue.put(run_id)
        return run_id

    def get(self, run_id: str) -> dict | None:
        meta = self._load(run_id)
        if meta is None:
            return None
        with self._lock:
            meta["live"] = self._live.get(run_id)
        if meta["status"] == "done":
            report = self.runs_dir / run_id / "report.json"
            if report.exists():
                meta["report"] = json.loads(report.read_text())
        return meta

    def list(self) -> list[dict]:
        runs = []
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            meta = self._load(d.name)
            if not meta:
                continue
            summary = {k: meta[k] for k in ("id", "status", "video_name", "created", "error")}
            report = d / "report.json"
            if meta["status"] == "done" and report.exists():
                r = json.loads(report.read_text())
                summary.update(drivers=len(r["drivers"]), events=len(r["events"]), duration_s=r["duration_s"],
                               scores=[{k: c[k] for k in ("driver", "score", "grade")} for c in r["drivers"]])
            runs.append(summary)
        return runs

    def cancel(self, run_id: str) -> bool:
        meta = self._load(run_id)
        if not meta or meta["status"] not in ("queued", "running"):
            return False
        self._cancel.add(run_id)
        return True

    def delete(self, run_id: str) -> bool:
        run_dir = self.runs_dir / run_id
        if not run_dir.is_dir():
            return False
        self._cancel.add(run_id)
        shutil.rmtree(run_dir, ignore_errors=True)
        return True

    # -- internals --------------------------------------------------------------
    def _worker(self):
        while True:
            run_id = self._queue.get()
            meta = self._load(run_id)
            if meta is None or run_id in self._cancel:
                self._finish(run_id, meta, "cancelled")
                continue
            meta["status"] = "running"
            meta["started"] = time.time()
            self._save(run_id, meta)
            run_dir = self.runs_dir / run_id

            def progress(p, run_id=run_id):
                with self._lock:
                    self._live[run_id] = p

            try:
                analyze(
                    meta["video"],
                    run_dir,
                    Options(**meta["options"]),
                    on_progress=progress,
                    should_stop=lambda run_id=run_id: run_id in self._cancel,
                )
                self._finish(run_id, meta, "done")
            except Cancelled:
                self._finish(run_id, meta, "cancelled")
            except Exception as e:  # report any failure to the UI instead of dying
                log.exception("run %s failed", run_id)
                self._finish(run_id, meta, "failed", str(e))

    def _finish(self, run_id, meta, status, error=None):
        with self._lock:
            self._live.pop(run_id, None)
        self._cancel.discard(run_id)
        if meta is None or not (self.runs_dir / run_id).is_dir():
            return
        meta.update(status=status, error=error, finished=time.time())
        self._save(run_id, meta)

    def _mark_interrupted(self):
        for d in self.runs_dir.iterdir():
            meta = self._load(d.name)
            if meta and meta["status"] in ("queued", "running"):
                meta.update(status="failed", error="interrupted (server restarted)")
                self._save(d.name, meta)

    def _load(self, run_id: str) -> dict | None:
        path = self.runs_dir / run_id / "run.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _save(self, run_id: str, meta: dict):
        path = self.runs_dir / run_id / "run.json"
        if not path.parent.is_dir():
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v for k, v in meta.items() if k not in ("live", "report")}, indent=2))
        tmp.replace(path)
