"""Video writing helpers. OpenCV writes mp4v, which browsers won't play, so we
re-encode to H.264 with ffmpeg when it's available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def to_h264(src: Path, dst: Path) -> bool:
    ff = ffmpeg_path()
    if not ff:
        shutil.move(src, dst)
        return False
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        shutil.move(src, dst)
        return False
    src.unlink(missing_ok=True)
    return True


def cut_clip(src: Path, dst: Path, start: float, duration: float) -> bool:
    ff = ffmpeg_path()
    if not ff:
        return False
    cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{max(start, 0):.2f}", "-i", str(src),
           "-t", f"{duration:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def merge_windows(times: list[float], pre: float, post: float, total: float) -> list[tuple[float, float]]:
    """Turn incident times into non-overlapping [start, end] clip windows."""
    windows: list[list[float]] = []
    for t in sorted(times):
        start, end = max(0.0, t - pre), min(total, t + post)
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    return [(round(s, 2), round(e, 2)) for s, e in windows]
