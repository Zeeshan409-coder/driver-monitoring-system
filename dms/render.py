"""Draws the annotated output: video with overlays, a live scorecard panel on the
right and a timeline of the whole trip along the bottom."""

from __future__ import annotations

import cv2
import numpy as np

from .behaviour import BEHAVIOURS, LABELS
from .scoring import WEIGHTS, grade

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {  # BGR
    "phone_call": (70, 70, 235),
    "texting": (200, 70, 200),
    "drinking": (0, 150, 255),
    "reaching": (0, 205, 235),
    "looking_away": (140, 110, 255),
    "hand_to_face": (205, 170, 60),
}
OK = (90, 200, 90)
PANEL_BG = (32, 30, 28)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)

SHORT = {"phone_call": "Phone call", "texting": "Using phone", "drinking": "Drinking",
         "reaching": "Reaching", "looking_away": "Looking away",
         "hand_to_face": "Hand at face"}

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4)]

VIDEO_W, VIDEO_H = 1280, 720
PANEL_W = 340
STRIP_H = 70
CANVAS = (VIDEO_H + STRIP_H, VIDEO_W + PANEL_W)


def _text(img, s, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _tag(img, s, org, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 1)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 10))
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), color, -1)
    _text(img, s, (x + 5, y - 5), scale, (255, 255, 255))


def draw_frame_overlays(img, frame_obs: dict, state, active: list, fixtures: list[dict]):
    """Driver skeleton + box, passengers greyed out, objects near the driver, ignored fixtures."""
    driver = state.driver
    for p in frame_obs["people"]:
        if driver is not None and p is driver:
            continue
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        if (y2 - y1) < 0.2 * VIDEO_H:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1)
        _tag(img, "passenger - ignored", (x1, y1), (95, 95, 95), 0.45)

    for o in fixtures:
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (130, 130, 130), 1)
        _tag(img, f"static {o['cls']} - ignored", (x1, y1), (95, 95, 95), 0.4)

    if driver is None:
        return
    color = COLORS[active[0]] if active else OK
    k = driver["kpts"]
    for a, b in SKELETON:
        if k[a][2] > 0.35 and k[b][2] > 0.35:
            cv2.line(img, (int(k[a][0]), int(k[a][1])), (int(k[b][0]), int(k[b][1])), (255, 220, 120), 2, cv2.LINE_AA)
    for x, y, c in k[:13]:  # face, arms and hips; legs aren't used, so they aren't drawn
        if c > 0.35:
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    x1, y1, x2, y2 = (int(v) for v in driver["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    _tag(img, "DRIVER", (x1, y1), color)

    for o in state.objects:
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        c = COLORS["drinking"] if o["cls"] != "phone" else COLORS["texting"]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        _tag(img, o["cls"], (x1, y2 + 22), c, 0.45)


def draw_alert(img, active_events: list, t: float):
    if not active_events:
        return
    e = active_events[0]
    s = f"{LABELS[e.behaviour].upper()}   {t - e.start:4.1f}s"
    (tw, th), _ = cv2.getTextSize(s, FONT, 0.9, 2)
    x = (VIDEO_W - tw) // 2
    cv2.rectangle(img, (x - 18, 14), (x + tw + 18, 14 + th + 22), COLORS[e.behaviour], -1)
    _text(img, s, (x, 14 + th + 11), 0.9, (255, 255, 255), 2)


def draw_panel(panel, driver: int | None, t: float, events_so_far: list, driving_so_far: float,
               active: list, driver_colors: dict):
    panel[:] = PANEL_BG
    _text(panel, "DRIVER MONITOR", (20, 34), 0.55, MUTED)
    if driver is None:
        _text(panel, "no driver visible", (20, 80), 0.6, MUTED)
        return
    cv2.rectangle(panel, (20, 52), (30, 84), driver_colors.get(driver, OK), -1)
    _text(panel, f"Driver {driver}", (42, 78), 0.85, TEXT, 2)

    # score gauge
    mine = [e for e in events_so_far if e.driver == driver]
    points = sum(WEIGHTS[e.behaviour][0] + WEIGHTS[e.behaviour][1] * min(e.duration, max(0.0, t - e.start))
                 for e in mine)
    score = max(0.0, 100.0 - points)
    cx, cy, r = PANEL_W // 2, 180, 74
    cv2.ellipse(panel, (cx, cy), (r, r), 0, 135, 405, (70, 70, 70), 12, cv2.LINE_AA)
    col = OK if score >= 75 else (0, 190, 255) if score >= 50 else (70, 70, 235)
    cv2.ellipse(panel, (cx, cy), (r, r), 0, 135, 135 + 270 * score / 100, col, 12, cv2.LINE_AA)
    s = f"{score:.0f}"
    (tw, _), _ = cv2.getTextSize(s, FONT, 1.6, 3)
    _text(panel, s, (cx - tw // 2, cy + 14), 1.6, TEXT, 3)
    label = f"safety score  -  grade {grade(score)}"
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    _text(panel, label, (cx - tw // 2, cy + r + 22), 0.5, MUTED)

    # current state
    y = 312
    if active:
        cv2.rectangle(panel, (20, y), (PANEL_W - 20, y + 40), COLORS[active[0]], -1)
        _text(panel, "DISTRACTED", (32, y + 27), 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(panel, (20, y), (PANEL_W - 20, y + 40), (40, 110, 40), -1)
        _text(panel, "ATTENTIVE", (32, y + 27), 0.7, (255, 255, 255), 2)

    # per behaviour counters
    y = 400
    _text(panel, "this driver so far", (20, y - 12), 0.45, MUTED)
    for b in BEHAVIOURS:
        evs = [e for e in mine if e.behaviour == b]
        secs = sum(min(e.duration, max(0.0, t - e.start)) for e in evs)
        on = b in active
        cv2.circle(panel, (28, y + 12), 6, COLORS[b], -1 if (evs or on) else 1, cv2.LINE_AA)
        _text(panel, SHORT[b], (44, y + 17), 0.5, TEXT if (evs or on) else MUTED, 2 if on else 1)
        _text(panel, f"{len(evs)}x  {secs:5.1f}s", (PANEL_W - 118, y + 17), 0.5, TEXT if evs else MUTED)
        y += 38

    distracted = sum(min(e.duration, max(0.0, t - e.start)) for e in mine)
    pct = 100 * distracted / driving_so_far if driving_so_far > 0 else 0
    _text(panel, f"eyes/hands off task: {pct:4.1f}% of {driving_so_far:4.0f}s", (20, VIDEO_H - 24), 0.45, MUTED)


def draw_strip(strip, t: float, duration: float, events: list, driver_spans: list, driver_colors: dict):
    strip[:] = (22, 22, 22)
    w = strip.shape[1]
    x_of = lambda s: int(12 + (w - 24) * s / max(duration, 1e-6))  # noqa: E731
    for d, s, e in driver_spans:
        cv2.rectangle(strip, (x_of(s), 8), (x_of(e), 16), driver_colors.get(d, OK), -1)
        _text(strip, f"driver {d}", (x_of(s) + 4, 32), 0.4, MUTED)
    rows = {b: 36 + 5 * i for i, b in enumerate(BEHAVIOURS)}
    for e in events:
        y = rows[e.behaviour]
        cv2.rectangle(strip, (x_of(e.start), y), (max(x_of(e.end), x_of(e.start) + 2), y + 4), COLORS[e.behaviour], -1)
    x = x_of(t)
    cv2.line(strip, (x, 4), (x, strip.shape[0] - 4), (255, 255, 255), 2)


def compose(video_frame, panel, strip):
    canvas = np.zeros((*CANVAS, 3), np.uint8)
    canvas[:VIDEO_H, :VIDEO_W] = video_frame
    canvas[:VIDEO_H, VIDEO_W:] = panel
    canvas[VIDEO_H:, :] = strip
    return canvas
