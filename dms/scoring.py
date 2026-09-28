"""Per-driver safety score, in the style of fleet telematics scorecards.

Every distraction costs points: a fixed amount for the event plus an amount per
second it lasts. Handheld phone use is weighted highest; it's the distraction
most strongly linked to crashes. The score is 100 minus the points, floored at 0.

The weights are a judgement call, not a standard. They're in one table so a
fleet can tune them.
"""

from __future__ import annotations

WEIGHTS = {  # behaviour: (points per event, points per second)
    "texting": (6.0, 0.6),
    "phone_call": (4.0, 0.4),
    "reaching": (4.0, 0.5),
    "drinking": (2.0, 0.2),
    "looking_away": (2.0, 0.5),
    "hand_to_face": (1.0, 0.1),
}


def grade(score: float) -> str:
    return "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "E"


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def scorecard(driver: int, driving_s: float, events: list) -> dict:
    mine = [e for e in events if e.driver == driver]
    points = sum(WEIGHTS[e.behaviour][0] + WEIGHTS[e.behaviour][1] * e.duration for e in mine)
    score = max(0.0, 100.0 - points)
    distracted = union_seconds([(e.start, e.end) for e in mine])
    by_type = {}
    for e in mine:
        b = by_type.setdefault(e.behaviour, {"events": 0, "seconds": 0.0})
        b["events"] += 1
        b["seconds"] = round(b["seconds"] + e.duration, 1)
    return {
        "driver": driver,
        "driving_s": round(driving_s, 1),
        "score": round(score),
        "grade": grade(score),
        "events": len(mine),
        "distracted_s": round(distracted, 1),
        "distracted_pct": round(100 * distracted / driving_s, 1) if driving_s else 0.0,
        "by_behaviour": by_type,
    }
