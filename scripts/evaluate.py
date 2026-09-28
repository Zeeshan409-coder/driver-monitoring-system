"""Compare detected events with hand-labelled ground truth.

    python scripts/evaluate.py runs/trip/report.json labels/driver-action-recognition.json

Two views:
  * events: a detection counts as correct if it overlaps a labelled event of the
    same behaviour by at least 30% of the shorter of the two.
  * time: every 0.2 s is either labelled with a behaviour or not; precision and
    recall are computed over those time slices.
"""

import argparse
import json

import numpy as np

BEHAVIOURS = ("phone_call", "texting", "drinking", "reaching", "looking_away", "hand_to_face")


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def match(pred, truth):
    return overlap(pred, truth) >= 0.3 * min(pred["end"] - pred["start"], truth["end"] - truth["start"])


def evaluate(pred_events, gt, step=0.2):
    duration = max(d["end"] for d in gt["drivers"])
    rows = {}
    for b in BEHAVIOURS:
        p = [e for e in pred_events if e["behaviour"] == b]
        g = [e for e in gt["events"] if e["behaviour"] == b]
        tp_p = sum(any(match(x, y) for y in g) for x in p)
        tp_g = sum(any(match(x, y) for x in p) for y in g)
        ts = np.arange(0, duration, step)
        pm = np.array([any(e["start"] <= t < e["end"] for e in p) for t in ts])
        gm = np.array([any(e["start"] <= t < e["end"] for e in g) for t in ts])
        rows[b] = {
            "labelled": len(g), "detected": len(p),
            "event_recall": tp_g / len(g) if g else None,
            "event_precision": tp_p / len(p) if p else None,
            "time_recall": float((pm & gm).sum() / gm.sum()) if gm.sum() else None,
            "time_precision": float((pm & gm).sum() / pm.sum()) if pm.sum() else None,
        }
    # any distraction vs attentive, per time slice (labelled 'controls' counts as distracted)
    ts = np.arange(0, duration, step)
    pm = np.array([any(e["start"] <= t < e["end"] for e in pred_events) for t in ts])
    gm = np.array([any(e["start"] <= t < e["end"] for e in gt["events"]) for t in ts])
    overall = {
        "accuracy": float((pm == gm).mean()),
        "recall": float((pm & gm).sum() / gm.sum()),
        "precision": float((pm & gm).sum() / pm.sum()) if pm.sum() else 0.0,
    }
    all_g = [e for e in gt["events"] if e["behaviour"] in BEHAVIOURS]
    all_p = pred_events
    def same(x, y):
        return x["behaviour"] == y["behaviour"] and match(x, y)

    ev_rec = sum(any(same(x, y) for x in all_p) for y in all_g) / len(all_g)
    ev_pre = sum(any(same(x, y) for y in all_g) for x in all_p) / max(1, len(all_p))
    return rows, overall, {"event_recall": ev_rec, "event_precision": ev_pre, "labelled": len(all_g),
                           "detected": len(all_p)}


def fmt(v):
    return "  -  " if v is None else f"{v * 100:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("labels")
    args = ap.parse_args()
    pred = json.load(open(args.report))["events"]
    gt = json.load(open(args.labels))
    rows, overall, ev = evaluate(pred, gt)
    print(f"{'behaviour':<14}{'labelled':>9}{'detected':>9}"
          "   event recall  event precision   time recall  time precision")
    for b, r in rows.items():
        print(f"{b:<14}{r['labelled']:>9}{r['detected']:>9}"
              f"   {fmt(r['event_recall']):>12}  {fmt(r['event_precision']):>15}"
              f"   {fmt(r['time_recall']):>11}  {fmt(r['time_precision']):>14}")
    print(f"\nall events: recall {fmt(ev['event_recall'])} ({ev['labelled']} labelled), "
          f"precision {fmt(ev['event_precision'])} ({ev['detected']} detected)")
    print(f"distracted vs attentive, per 0.2 s: accuracy {fmt(overall['accuracy'])}, "
          f"recall {fmt(overall['recall'])}, precision {fmt(overall['precision'])}")
    print(json.dumps({"per_behaviour": rows, "overall": overall, "events": ev}))


if __name__ == "__main__":
    main()
