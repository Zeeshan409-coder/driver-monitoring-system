"""Stage 2: from per-frame observations to driver behaviour events.

Steps:
  1. find the driver in every frame (and ignore passengers),
  2. drop objects that never move within a shot (e.g. a phone mounted on the
     windscreen is not "phone use"),
  3. describe the driver's pose relative to their own body size,
  4. calibrate a per-shot baseline (where this driver normally sits and how
     big they appear),
  5. label each frame, then smooth over time into events.

All distances are measured in "body units": a fifth of the driver's usual height
in the frame for that shot (about one shoulder width when facing the camera).
So thresholds work the same whether the driver sits close to the camera or far
from it, and don't jump when the driver turns sideways and the shoulders look
narrow.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from .perception import (
    L_EAR,
    L_ELBOW,
    L_EYE,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_EAR,
    R_ELBOW,
    R_EYE,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)

BEHAVIOURS = ("phone_call", "texting", "drinking", "reaching", "looking_away", "hand_to_face")
LABELS = {
    "phone_call": "Phone call",
    "texting": "Using phone",
    "drinking": "Drinking",
    "reaching": "Reaching / turned away",
    "looking_away": "Looking away",
    "hand_to_face": "Hand at face",
}
PRIORITY = {b: i for i, b in enumerate(BEHAVIOURS)}  # earlier wins when several apply


@dataclass
class Config:
    driver_side: str = "right"        # which side of the image the driver sits on
    min_driver_height: float = 0.25   # fraction of frame height
    kpt_conf: float = 0.35
    head_radius: float = 1.2          # wrist/object this close to the head (in body units) = "at head"
    hand_radius: float = 1.0          # object this close to a wrist = "in hand"
    reach_shift: float = 0.55         # torso moved this far from its usual place = leaning / turned away
    reach_side: float = 1.3           # wrist this far past the torso towards the passenger side = reaching
    turn_shoulders: float = 0.6       # shoulders look this much wider than usual = body turned towards the back
    hand_on_head: float = 1.8         # wrist above the shoulders and this close to the head = hand on the head
    grip: float = 0.6                 # wrist this close to an object's outline = holding it
    memory_s: float = 8.0             # hand at the head this soon after a phone/drink was seen = still using it
    memory_min_sightings: int = 5     # ...but only if it was clearly seen (not a few stray detections)
    memory_conf: float = 0.3
    ear_hold: float = 0.5             # holding a phone to the ear puts the wrist this far below the ear
    hold_grace_s: float = 1.5         # an ongoing hidden call survives the hand keypoint jumping away this long
    max_phone_size: float = 1.0       # "phones" bigger than this (body units) are false detections
    hold_gap_s: float = 6.0           # a held bottle often vanishes behind the hand: bridge gaps this long
    window_s: float = 1.2             # majority vote window for frame labels
    min_event_s: float = 1.5
    head_turn: float = 0.3            # head facing the other way than usual by this much = looking away
    glance_min_s: float = 0.8         # glances are short: shorter minimum and smoothing for looking away
    merge_gap_s: float = 2.0
    static_iou: float = 0.6           # objects this still across a shot are fixtures
    static_fraction: float = 0.5
    static_span_s: float = 30.0       # ...or keep turning up in the same spot over this long


@dataclass
class Event:
    behaviour: str
    start: float
    end: float
    driver: int
    peak_t: float
    confidence: float
    evidence: str = ""

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        d["label"] = LABELS[self.behaviour]
        return d


@dataclass
class FrameState:
    t: float
    shot: int
    driver: dict | None = None
    objects: list = field(default_factory=list)   # objects near the driver (fixtures removed)
    flags: dict = field(default_factory=dict)      # behaviour -> evidence string
    features: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers
def _pt(kpts, i, conf):
    if kpts is None:
        return None
    x, y, c = kpts[i]
    return np.array([x, y]) if c >= conf else None


def _mean(points):
    pts = [p for p in points if p is not None]
    return np.mean(pts, axis=0) if pts else None


def _center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _box_dist(pt, box):
    """Distance from a point to a box (0 inside)."""
    dx = max(box[0] - pt[0], 0.0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0.0, pt[1] - box[3])
    return float(np.hypot(dx, dy))


def _inside(pt, box, margin=0.0):
    w, h = box[2] - box[0], box[3] - box[1]
    in_x = box[0] - margin * w <= pt[0] <= box[2] + margin * w
    in_y = box[1] - margin * h <= pt[1] <= box[3] + margin * h
    return in_x and in_y


# ----------------------------------------------------------------------------- analyser
class BehaviourAnalyzer:
    def __init__(self, width: int, height: int, config: Config | None = None):
        self.w, self.h = width, height
        self.cfg = config or Config()

    # 1. driver selection ------------------------------------------------------------
    def _candidates(self, people):
        return [p for p in people
                if p["kpts"] is not None and (p["box"][3] - p["box"][1]) >= self.cfg.min_driver_height * self.h]

    def _side_score(self, p):
        x = _center(p["box"])[0] / self.w
        return x if self.cfg.driver_side == "right" else 1 - x

    def pick_drivers(self, frames: list[dict]) -> list[dict | None]:
        """First pass: driver = person furthest to the driver's side. Second pass: per shot,
        the person closest to where the driver usually sits (handles people leaning over)."""
        first = []
        for f in frames:
            cands = self._candidates(f["people"])
            first.append(max(cands, key=self._side_score) if cands else None)

        seat = defaultdict(list)
        for f, d in zip(frames, first, strict=True):
            if d is not None:
                seat[f["shot"]].append(_center(d["box"]))
        seat = {s: np.median(np.array(v), axis=0) for s, v in seat.items()}

        out = []
        for f, d in zip(frames, first, strict=True):
            cands = self._candidates(f["people"])
            if not cands or f["shot"] not in seat:
                out.append(d)
                continue
            out.append(min(cands, key=lambda p: np.linalg.norm(_center(p["box"]) - seat[f["shot"]])))
        return out

    # 2. fixtures ----------------------------------------------------------------------
    def fixture_mask(self, frames: list[dict], drivers: list[dict | None] | None = None) -> list[list[bool]]:
        """Marks detections that sit in the same place for much of a shot and are rarely in the
        driver's hand (a phone mounted on the windscreen, a cup in the holder)."""
        if drivers is None:
            drivers = self.pick_drivers(frames)
        by_shot = defaultdict(list)
        for i, f in enumerate(frames):
            by_shot[f["shot"]].append(i)
        mask = [[False] * len(f["objects"]) for f in frames]
        for idxs in by_shot.values():
            if len(idxs) < 10:
                continue
            dets = [(i, j, o) for i in idxs for j, o in enumerate(frames[i]["objects"])]
            for i, j, o in dets:
                same = [(i2, o2) for i2, _, o2 in dets
                        if o2["cls"] == o["cls"] and _iou(o["box"], o2["box"]) >= self.cfg.static_iou]
                span = frames[max(i2 for i2, _ in same)]["t"] - frames[min(i2 for i2, _ in same)]["t"]
                if len({i2 for i2, _ in same}) < self.cfg.static_fraction * len(idxs) \
                        and span < self.cfg.static_span_s:
                    continue
                handled = sum(self._in_hand(drivers[i2], o2) for i2, o2 in same)
                if handled < 0.3 * len(same):
                    mask[i][j] = True
        return mask

    def _in_hand(self, driver: dict | None, o: dict) -> bool:
        if driver is None or driver["kpts"] is None:
            return False
        f = self.features(driver)
        c = _center(o["box"])
        return any(np.linalg.norm(c - w) / f["scale"] < self.cfg.hand_radius * 1.3 for w in f["wrists"])

    # 3. pose features -------------------------------------------------------------------
    def features(self, driver: dict, scale: float | None = None) -> dict | None:
        k, c = driver["kpts"], self.cfg.kpt_conf
        ls, rs = _pt(k, L_SHOULDER, c), _pt(k, R_SHOULDER, c)
        box = driver["box"]
        if scale is None:
            scale = 0.2 * (box[3] - box[1])
        if ls is not None and rs is not None:
            torso = (ls + rs) / 2
        else:
            torso = _mean([ls, rs, _pt(k, L_HIP, c), _pt(k, R_HIP, c)])
        head = _mean([_pt(k, i, c) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)])
        wrists = [w for w in (_pt(k, L_WRIST, c), _pt(k, R_WRIST, c)) if w is not None]
        shoulders = float(np.linalg.norm(ls - rs)) if ls is not None and rs is not None else None
        # which way the face points, seen from the side: nose to the left or right of the visible ear.
        # When the driver turns their head towards the camera or the back, the sign flips.
        nose = _pt(k, NOSE, c)
        ear_i = L_EAR if k[L_EAR][2] >= k[R_EAR][2] else R_EAR
        ear = _pt(k, ear_i, c)
        yaw = float((nose[0] - ear[0]) / scale) if nose is not None and ear is not None else None
        return {"scale": float(scale), "torso": torso, "head": head, "wrists": wrists, "box": box,
                "shoulders": shoulders, "yaw": yaw}

    def _owned_by_driver(self, o: dict, driver: dict, others: list[dict], u: float) -> bool:
        """False when the object is really someone else's (a passenger's phone that overlaps the
        driver in the image): another person's hand is closer to it, or it sits inside their body."""
        c = self.cfg.kpt_conf
        mine = [w for w in (_pt(driver["kpts"], L_WRIST, c), _pt(driver["kpts"], R_WRIST, c)) if w is not None]
        d_mine = min((_box_dist(w, o["box"]) for w in mine), default=np.inf) / u
        for p in others:
            theirs = [w for w in (_pt(p["kpts"], L_WRIST, c), _pt(p["kpts"], R_WRIST, c)) if w is not None] \
                if p["kpts"] is not None else []
            d_theirs = min((_box_dist(w, o["box"]) for w in theirs), default=np.inf) / u
            if d_theirs < d_mine and d_theirs < 1.5:
                return False
            if _inside(_center(o["box"]), p["box"]) and d_mine > 1.5:
                return False
        return True

    # 4 + 5. labelling -------------------------------------------------------------------
    def analyse(self, frames: list[dict]) -> tuple[list[FrameState], list[Event]]:
        drivers = self.pick_drivers(frames)
        fixtures = self.fixture_mask(frames, drivers)
        self.fixtures = fixtures
        cfg = self.cfg

        # the driver's usual size and seat position in each shot
        heights, torsos, widths, yaws = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for f, d in zip(frames, drivers, strict=True):
            if d is not None:
                heights[f["shot"]].append(d["box"][3] - d["box"][1])
        unit = {shot: 0.2 * float(np.median(h)) for shot, h in heights.items()}

        states: list[FrameState] = []
        for f, d, fx in zip(frames, drivers, fixtures, strict=True):
            st = FrameState(t=f["t"], shot=f["shot"], driver=d)
            if d is not None:
                st.features = self.features(d, unit[f["shot"]])
                if st.features["torso"] is not None:
                    torsos[f["shot"]].append(st.features["torso"])
                others = [p for p in f["people"] if p is not d]
                st.objects = [o for o, is_fixture in zip(f["objects"], fx, strict=True)
                              if not is_fixture and _inside(_center(o["box"]), d["box"], margin=0.15)
                              and self._owned_by_driver(o, d, others, unit[f["shot"]])]
                if st.features["shoulders"] is not None:
                    widths[f["shot"]].append(st.features["shoulders"] / unit[f["shot"]])
                if st.features["yaw"] is not None:
                    yaws[f["shot"]].append(st.features["yaw"])
            states.append(st)
        seat = {shot: np.median(t, axis=0) for shot, t in torsos.items()}
        usual_width = {shot: float(np.median(w)) for shot, w in widths.items()}
        usual_yaw = {shot: float(np.median(y)) for shot, y in yaws.items()}
        passenger_dir = -1.0 if cfg.driver_side == "right" else 1.0  # passenger side is the other side

        sightings = defaultdict(lambda: {"phone": [], "drink": []})  # shot -> times seen in the driver's hand
        hidden: dict = {}  # shot -> (kind, last time) while a hand keeps holding a hidden object at the head
        for st in states:
            f = st.features
            if not f:
                continue
            u, head, wrists = f["scale"], f["head"], f["wrists"]

            def near_head(p, head=head, u=u):
                return head is not None and np.linalg.norm(p - head) / u < cfg.head_radius

            def in_hand(p, wrists=wrists, u=u):
                return any(np.linalg.norm(p - w) / u < cfg.hand_radius for w in wrists)

            def held(o, wrists=wrists, u=u):
                return in_hand(_center(o["box"])) or any(_box_dist(w, o["box"]) / u < cfg.grip for w in wrists)

            # hand on top of / behind the head: wrist above shoulder level, next to the head
            sh_y = f["torso"][1] if f["torso"] is not None else None
            hand_on_head = head is not None and sh_y is not None and any(
                w[1] < sh_y and np.linalg.norm(w - head) / u < cfg.hand_on_head for w in wrists)

            st.objects = [o for o in st.objects if o["cls"] != "phone"
                          or max(o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]) / u <= cfg.max_phone_size]
            phones = [o for o in st.objects if o["cls"] == "phone"]
            # a drink only counts once it's picked up: a bottle in the door pocket is not drinking
            drinks = [o for o in st.objects if o["cls"] in ("bottle", "cup")
                      and (held(o) or near_head(_center(o["box"])))]
            wrist_at_head = any(near_head(w) for w in wrists)
            # the pose model sometimes puts both arms on the one it can see (the other hand is hidden
            # behind the head holding the phone); then it can't tell us where the hands are
            k = st.driver["kpts"]
            arms = [_pt(k, i, cfg.kpt_conf) for i in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)]
            collapsed = all(a is not None for a in arms) \
                and np.linalg.norm(arms[0] - arms[1]) / u < 0.6 and np.linalg.norm(arms[2] - arms[3]) / u < 0.6
            # a phone held to the ear: the wrist is below the ear, not on top of the head
            ear_y = _mean([_pt(st.driver["kpts"], i, cfg.kpt_conf) for i in (L_EAR, R_EAR)])
            ear_y = ear_y[1] if ear_y is not None else (head[1] if head is not None else None)
            holding_to_ear = ear_y is not None and any(
                near_head(w) and (w[1] - ear_y) / u >= cfg.ear_hold for w in wrists)
            seen = sightings[st.shot]
            if any((held(p) or near_head(_center(p["box"]))) and p["conf"] >= cfg.memory_conf for p in phones):
                seen["phone"].append(st.t)
            if any(o["conf"] >= cfg.memory_conf for o in drinks):
                seen["drink"].append(st.t)

            if any(near_head(_center(p["box"])) for p in phones):
                st.flags["phone_call"] = "phone at the head"
                if any(p["conf"] >= cfg.memory_conf for p in phones):
                    hidden[st.shot] = ("phone", st.t)  # if the hand now covers it, the call goes on
            elif phones:
                in_use = any(held(p) for p in phones)
                st.flags["texting"] = "phone in hand" if in_use else "phone out, near the driver"
            if drinks:
                at_mouth = any(near_head(_center(o["box"])) for o in drinks)
                st.flags["drinking"] = "bottle/cup at the mouth" if at_mouth else "holding a drink"

            if (wrist_at_head or hand_on_head) and not phones and not drinks:
                recent = {k: ts[-1] for k, ts in seen.items()
                          if sum(st.t - x <= cfg.memory_s for x in ts) >= cfg.memory_min_sightings}
                if not holding_to_ear:  # a hidden call only starts with the hand held to the ear
                    recent.pop("phone", None)
                ongoing = hidden.get(st.shot)
                if ongoing and st.t - ongoing[1] <= cfg.hold_grace_s:
                    recent = {ongoing[0]: st.t}  # the hand never left the ear
                if recent:  # the object is probably hidden by the hand
                    kind = max(recent, key=recent.get)
                    hidden[st.shot] = (kind, st.t)
                    if kind == "phone":
                        st.flags["phone_call"] = "hand at the ear, phone hidden"
                    else:
                        st.flags["drinking"] = "hand at the mouth, drink hidden"
                else:
                    st.flags["hand_to_face"] = "hand at the face" if wrist_at_head else "hand on the head"
            elif collapsed and not phones and not drinks:
                ongoing = hidden.get(st.shot)
                if ongoing and ongoing[0] == "phone" and st.t - ongoing[1] <= cfg.hold_grace_s:
                    hidden[st.shot] = ("phone", st.t)
                    st.flags["phone_call"] = "hand at the ear, phone hidden"

            base = usual_yaw.get(st.shot)
            if f["yaw"] is not None and base is not None and abs(base) > cfg.head_turn \
                    and np.sign(f["yaw"]) != np.sign(base) and abs(f["yaw"]) > cfg.head_turn:
                st.flags["looking_away"] = "head turned away from the road"

            if f["torso"] is not None and st.shot in seat:
                shift = float(np.linalg.norm(f["torso"] - seat[st.shot]) / u)
                side = max((passenger_dir * (w[0] - f["torso"][0]) / u for w in wrists), default=0.0)
                f["shift"], f["side_reach"] = shift, side
                wider = (f["shoulders"] / u - usual_width[st.shot]) \
                    if f["shoulders"] is not None and st.shot in usual_width else 0.0
                f["turn"] = wider
                if side > cfg.reach_side and not phones and not drinks:
                    st.flags["reaching"] = "arm stretched towards the passenger side"
                elif wider > cfg.turn_shoulders:
                    st.flags["reaching"] = "body turned towards the back"
                elif shift > cfg.reach_shift:
                    st.flags["reaching"] = f"body moved {shift:.1f} body units"

        self._fill_between_sightings(states, "drinking", cfg.hold_gap_s)
        return states, self.to_events(states)

    @staticmethod
    def _fill_between_sightings(states: list[FrameState], behaviour: str, max_gap: float):
        """A bottle held to the mouth is mostly hidden by the hand, so the detector only sees it
        every second or so. Between two sightings close together the driver is still drinking."""
        seen = [i for i, st in enumerate(states) if behaviour in st.flags and "hidden" not in st.flags[behaviour]]
        for a, b in zip(seen, seen[1:], strict=False):
            if states[a].shot != states[b].shot or states[b].t - states[a].t > max_gap:
                continue
            for st in states[a + 1:b]:
                if st.driver is not None and "phone_call" not in st.flags and "texting" not in st.flags:
                    st.flags.setdefault(behaviour, states[a].flags[behaviour])

    # smoothing --------------------------------------------------------------------------
    def to_events(self, states: list[FrameState]) -> list[Event]:
        cfg = self.cfg
        if not states:
            return []
        ts = np.array([s.t for s in states])
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.2
        events: list[Event] = []
        for b in BEHAVIOURS:
            glance = b == "looking_away"
            half = 1 if glance else max(1, int(round(cfg.window_s / dt / 2)))
            min_len = cfg.glance_min_s if glance else cfg.min_event_s
            raw = np.array([b in s.flags for s in states], dtype=float)
            has_driver = np.array([s.driver is not None for s in states])
            smooth = np.array([raw[max(0, i - half): i + half + 1].mean() for i in range(len(raw))])
            on = (smooth >= 0.5) & has_driver
            # split into runs, never across a shot boundary
            runs, start = [], None
            for i in range(len(on) + 1):
                cut = i < len(on) and start is not None and states[i].shot != states[start].shot
                if i < len(on) and on[i] and not cut:
                    start = i if start is None else start
                    continue
                if start is not None:
                    runs.append((start, i - 1))
                    start = i if (i < len(on) and on[i]) else None
            # merge close runs, drop short ones
            merged = []
            gap = {"texting": 3.0, "looking_away": 1.0}.get(b, cfg.merge_gap_s)  # drinking gaps are filled per frame
            for r in runs:
                if merged and ts[r[0]] - ts[merged[-1][1]] <= gap and \
                        states[r[0]].shot == states[merged[-1][1]].shot:
                    merged[-1] = (merged[-1][0], r[1])
                else:
                    merged.append(r)
            for a, z in merged:
                start_t, end_t = float(ts[a]), float(ts[z] + dt)
                if end_t - start_t < min_len:
                    continue
                seg = raw[a:z + 1]
                peak = a + int(np.argmax(smooth[a:z + 1]))
                ev = next((s.flags[b] for s in states[a:z + 1] if b in s.flags), "")
                events.append(Event(b, round(start_t, 2), round(end_t, 2), driver=0,
                                    peak_t=round(float(ts[peak]), 2), confidence=round(float(seg.mean()), 2),
                                    evidence=ev))
        return self.resolve_overlaps(sorted(events, key=lambda e: e.start))

    @staticmethod
    def resolve_overlaps(events: list[Event]) -> list[Event]:
        """A phone call often also looks like 'hand at face'; keep the more specific label."""
        keep = []
        for e in events:
            dominated = any(
                o is not e and PRIORITY[o.behaviour] < PRIORITY[e.behaviour]
                and min(o.end, e.end) - max(o.start, e.start) > 0.6 * e.duration
                for o in events
            )
            if not dominated:
                keep.append(e)
        return keep
