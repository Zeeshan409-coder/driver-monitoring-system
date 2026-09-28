"""Builders for fake model output, so the logic can be tested without running YOLO."""

W, H = 1280, 720


def person(cx=900, top=150, height=500, shoulder=160, pid=1, wrists=None, head=True, conf=0.9):
    """A seated person facing the camera. Coordinates follow COCO keypoint order."""
    k = [[0.0, 0.0, 0.0] for _ in range(17)]
    sy = top + 0.3 * height
    hx, hy = cx, top + 0.12 * height
    if head:
        k[0] = [hx, hy, conf]
        k[1] = [hx + 15, hy - 10, conf]
        k[2] = [hx - 15, hy - 10, conf]
        k[3] = [hx + 35, hy, conf]
        k[4] = [hx - 35, hy, conf]
    k[5] = [cx + shoulder / 2, sy, conf]
    k[6] = [cx - shoulder / 2, sy, conf]
    k[7] = [cx + shoulder / 2 + 10, sy + 110, conf]
    k[8] = [cx - shoulder / 2 - 10, sy + 110, conf]
    lw, rw = wrists or ((cx + 60, sy + 200), (cx - 60, sy + 200))
    k[9] = [lw[0], lw[1], conf]
    k[10] = [rw[0], rw[1], conf]
    k[11] = [cx + 60, top + 0.8 * height, conf]
    k[12] = [cx - 60, top + 0.8 * height, conf]
    box = [cx - 170, top, cx + 170, top + height]
    return {"id": pid, "box": box, "conf": 0.9, "kpts": k}


def obj(cls, cx, cy, size=50):
    return {"cls": cls, "box": [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], "conf": 0.8}


def frames(n, fps=5, shot=0, t0=0.0, make=None):
    """n frames; make(i) -> (people, objects)."""
    out = []
    for i in range(n):
        people, objects = make(i) if make else ([person()], [])
        out.append({"t": round(t0 + i / fps, 3), "shot": shot, "people": people, "objects": objects})
    return out


def head_of(p):
    return p["kpts"][0][0], p["kpts"][0][1]


def lap_of(p):
    return p["kpts"][9][0], p["kpts"][9][1]
