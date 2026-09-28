"""Stage 1: what the models see in each frame.

Two models run on every analysed frame:
  * YOLO11-pose for people and their 17 body keypoints (nose, eyes, ears,
    shoulders, elbows, wrists, ...),
  * YOLO11 object detection for the things a driver shouldn't be holding:
    phones, bottles, cups.

The output is plain data (lists and dicts), so it can be cached to JSON and the
behaviour logic can be re-run or tuned without touching the models again.
"""

from __future__ import annotations

import numpy as np

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP = 11, 12

OBJECT_CLASSES = {67: "phone", 39: "bottle", 41: "cup"}  # COCO ids we care about


class Perceiver:
    def __init__(self, pose_model: str = "yolo11n-pose.pt", object_model: str = "yolo11s.pt",
                 imgsz: int = 960, device: str | None = None):
        from ultralytics import YOLO

        self.pose = YOLO(pose_model)
        self.objects = YOLO(object_model)
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame: np.ndarray) -> dict:
        pose = self.pose.track(frame, persist=True, imgsz=self.imgsz, conf=0.3, verbose=False,
                               tracker="bytetrack.yaml", device=self.device)[0]
        people = []
        if pose.boxes is not None and len(pose.boxes):
            ids = pose.boxes.id.int().tolist() if pose.boxes.id is not None else [-1] * len(pose.boxes)
            kpts = pose.keypoints.data.cpu().numpy() if pose.keypoints is not None else None
            for i, (box, conf) in enumerate(zip(pose.boxes.xyxy.tolist(), pose.boxes.conf.tolist(), strict=True)):
                people.append({
                    "id": ids[i],
                    "box": [round(v, 1) for v in box],
                    "conf": round(conf, 3),
                    "kpts": np.round(kpts[i], 2).tolist() if kpts is not None else None,
                })

        det = self.objects(frame, imgsz=self.imgsz, conf=0.2, classes=list(OBJECT_CLASSES),
                           verbose=False, device=self.device)[0]
        objects = [
            {"cls": OBJECT_CLASSES[int(c)], "box": [round(v, 1) for v in b], "conf": round(s, 3)}
            for b, c, s in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist(), det.boxes.conf.tolist(), strict=True)
        ]
        return {"people": people, "objects": objects}
