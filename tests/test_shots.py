import numpy as np

from dms.shots import ShotDetector


def test_cut_is_detected_and_steady_scene_is_not():
    det = ShotDetector()
    rng = np.random.default_rng(0)
    a = np.zeros((360, 640, 3), np.uint8)
    a[:, :320] = (30, 60, 120)
    b = np.zeros((360, 640, 3), np.uint8)
    b[:] = (200, 180, 40)
    shots = []
    for i in range(30):
        frame = a if i < 15 else b
        noisy = np.clip(frame.astype(int) + rng.integers(-5, 6, frame.shape), 0, 255).astype(np.uint8)
        shots.append(det.update(noisy, i / 5)[0])
    assert shots[:15] == [0] * 15
    assert shots[15:] == [1] * 15
