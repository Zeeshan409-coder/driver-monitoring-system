import numpy as np

from dms.identity import assign_drivers, upper_body_signature
from helpers import person


def frame_with_jacket(bgr):
    img = np.full((720, 1280, 3), 40, np.uint8)
    img[250:600, 700:1100] = bgr
    return img


def test_signature_is_normalised():
    sig = upper_body_signature(frame_with_jacket((40, 60, 200)), person())
    assert sig is not None and abs(sum(sig) - 1) < 1e-3


def test_tiny_crop_has_no_signature():
    p = person()
    p["kpts"] = None
    p["box"] = [10, 10, 15, 15]
    assert upper_body_signature(np.zeros((720, 1280, 3), np.uint8), p) is None


def test_same_jacket_same_driver_different_jacket_new_driver():
    p = person()
    red = upper_body_signature(frame_with_jacket((40, 40, 200)), p)
    beige = upper_body_signature(frame_with_jacket((150, 190, 210)), p)
    blue = upper_body_signature(frame_with_jacket((200, 90, 30)), p)
    shots = [0, 0, 1, 1, 2, 2, 3, 3]
    sigs = [red, red, red, red, beige, beige, blue, blue]
    assert assign_drivers(shots, sigs) == {0: 1, 1: 1, 2: 2, 3: 3}


def test_returning_driver_gets_their_old_number():
    p = person()
    red = upper_body_signature(frame_with_jacket((40, 40, 200)), p)
    blue = upper_body_signature(frame_with_jacket((200, 90, 30)), p)
    assert assign_drivers([0, 1, 2], [red, blue, red]) == {0: 1, 1: 2, 2: 1}


def test_shot_without_signature_inherits_previous_driver():
    p = person()
    red = upper_body_signature(frame_with_jacket((40, 40, 200)), p)
    assert assign_drivers([0, 1], [red, None]) == {0: 1, 1: 1}
