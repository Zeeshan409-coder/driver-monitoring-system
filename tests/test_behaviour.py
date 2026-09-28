from dms.behaviour import BehaviourAnalyzer, Config, Event
from helpers import H, W, frames, head_of, obj, person


def analyse(fs, **cfg):
    return BehaviourAnalyzer(W, H, Config(**cfg)).analyse(fs)


def behaviours(events):
    return [e.behaviour for e in events]


def test_calm_driver_has_no_events():
    _, events = analyse(frames(100))
    assert events == []


def test_phone_at_head_is_a_call():
    def make(i):
        p = person()
        hx, hy = head_of(p)
        if 20 <= i < 60:
            p["kpts"][9][:2] = [hx + 50, hy + 10]
            return [p], [obj("phone", hx + 55, hy)]
        return [p], []

    _, events = analyse(frames(100, make=make))
    assert behaviours(events) == ["phone_call"]
    e = events[0]
    assert 3.5 <= e.start <= 4.5 and 11.5 <= e.end <= 12.5
    assert e.evidence == "phone at the head"


def test_phone_in_lap_is_texting():
    def make(i):
        p = person()
        lx, ly = p["kpts"][9][:2]
        return ([p], [obj("phone", lx - 20, ly - 10)]) if 10 <= i < 50 else ([p], [])

    _, events = analyse(frames(80, make=make))
    assert behaviours(events) == ["texting"]


def test_call_is_not_also_reported_as_hand_at_face():
    def make(i):
        p = person()
        hx, hy = head_of(p)
        p["kpts"][9][:2] = [hx + 40, hy]
        return [p], ([obj("phone", hx + 45, hy)] if i < 40 else [])

    _, events = analyse(frames(80, make=make))
    assert "phone_call" in behaviours(events)
    call = next(e for e in events if e.behaviour == "phone_call")
    assert all(not (e.behaviour == "hand_to_face" and e.start < call.end) for e in events)


def test_drinking():
    def make(i):
        p = person()
        hx, hy = head_of(p)
        if 20 <= i < 45:
            p["kpts"][10][:2] = [hx - 20, hy + 40]
            return [p], [obj("bottle", hx - 15, hy + 30, 60)]
        return [p], []

    _, events = analyse(frames(80, make=make))
    assert behaviours(events) == ["drinking"]
    assert events[0].evidence == "bottle/cup at the mouth"


def test_hand_at_face_without_object():
    def make(i):
        p = person()
        if 30 <= i < 60:
            hx, hy = head_of(p)
            p["kpts"][9][:2] = [hx + 20, hy + 30]
        return [p], []

    _, events = analyse(frames(100, make=make))
    assert behaviours(events) == ["hand_to_face"]


def test_call_continues_when_the_hand_hides_the_phone():
    # phone seen in hand, then held at the ear where the detector loses it
    def make(i):
        p = person()
        hx, hy = head_of(p)
        if i < 20:
            lx, ly = p["kpts"][9][:2]
            return [p], [obj("phone", lx, ly)]
        if i < 60:
            p["kpts"][9][:2] = [hx + 40, hy + 60]
        return [p], []

    _, events = analyse(frames(90, make=make))
    assert behaviours(events) == ["texting", "phone_call"]
    assert events[1].evidence == "hand at the ear, phone hidden"


def test_hand_at_face_long_after_phone_is_not_a_call():
    def make(i):
        p = person()
        if i < 10:
            lx, ly = p["kpts"][9][:2]
            return [p], [obj("phone", lx, ly)]
        if 70 <= i < 100:
            hx, hy = head_of(p)
            p["kpts"][9][:2] = [hx + 20, hy + 30]
        return [p], []

    _, events = analyse(frames(120, make=make))
    assert behaviours(events) == ["texting", "hand_to_face"]


def test_drink_hidden_at_the_mouth_is_still_drinking():
    def make(i):
        p = person()
        hx, hy = head_of(p)
        p["kpts"][10][:2] = [hx - 20, hy + 40]
        return ([p], [obj("bottle", hx - 15, hy + 30, 60)]) if i < 15 or 25 <= i < 30 else ([p], [])

    _, events = analyse(frames(40, make=make))
    assert behaviours(events) == ["drinking"]
    assert events[0].end >= 7.5


def test_drinking_with_the_bottle_seen_only_now_and_then():
    # the hand covers the bottle: the detector catches it about once a second
    def make(i):
        p = person()
        hx, hy = head_of(p)
        if not 10 <= i < 50:
            return [p], []
        p["kpts"][10][:2] = [hx - 20, hy + 40]
        return ([p], [obj("bottle", hx - 15, hy + 30, 60)]) if i % 5 == 0 else ([p], [])

    _, events = analyse(frames(70, make=make))
    assert behaviours(events) == ["drinking"]
    assert events[0].start <= 2.2 and events[0].end >= 9.0


def test_bottle_that_is_not_picked_up_is_ignored():
    # e.g. the gear stick, detected now and then as a "bottle"
    def make(i):
        p = person()
        x2, y2 = p["box"][2:]
        return ([p], [obj("bottle", x2 - 20, y2 - 30)]) if i % 3 else ([p], [])

    _, events = analyse(frames(60, make=make))
    assert events == []


def test_huge_phone_detection_is_rejected():
    def make(i):
        p = person()
        hx, hy = head_of(p)
        return [p], [obj("phone", hx, hy + 60, 300)] if 10 <= i < 40 else []

    _, events = analyse(frames(60, make=make))
    assert "phone_call" not in behaviours(events) and "texting" not in behaviours(events)


def test_arm_stretched_to_the_back_seat_is_reaching():
    def make(i):
        p = person()
        if 30 <= i < 50:
            p["kpts"][10][:2] = [p["box"][0] - 80, p["kpts"][10][1] - 50]  # far towards the passenger side
        return [p], []

    _, events = analyse(frames(80, make=make))
    assert behaviours(events) == ["reaching"]
    assert events[0].evidence == "arm stretched towards the passenger side"


def test_head_turned_away_for_a_second_is_looking_away():
    # side view: the nose is normally on one side of the visible ear; turning the head flips it
    def make(i):
        p = person()
        if 40 <= i < 47:
            hx = p["kpts"][3][0]
            p["kpts"][0][0] = hx + 40
        return [p], []

    _, events = analyse(frames(80, make=make))
    assert behaviours(events) == ["looking_away"]
    assert 7.5 <= events[0].start <= 8.3 and events[0].duration >= 0.8


def test_a_single_frame_head_flick_is_ignored():
    def make(i):
        p = person()
        if i in (20, 50):
            p["kpts"][0][0] = p["kpts"][3][0] + 40
        return [p], []

    _, events = analyse(frames(80, make=make))
    assert events == []


def test_leaning_away_is_reaching():
    def make(i):
        return [person(cx=700 if 50 <= i < 70 else 900)], []

    _, events = analyse(frames(120, make=make))
    assert behaviours(events) == ["reaching"]
    assert events[0].evidence.startswith("body moved")


def test_single_frame_flicker_is_ignored():
    def make(i):
        p = person()
        hx, hy = head_of(p)
        return ([p], [obj("phone", hx + 50, hy)]) if i in (10, 30, 31, 55) else ([p], [])

    _, events = analyse(frames(80, make=make))
    assert events == []


def test_short_gaps_are_merged():
    def make(i):
        p = person()
        lx, ly = p["kpts"][9][:2]
        on = 10 <= i < 30 or 36 <= i < 60  # 1.2 s gap
        return ([p], [obj("phone", lx, ly)]) if on else ([p], [])

    _, events = analyse(frames(80, make=make))
    assert len(events) == 1 and events[0].behaviour == "texting"


def test_events_never_cross_a_shot_cut():
    def make(i):
        p = person()
        lx, ly = p["kpts"][9][:2]
        return [p], [obj("phone", lx, ly)] if 20 <= i < 60 else []

    fs = frames(80, make=make)
    for f in fs[40:]:
        f["shot"] = 1
    _, events = analyse(fs)
    assert len(events) == 2
    cut = fs[40]["t"]
    assert events[0].end <= cut + 1e-6 and events[1].start >= cut - 1e-6


def test_mounted_phone_is_ignored():
    # a phone on the windscreen mount: same spot all shot, inside the driver's area, never in hand
    def make(i):
        p = person()
        x1, y1 = p["box"][:2]
        return [p], [obj("phone", x1 + 30, y1 + 40)]

    fs = frames(60, make=make)
    analyzer = BehaviourAnalyzer(W, H)
    assert all(all(m) for m in analyzer.fixture_mask(fs))
    _, events = analyzer.analyse(fs)
    assert events == []


def test_phone_held_still_all_shot_is_still_phone_use():
    def make(i):
        p = person()
        lx, ly = p["kpts"][9][:2]
        return [p], [obj("phone", lx, ly - 10)]

    _, events = analyse(frames(60, make=make))
    assert behaviours(events) == ["texting"]


def test_passengers_phone_overlapping_the_driver_is_not_phone_use():
    # the passenger holds a phone in the foreground; in the image it lands inside the driver's box
    driver = person(cx=900, pid=1)
    passenger = person(cx=560, pid=2, wrists=((760, 560), (500, 560)))
    fs = frames(60, make=lambda i: ([passenger, driver], [obj("phone", 765, 565)] if i % 2 else []))
    _, events = analyse(fs)
    assert events == []


def test_hidden_call_lasts_while_the_hand_stays_at_the_ear():
    def make(i):
        p = person()
        hx, hy = head_of(p)
        if i < 10:  # phone clearly seen at the head
            p["kpts"][9][:2] = [hx + 40, hy + 60]
            return [p], [obj("phone", hx + 45, hy + 5)]
        if i < 100:  # 18 s with the hand at the ear, phone covered
            p["kpts"][9][:2] = [hx + 40, hy + 60]
        return [p], []

    _, events = analyse(frames(120, make=make))
    assert behaviours(events) == ["phone_call"]
    assert events[0].end >= 19


def test_call_survives_the_pose_model_putting_both_arms_on_the_wheel():
    # the hand holding the phone is hidden behind the head; the pose model then puts both
    # arms on the one it can see. That must not end the call.
    def make(i):
        p = person()
        hx, hy = head_of(p)
        if i < 10:
            p["kpts"][9][:2] = [hx + 40, hy + 60]
            return [p], [obj("phone", hx + 45, hy + 20)]
        if i < 40:
            k = p["kpts"]
            k[7][:2] = [hx + 150, hy + 150]
            k[8][:2] = [hx + 160, hy + 155]
            k[9][:2] = [hx + 260, hy + 150]
            k[10][:2] = [hx + 265, hy + 152]
        return [p], []

    _, events = analyse(frames(60, make=make))
    assert behaviours(events) == ["phone_call"]
    assert events[0].end >= 7.8


def test_arms_overlapping_without_a_call_is_nothing():
    def make(i):
        p = person()
        k = p["kpts"]
        k[8][:2] = k[7][:2]
        k[10][:2] = k[9][:2]
        return [p], []

    _, events = analyse(frames(40, make=make))
    assert events == []


def test_body_turned_to_the_back_seat_is_reaching():
    # turning round makes the shoulders look much wider to a side camera
    def make(i):
        return [person(shoulder=260 if 40 <= i < 60 else 110)], []

    _, events = analyse(frames(100, make=make))
    assert behaviours(events) == ["reaching"]
    assert events[0].evidence == "body turned towards the back"


def test_hand_on_top_of_the_head_after_holding_a_phone_is_not_a_call():
    # phone seen in hand, then the hand goes up to the top of the head: that's not a call
    def make(i):
        p = person()
        hx, hy = head_of(p)
        if i < 15:
            lx, ly = p["kpts"][9][:2]
            return [p], [obj("phone", lx, ly)]
        if 20 <= i < 40:
            p["kpts"][9][:2] = [hx + 20, hy - 10]
        return [p], []

    _, events = analyse(frames(60, make=make))
    assert behaviours(events) == ["texting", "hand_to_face"]


def test_hand_on_top_of_the_head():
    def make(i):
        p = person()
        if 30 <= i < 60:
            hx, hy = head_of(p)
            p["kpts"][10][:2] = [hx - 30, hy - 110]
        return [p], []

    _, events = analyse(frames(90, make=make))
    assert behaviours(events) == ["hand_to_face"]


def test_tall_bottle_held_by_its_neck_counts():
    def make(i):
        p = person()
        lx, ly = p["kpts"][9][:2]
        bottle = {"cls": "bottle", "box": [lx - 25, ly - 20, lx + 25, ly + 230], "conf": 0.6}
        return [p], [bottle] if 10 <= i < 40 else []

    _, events = analyse(frames(60, make=make))
    assert behaviours(events) == ["drinking"]


def test_passenger_is_not_the_driver():
    driver = person(cx=950, pid=1)
    passenger = person(cx=300, pid=2)
    hx, hy = head_of(passenger)
    fs = frames(60, make=lambda i: ([passenger, driver], [obj("phone", hx + 40, hy)]))
    states, events = analyse(fs)
    assert all(s.driver["id"] == 1 for s in states)
    assert events == []


def test_left_hand_drive_setting():
    driver = person(cx=300, pid=1)
    other = person(cx=950, pid=2)
    fs = frames(10, make=lambda i: ([driver, other], []))
    states, _ = analyse(fs, driver_side="left")
    assert all(s.driver["id"] == 1 for s in states)


def test_small_background_people_are_ignored():
    driver = person(pid=1)
    far = person(cx=1200, top=100, height=100, pid=9)
    states, _ = analyse(frames(10, make=lambda i: ([driver, far], [])))
    assert all(s.driver["id"] == 1 for s in states)


def test_no_people_means_no_driver():
    states, events = analyse(frames(20, make=lambda i: ([], [])))
    assert all(s.driver is None for s in states)
    assert events == []


def test_resolve_overlaps_keeps_the_specific_label():
    call = Event("phone_call", 10, 20, 1, 15, 0.9)
    face = Event("hand_to_face", 11, 19, 1, 15, 0.9)
    later = Event("hand_to_face", 30, 35, 1, 32, 0.9)
    kept = BehaviourAnalyzer.resolve_overlaps([call, face, later])
    assert kept == [call, later]


def test_event_dict_has_label_and_duration():
    d = Event("texting", 1.0, 4.5, 2, 2.0, 0.8, "phone in hand").to_dict()
    assert d["duration"] == 3.5 and d["label"] == "Using phone" and d["driver"] == 2
