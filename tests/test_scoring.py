import pytest

from dms.behaviour import Event
from dms.scoring import WEIGHTS, grade, scorecard, union_seconds


def test_union_merges_overlaps():
    assert union_seconds([(0, 10), (5, 15), (20, 25)]) == 20
    assert union_seconds([]) == 0


@pytest.mark.parametrize("score,g", [(100, "A"), (90, "A"), (89, "B"), (75, "B"),
                                     (60, "C"), (40, "D"), (39, "E"), (0, "E")])
def test_grades(score, g):
    assert grade(score) == g


def test_clean_driver_scores_100():
    card = scorecard(1, 120, [])
    assert card["score"] == 100 and card["grade"] == "A" and card["distracted_pct"] == 0


def test_points_and_per_driver_split():
    events = [Event("texting", 0, 10, 1, 5, 1), Event("drinking", 20, 25, 1, 22, 1), Event("texting", 0, 60, 2, 5, 1)]
    one = scorecard(1, 100, events)
    texting, drinking = WEIGHTS["texting"], WEIGHTS["drinking"]
    expected = 100 - (texting[0] + 10 * texting[1] + drinking[0] + 5 * drinking[1])
    assert one["score"] == round(expected)
    assert one["events"] == 2
    assert one["distracted_s"] == 15 and one["distracted_pct"] == 15
    assert one["by_behaviour"]["texting"] == {"events": 1, "seconds": 10}


def test_score_floors_at_zero():
    events = [Event("texting", i * 10, i * 10 + 9, 1, 0, 1) for i in range(20)]
    card = scorecard(1, 200, events)
    assert card["score"] == 0 and card["grade"] == "E"


def test_phone_use_costs_more_than_touching_the_face():
    a = scorecard(1, 60, [Event("texting", 0, 5, 1, 0, 1)])
    b = scorecard(1, 60, [Event("hand_to_face", 0, 5, 1, 0, 1)])
    assert a["score"] < b["score"]
