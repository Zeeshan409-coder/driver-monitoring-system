from evaluate import evaluate, match

GT = {
    "drivers": [{"driver": 1, "start": 0, "end": 100}],
    "events": [
        {"behaviour": "texting", "start": 10, "end": 20},
        {"behaviour": "drinking", "start": 50, "end": 60},
    ],
}


def test_match_needs_real_overlap():
    assert match({"start": 12, "end": 22}, {"start": 10, "end": 20})
    assert not match({"start": 19, "end": 40}, {"start": 10, "end": 20})


def test_perfect_prediction():
    rows, overall, ev = evaluate(GT["events"], GT)
    assert ev["event_recall"] == 1 and ev["event_precision"] == 1
    assert overall["accuracy"] == 1
    assert rows["texting"]["time_recall"] == 1


def test_wrong_label_and_missed_event():
    pred = [{"behaviour": "phone_call", "start": 10, "end": 20}]
    rows, overall, ev = evaluate(pred, GT)
    assert ev["event_recall"] == 0 and ev["event_precision"] == 0
    assert rows["texting"]["event_recall"] == 0
    assert rows["phone_call"]["event_precision"] == 0
    assert 0.4 < overall["recall"] < 0.6   # it did notice the driver was distracted
