"""CPU-only result gates shared by evaluation and independent verification."""

import math


def align(rows, records, labels):
    assert len(records) == len(rows)
    assert len({row["id"] for row in records}) == len(records)
    for original, prediction in zip(rows, records):
        assert original["id"] == prediction["id"]
        assert labels.index(original["label"]) == prediction["gold"]


def assert_metrics(actual, expected):
    for key in ("n", "correct", "label_order", "confusion_matrix"):
        assert actual[key] == expected[key], key
    for key in ("accuracy", "macro_f1", "nll"):
        assert math.isclose(actual[key], expected[key], rel_tol=1e-8, abs_tol=1e-9), key


def paired(original, trained):
    result = {
        key: 0
        for key in (
            "both_correct",
            "base_wrong_trained_correct",
            "base_correct_trained_wrong",
            "both_wrong",
        )
    }
    for base, adapted in zip(original, trained):
        assert base["id"] == adapted["id"] and base["gold"] == adapted["gold"]
        base_correct = base["pred"] == base["gold"]
        trained_correct = adapted["pred"] == adapted["gold"]
        if base_correct and trained_correct:
            key = "both_correct"
        elif trained_correct:
            key = "base_wrong_trained_correct"
        elif base_correct:
            key = "base_correct_trained_wrong"
        else:
            key = "both_wrong"
        result[key] += 1
    assert sum(result.values()) == len(original) == len(trained)
    return result
