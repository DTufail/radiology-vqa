"""Tests for benchmark metrics: normalize_answer, is_match, compute_metrics."""

import pytest

from radiology_vqa.benchmark.metrics import compute_metrics, is_match, normalize_answer


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace():
    assert normalize_answer("  yes  ") == "yes"


def test_normalize_lowercases():
    assert normalize_answer("LEFT LUNG") == "left lung"


def test_normalize_removes_trailing_period():
    assert normalize_answer("  Yes. ") == "yes"


def test_normalize_removes_trailing_punctuation():
    assert normalize_answer("Pneumonia!") == "pneumonia"


def test_normalize_empty_string():
    assert normalize_answer("") == ""


def test_normalize_already_clean():
    assert normalize_answer("pneumonia") == "pneumonia"


def test_normalize_mixed():
    assert normalize_answer("  Bacterial Pneumonia. ") == "bacterial pneumonia"


# ---------------------------------------------------------------------------
# is_match
# ---------------------------------------------------------------------------


def test_is_match_closed_exact():
    assert is_match("yes", "yes", "closed") is True


def test_is_match_case_insensitive():
    assert is_match("Yes", "yes", "closed") is True


def test_is_match_with_trailing_period():
    assert is_match("yes.", "yes", "closed") is True


def test_is_match_no_match():
    assert is_match("no", "yes", "closed") is False


def test_is_match_open_exact():
    assert is_match("pneumonia", "pneumonia", "open") is True


def test_is_match_open_case_insensitive():
    assert is_match("Pneumonia", "pneumonia", "open") is True


def test_is_match_open_no_match():
    assert is_match("atelectasis", "pneumonia", "open") is False


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_basic():
    per_sample = [
        {"answer_type": "closed", "correct": True},
        {"answer_type": "closed", "correct": False},
        {"answer_type": "open", "correct": True},
    ]
    m = compute_metrics(per_sample)

    assert m["overall_accuracy"] == pytest.approx(2 / 3)
    assert m["closed_accuracy"] == pytest.approx(0.5)
    assert m["open_accuracy"] == pytest.approx(1.0)
    assert m["total"] == 3
    assert m["total_closed"] == 2
    assert m["total_open"] == 1
    assert m["correct_closed"] == 1
    assert m["correct_open"] == 1


def test_compute_metrics_all_correct():
    per_sample = [
        {"answer_type": "closed", "correct": True},
        {"answer_type": "open", "correct": True},
    ]
    m = compute_metrics(per_sample)
    assert m["overall_accuracy"] == pytest.approx(1.0)


def test_compute_metrics_all_wrong():
    per_sample = [
        {"answer_type": "closed", "correct": False},
        {"answer_type": "open", "correct": False},
    ]
    m = compute_metrics(per_sample)
    assert m["overall_accuracy"] == pytest.approx(0.0)


def test_compute_metrics_zero_division_no_open():
    """If there are no open questions, open_accuracy should be 0.0 (not crash)."""
    per_sample = [
        {"answer_type": "closed", "correct": True},
        {"answer_type": "closed", "correct": True},
    ]
    m = compute_metrics(per_sample)
    assert m["open_accuracy"] == pytest.approx(0.0)
    assert m["total_open"] == 0


def test_compute_metrics_zero_division_no_closed():
    """If there are no closed questions, closed_accuracy should be 0.0."""
    per_sample = [
        {"answer_type": "open", "correct": True},
        {"answer_type": "open", "correct": False},
    ]
    m = compute_metrics(per_sample)
    assert m["closed_accuracy"] == pytest.approx(0.0)
    assert m["total_closed"] == 0


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m["overall_accuracy"] == pytest.approx(0.0)
    assert m["total"] == 0
