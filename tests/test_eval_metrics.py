"""Tests for evaluation/metrics.py — pure VQA metrics, no GPU, no model loading."""

import pytest

from radiology_vqa.evaluation.metrics import (
    batch_bleu_1,
    batch_token_f1,
    bert_score_f1,
    bleu_1,
    closed_confusion_matrix,
    closed_precision_recall_f1,
    compute_all_metrics,
    exact_match_accuracy,
    normalize_answer,
    token_f1,
)

# Skip BERTScore tests if bert_score is not installed.
_bert_score_available = pytest.importorskip  # reference only for type hints
try:
    import bert_score as _bs  # noqa: F401

    _BERTSCORE_INSTALLED = True
except ImportError:
    _BERTSCORE_INSTALLED = False


# ── normalize_answer ───────────────────────────────────────────────────────────


class TestNormalizeAnswer:
    def test_removes_leading_article_the(self):
        assert normalize_answer("The Left Lung") == "left lung"

    def test_removes_trailing_punctuation_period(self):
        assert normalize_answer("Yes.") == "yes"

    def test_preserves_internal_hyphen(self):
        assert normalize_answer("X-ray") == "x-ray"

    def test_collapses_multiple_spaces(self):
        assert normalize_answer("  CT  scan  ") == "ct scan"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_removes_article_a(self):
        assert normalize_answer("A  large  tumor") == "large tumor"

    def test_removes_article_an(self):
        assert normalize_answer("An abnormality") == "abnormality"

    def test_lowercases(self):
        assert normalize_answer("LUNG") == "lung"

    def test_removes_question_mark(self):
        assert normalize_answer("Yes?") == "yes"

    def test_strips_outer_whitespace(self):
        assert normalize_answer("  yes  ") == "yes"


# ── exact_match_accuracy ───────────────────────────────────────────────────────


class TestExactMatchAccuracy:
    def test_all_correct(self):
        assert exact_match_accuracy(["yes", "no", "yes"], ["yes", "no", "yes"]) == 1.0

    def test_all_wrong(self):
        assert exact_match_accuracy(["yes", "yes"], ["no", "no"]) == 0.0

    def test_mixed(self):
        result = exact_match_accuracy(["yes", "no", "yes"], ["yes", "yes", "no"])
        assert result == pytest.approx(1 / 3)

    def test_case_insensitive(self):
        assert exact_match_accuracy(["Yes"], ["yes"]) == 1.0

    def test_empty_input(self):
        assert exact_match_accuracy([], []) == 0.0

    def test_whitespace_normalized(self):
        assert exact_match_accuracy(["  yes  "], ["yes"]) == 1.0

    def test_article_normalization(self):
        # "The liver" and "liver" both normalize to "liver"
        assert exact_match_accuracy(["The liver"], ["liver"]) == 1.0


# ── closed_precision_recall_f1 ─────────────────────────────────────────────────


class TestClosedPrecisionRecallF1:
    def test_perfect(self):
        result = closed_precision_recall_f1(["yes", "no", "yes"], ["yes", "no", "yes"])
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        result = closed_precision_recall_f1(["yes", "yes"], ["no", "no"])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_yes_biased_model(self):
        # All "yes" predictions; 6/10 GT is "yes"
        preds = ["yes"] * 10
        gts = ["yes"] * 6 + ["no"] * 4
        result = closed_precision_recall_f1(preds, gts)
        assert result["recall"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(0.6)
        # f1 = 2 * 0.6 * 1.0 / (0.6 + 1.0) = 1.2/1.6 = 0.75
        assert result["f1"] == pytest.approx(0.75)

    def test_empty_input(self):
        result = closed_precision_recall_f1([], [])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_returns_all_keys(self):
        result = closed_precision_recall_f1(["yes"], ["yes"])
        assert set(result.keys()) == {"precision", "recall", "f1"}


# ── closed_confusion_matrix ────────────────────────────────────────────────────


class TestClosedConfusionMatrix:
    def test_all_four_categories(self):
        preds = ["yes", "yes", "no", "no"]
        gts = ["yes", "no", "yes", "no"]
        result = closed_confusion_matrix(preds, gts)
        assert result["tp"] == 1  # pred=yes, gt=yes
        assert result["fp"] == 1  # pred=yes, gt=no
        assert result["fn"] == 1  # pred=no, gt=yes
        assert result["tn"] == 1  # pred=no, gt=no

    def test_all_true_positives(self):
        result = closed_confusion_matrix(["yes", "yes"], ["yes", "yes"])
        assert result == {"tp": 2, "tn": 0, "fp": 0, "fn": 0}

    def test_returns_all_keys(self):
        result = closed_confusion_matrix(["yes"], ["yes"])
        assert set(result.keys()) == {"tp", "tn", "fp", "fn"}


# ── token_f1 ───────────────────────────────────────────────────────────────────


class TestTokenF1:
    def test_partial_overlap(self):
        # "left lung" vs "left lower lung"
        # pred={left, lung}, gt={left, lower, lung}
        # intersection={left, lung}=2, precision=2/2=1.0, recall=2/3
        # f1 = 2*(1.0)*(2/3)/(1.0+2/3) = (4/3)/(5/3) = 4/5 = 0.8
        result = token_f1("left lung", "left lower lung")
        assert result == pytest.approx(0.8)

    def test_exact_match(self):
        assert token_f1("yes", "yes") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("yes", "no") == pytest.approx(0.0)

    def test_empty_prediction(self):
        assert token_f1("", "left lung") == 0.0

    def test_empty_ground_truth(self):
        assert token_f1("left lung", "") == 0.0

    def test_single_word_match(self):
        assert token_f1("lung", "lung") == pytest.approx(1.0)

    def test_article_removed_before_comparison(self):
        # "the liver" and "liver" should match fully after normalization
        assert token_f1("the liver", "liver") == pytest.approx(1.0)


# ── batch_token_f1 ────────────────────────────────────────────────────────────


class TestBatchTokenF1:
    def test_returns_mean(self):
        # token_f1("yes","yes")=1.0, token_f1("no","yes")=0.0 → mean=0.5
        result = batch_token_f1(["yes", "no"], ["yes", "yes"])
        assert result == pytest.approx(0.5)

    def test_empty(self):
        assert batch_token_f1([], []) == 0.0

    def test_all_correct(self):
        result = batch_token_f1(["liver", "lung"], ["liver", "lung"])
        assert result == pytest.approx(1.0)


# ── bleu_1 ────────────────────────────────────────────────────────────────────


class TestBleu1:
    def test_identical_strings(self):
        result = bleu_1("yes", "yes")
        assert result >= 0.9  # Should be 1.0 with unigram BLEU

    def test_no_overlap(self):
        result = bleu_1("spleen", "liver")
        assert result < 0.2  # epsilon smoothing gives small non-zero value

    def test_partial_overlap(self):
        # "left lung" vs "left lower lung"
        result = bleu_1("left lung", "left lower lung")
        assert 0.0 < result < 1.0

    def test_empty_prediction(self):
        assert bleu_1("", "yes") == 0.0

    def test_empty_reference(self):
        assert bleu_1("yes", "") == 0.0


# ── batch_bleu_1 ─────────────────────────────────────────────────────────────


class TestBatchBleu1:
    def test_returns_mean_of_individual_scores(self):
        s1 = bleu_1("yes", "yes")
        s2 = bleu_1("no", "yes")
        expected = (s1 + s2) / 2
        result = batch_bleu_1(["yes", "no"], ["yes", "yes"])
        assert result == pytest.approx(expected, abs=1e-9)

    def test_empty(self):
        assert batch_bleu_1([], []) == 0.0


# ── bert_score_f1 (slow) ──────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _BERTSCORE_INSTALLED, reason="bert_score not installed")
class TestBertScoreF1:
    def test_identical_strings_high_f1(self):
        result = bert_score_f1(["the liver is normal"], ["the liver is normal"])
        assert result["f1"] > 0.9

    def test_returns_all_keys(self):
        result = bert_score_f1(["yes"], ["no"])
        assert set(result.keys()) == {"precision", "recall", "f1"}

    def test_values_in_range(self):
        result = bert_score_f1(["liver", "lung"], ["liver", "spleen"])
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0


# ── bert_score_f1 fallback when not installed ─────────────────────────────────


class TestBertScoreFallback:
    def test_not_installed_returns_minus_one(self, monkeypatch):
        import sys

        # Temporarily hide bert_score
        monkeypatch.setitem(sys.modules, "bert_score", None)
        result = bert_score_f1(["yes"], ["no"])
        assert result == {"precision": -1.0, "recall": -1.0, "f1": -1.0}


# ── compute_all_metrics ───────────────────────────────────────────────────────


class TestComputeAllMetrics:
    def test_returns_all_expected_keys(self):
        preds = ["yes", "no", "liver", "lung"]
        gts = ["yes", "yes", "liver", "spleen"]
        atypes = ["closed", "closed", "open", "open"]
        result = compute_all_metrics(preds, gts, atypes, compute_bertscore=False)
        expected_keys = {
            "overall_accuracy",
            "closed_accuracy",
            "closed_precision",
            "closed_recall",
            "closed_f1",
            "closed_confusion",
            "closed_count",
            "open_accuracy",
            "open_token_f1",
            "open_bleu_1",
            "open_bertscore_f1",
            "open_bertscore_precision",
            "open_bertscore_recall",
            "open_count",
            "total_count",
        }
        assert set(result.keys()) == expected_keys

    def test_counts_sum_to_total(self):
        preds = ["yes", "liver", "no", "lung"]
        gts = ["yes", "liver", "no", "lung"]
        atypes = ["closed", "open", "closed", "open"]
        result = compute_all_metrics(preds, gts, atypes, compute_bertscore=False)
        assert result["closed_count"] + result["open_count"] == result["total_count"]

    def test_no_bertscore_returns_minus_one(self):
        result = compute_all_metrics(["liver"], ["liver"], ["open"], compute_bertscore=False)
        assert result["open_bertscore_f1"] == -1.0
        assert result["open_bertscore_precision"] == -1.0
        assert result["open_bertscore_recall"] == -1.0

    def test_all_closed_no_open(self):
        result = compute_all_metrics(
            ["yes", "no"], ["yes", "no"], ["closed", "closed"], compute_bertscore=False
        )
        assert result["open_count"] == 0
        assert result["closed_count"] == 2
        assert result["closed_accuracy"] == pytest.approx(1.0)

    def test_perfect_accuracy(self):
        preds = ["yes", "liver"]
        gts = ["yes", "liver"]
        atypes = ["closed", "open"]
        result = compute_all_metrics(preds, gts, atypes, compute_bertscore=False)
        assert result["overall_accuracy"] == pytest.approx(1.0)
        assert result["closed_accuracy"] == pytest.approx(1.0)
        assert result["open_accuracy"] == pytest.approx(1.0)
