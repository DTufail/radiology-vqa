"""Tests for evaluation/calibration.py — confidence calibration, no GPU."""

import pytest

from radiology_vqa.evaluation.calibration import (
    calibration_bins,
    confidence_discrimination,
    expected_calibration_error,
    threshold_analysis,
)


# ── expected_calibration_error ────────────────────────────────────────────────


class TestExpectedCalibrationError:
    def test_perfectly_calibrated(self):
        # 10 samples at confidence=0.8, exactly 8 correct (80%)
        # All fall in bin [0.8, 0.9): acc=0.8, conf=0.8, gap=0 → ECE=0
        confidences = [0.8] * 10
        correct = [True] * 8 + [False] * 2
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_maximally_miscalibrated(self):
        # All confidence=1.0, only 50% correct → ECE=0.5
        confidences = [1.0] * 10
        correct = [True] * 5 + [False] * 5
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece == pytest.approx(0.5, abs=1e-9)

    def test_empty_input(self):
        assert expected_calibration_error([], [], n_bins=10) == 0.0

    def test_single_sample(self):
        # confidence=0.9, correct=True
        # Falls in last bin [0.9, 1.0]: acc=1.0, conf=0.9, gap=0.1, weight=1 → ECE=0.1
        ece = expected_calibration_error([0.9], [True], n_bins=10)
        assert ece == pytest.approx(0.1, abs=1e-9)

    def test_always_non_negative(self):
        import random

        random.seed(42)
        confs = [random.random() for _ in range(50)]
        correct_vals = [random.choice([True, False]) for _ in range(50)]
        ece = expected_calibration_error(confs, correct_vals)
        assert ece >= 0.0
        assert ece <= 1.0

    def test_all_wrong_high_confidence(self):
        # conf=0.9, all wrong: acc=0.0, gap=0.9, ECE=0.9
        confidences = [0.9] * 5
        correct = [False] * 5
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece == pytest.approx(0.9, abs=1e-9)


# ── calibration_bins ─────────────────────────────────────────────────────────


class TestCalibrationBins:
    def test_returns_n_bins_entries(self):
        confidences = [0.1, 0.5, 0.9]
        correct = [True, False, True]
        bins = calibration_bins(confidences, correct, n_bins=10)
        assert len(bins) == 10

    def test_all_required_keys(self):
        bins = calibration_bins([0.5], [True], n_bins=5)
        required = {"bin_start", "bin_end", "count", "mean_confidence", "accuracy", "gap"}
        for b in bins:
            assert set(b.keys()) == required

    def test_empty_bins_have_zero_count(self):
        # Both samples in [0.8, 0.9); all 9 other bins empty
        confidences = [0.85, 0.87]
        correct = [True, False]
        bins = calibration_bins(confidences, correct, n_bins=10)
        empty_bins = [b for b in bins if b["count"] == 0]
        assert len(empty_bins) == 9

    def test_bin_boundaries(self):
        bins = calibration_bins([0.5], [True], n_bins=10)
        assert bins[0]["bin_start"] == pytest.approx(0.0)
        assert bins[0]["bin_end"] == pytest.approx(0.1)
        assert bins[-1]["bin_start"] == pytest.approx(0.9)
        assert bins[-1]["bin_end"] == pytest.approx(1.0)

    def test_counts_sum_to_n(self):
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
        correct = [True] * 5
        bins = calibration_bins(confidences, correct, n_bins=5)
        assert sum(b["count"] for b in bins) == 5

    def test_accuracy_in_populated_bin(self):
        # 4 samples in [0.5, 0.6): 3 correct → accuracy = 0.75
        confidences = [0.51, 0.52, 0.53, 0.54]
        correct = [True, True, True, False]
        bins = calibration_bins(confidences, correct, n_bins=10)
        bin_5 = next(b for b in bins if pytest.approx(b["bin_start"], abs=1e-9) == 0.5)
        assert bin_5["count"] == 4
        assert bin_5["accuracy"] == pytest.approx(0.75)


# ── confidence_discrimination ─────────────────────────────────────────────────


class TestConfidenceDiscrimination:
    def test_perfect_separation(self):
        # High confidence → correct; low confidence → wrong
        confidences = [0.9, 0.95, 0.1, 0.05]
        correct = [True, True, False, False]
        result = confidence_discrimination(confidences, correct)
        assert result["auroc"] > 0.9
        assert result["confidence_gap"] > 0

    def test_all_correct_auroc_is_half(self):
        # Only one class → AUROC undefined → returns 0.5
        confidences = [0.9, 0.8]
        correct = [True, True]
        result = confidence_discrimination(confidences, correct)
        assert result["auroc"] == pytest.approx(0.5, abs=0.01)

    def test_returns_all_keys(self):
        result = confidence_discrimination([0.7, 0.3], [True, False])
        assert set(result.keys()) == {
            "mean_correct_confidence",
            "mean_wrong_confidence",
            "confidence_gap",
            "auroc",
        }

    def test_confidence_gap_formula(self):
        confidences = [0.9, 0.6]
        correct = [True, False]
        result = confidence_discrimination(confidences, correct)
        assert result["confidence_gap"] == pytest.approx(
            result["mean_correct_confidence"] - result["mean_wrong_confidence"]
        )

    def test_empty_input(self):
        result = confidence_discrimination([], [])
        assert result["auroc"] == pytest.approx(0.5)
        assert result["confidence_gap"] == pytest.approx(0.0)

    def test_mean_values_correct(self):
        confidences = [0.8, 0.9, 0.3, 0.4]
        correct = [True, True, False, False]
        result = confidence_discrimination(confidences, correct)
        assert result["mean_correct_confidence"] == pytest.approx(0.85)
        assert result["mean_wrong_confidence"] == pytest.approx(0.35)
        assert result["confidence_gap"] == pytest.approx(0.5)


# ── threshold_analysis ────────────────────────────────────────────────────────


class TestThresholdAnalysis:
    def test_returns_one_entry_per_threshold(self):
        thresholds = [0.5, 0.7, 0.9]
        confidences = [0.6, 0.8, 0.95]
        correct = [True, True, False]
        result = threshold_analysis(confidences, correct, thresholds=thresholds)
        assert len(result) == 3

    def test_coverage_non_increasing(self):
        confidences = [0.3, 0.5, 0.7, 0.9]
        correct = [True, False, True, True]
        result = threshold_analysis(confidences, correct)
        coverages = [r["coverage"] for r in result]
        assert all(coverages[i] >= coverages[i + 1] for i in range(len(coverages) - 1))

    def test_required_keys(self):
        result = threshold_analysis([0.5, 0.8], [True, False])
        for entry in result:
            assert set(entry.keys()) == {"threshold", "coverage", "accuracy", "count"}

    def test_zero_coverage_above_max_confidence(self):
        confidences = [0.3, 0.5, 0.7]
        correct = [True, False, True]
        result = threshold_analysis(confidences, correct, thresholds=[0.95])
        assert result[0]["coverage"] == pytest.approx(0.0)
        assert result[0]["count"] == 0
        assert result[0]["accuracy"] == pytest.approx(0.0)

    def test_full_coverage_at_zero_threshold(self):
        confidences = [0.3, 0.5, 0.7]
        correct = [True, False, True]
        result = threshold_analysis(confidences, correct, thresholds=[0.0])
        assert result[0]["coverage"] == pytest.approx(1.0)
        assert result[0]["count"] == 3
