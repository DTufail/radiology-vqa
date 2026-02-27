"""Unit tests for Phase 6C calibration classes.

All tests are fast — no GPU, no model loading.
"""

import json
import math

import numpy as np
import pytest

from radiology_vqa.calibration.platt import PlattScaler, _sigmoid
from radiology_vqa.calibration.isotonic import IsotonicCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_overconfident_data(n: int = 200, rng_seed: int = 0):
    """Synthetic data: conf ∈ [0.80, 0.95], but accuracy ≈ 50%.

    A perfectly calibrated model would have confidence ≈ 0.5 on this data.
    Platt fitting should compress the scores downward.
    """
    rng = np.random.default_rng(rng_seed)
    confidences = (rng.uniform(0.80, 0.95, n)).tolist()
    # Labels are ~50% correct regardless of confidence.
    correct = (rng.uniform(0, 1, n) < 0.5).tolist()
    return confidences, correct


def _make_logistic_data(n: int = 1000, a: float = 1.0, b: float = 0.0, rng_seed: int = 42):
    """Data generated from a logistic model: P(y=1) = sigmoid(a*conf + b).

    Fitting Platt on enough samples should recover the true a, b.
    """
    rng = np.random.default_rng(rng_seed)
    confidences = rng.uniform(0.2, 0.95, n).tolist()
    probs = [_sigmoid(a * c + b) for c in confidences]
    correct = [rng.uniform() < p for p in probs]
    return confidences, correct


# ---------------------------------------------------------------------------
# PlattScaler tests
# ---------------------------------------------------------------------------

class TestPlattScalerBeforeFit:
    def test_identity_before_fit(self):
        """Before fitting (a=1, b=0), calibrate(x) = sigmoid(x)."""
        scaler = PlattScaler()
        for x in [0.3, 0.5, 0.7, 0.9]:
            expected = _sigmoid(x)
            # sigmoid maps [0,1] to ~[0.57, 0.71] for these inputs; clamp won't hit.
            assert abs(scaler.calibrate(x) - max(0.01, min(0.99, expected))) < 1e-9

    def test_default_params(self):
        scaler = PlattScaler()
        assert scaler.a == 1.0
        assert scaler.b == 0.0
        assert scaler.fitted is False


class TestPlattScalerClamp:
    def test_clamp_high(self):
        """Output is clamped to 0.99 even if sigmoid would be higher."""
        scaler = PlattScaler()
        scaler.a = 100.0  # huge a → sigmoid ≈ 1.0
        scaler.b = 0.0
        assert scaler.calibrate(1.0) == pytest.approx(0.99)

    def test_clamp_low(self):
        """Output is clamped to 0.01 even if sigmoid would be lower."""
        scaler = PlattScaler()
        scaler.a = -100.0  # very negative → sigmoid ≈ 0.0
        scaler.b = 0.0
        assert scaler.calibrate(1.0) == pytest.approx(0.01)

    def test_output_always_in_range(self):
        scaler = PlattScaler()
        for conf in [0.0, 0.1, 0.5, 0.9, 1.0]:
            out = scaler.calibrate(conf)
            assert 0.01 <= out <= 0.99


class TestPlattScalerFitOverconfident:
    def test_overconfident_a_less_than_one(self):
        """Overconfident data (high conf, ~50% accuracy) → fitted a < 1."""
        confidences, correct = _make_overconfident_data(n=300, rng_seed=7)
        scaler = PlattScaler()
        result = scaler.fit(confidences, correct)
        # The slope should compress high confidence downward.
        assert scaler.a < 1.0, f"Expected a < 1 for overconfident data, got a={scaler.a:.4f}"
        assert scaler.fitted is True
        assert result["n_samples"] == 300

    def test_overconfident_calibrated_lower(self):
        """After fitting on overconfident data, calibrated conf at 0.9 < 0.9."""
        confidences, correct = _make_overconfident_data(n=300, rng_seed=13)
        scaler = PlattScaler()
        scaler.fit(confidences, correct)
        # Raw confidence of 0.9 should map to something lower.
        assert scaler.calibrate(0.9) < 0.9


class TestPlattScalerFitPerfect:
    def test_logistic_data_a_near_one(self):
        """Data from logistic model with a=1, b=0 → fitted a ≈ 1, b ≈ 0."""
        confidences, correct = _make_logistic_data(n=2000, a=1.0, b=0.0, rng_seed=42)
        scaler = PlattScaler()
        scaler.fit(confidences, correct)
        # With enough data, should recover a ≈ 1, b ≈ 0 within a generous margin.
        assert abs(scaler.a - 1.0) < 0.4, f"a={scaler.a:.4f} should be near 1.0"
        assert abs(scaler.b - 0.0) < 0.5, f"b={scaler.b:.4f} should be near 0.0"

    def test_logistic_data_fit_returns_dict(self):
        confidences, correct = _make_logistic_data(n=500, a=1.0, b=0.0)
        scaler = PlattScaler()
        result = scaler.fit(confidences, correct)
        assert set(result.keys()) == {"a", "b", "ece_before", "ece_after", "n_samples"}
        assert result["n_samples"] == 500
        assert isinstance(result["ece_before"], float)
        assert isinstance(result["ece_after"], float)


class TestPlattScalerECEImproves:
    def test_ece_improves_on_overconfident(self):
        """ECE after fitting < ECE before fitting on overconfident data."""
        confidences, correct = _make_overconfident_data(n=400, rng_seed=99)
        scaler = PlattScaler()
        result = scaler.fit(confidences, correct)
        assert result["ece_after"] < result["ece_before"], (
            f"Expected ECE to improve: before={result['ece_before']:.4f}, "
            f"after={result['ece_after']:.4f}"
        )


class TestPlattScalerSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        """Save to JSON, load back, parameters identical."""
        confidences, correct = _make_overconfident_data(n=200, rng_seed=5)
        scaler = PlattScaler()
        scaler.fit(confidences, correct)

        path = str(tmp_path / "platt.json")
        scaler.save(path)

        loaded = PlattScaler.load(path)
        assert loaded.a == pytest.approx(scaler.a)
        assert loaded.b == pytest.approx(scaler.b)
        assert loaded.fitted is True

    def test_save_produces_valid_json(self, tmp_path):
        scaler = PlattScaler()
        scaler.a = 0.75
        scaler.b = -0.5
        scaler.fitted = True
        path = str(tmp_path / "platt.json")
        scaler.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["method"] == "platt"
        assert data["a"] == pytest.approx(0.75)
        assert data["b"] == pytest.approx(-0.5)
        assert data["fitted"] is True

    def test_loaded_calibrate_matches_original(self, tmp_path):
        """Loaded scaler produces identical calibrated values."""
        confidences, correct = _make_overconfident_data(n=200, rng_seed=17)
        scaler = PlattScaler()
        scaler.fit(confidences, correct)
        path = str(tmp_path / "platt.json")
        scaler.save(path)

        loaded = PlattScaler.load(path)
        for c in [0.5, 0.7, 0.85, 0.9, 0.95]:
            assert loaded.calibrate(c) == pytest.approx(scaler.calibrate(c))


class TestPlattScalerValidation:
    def test_empty_confidences_raises(self):
        scaler = PlattScaler()
        with pytest.raises(ValueError, match="non-empty"):
            scaler.fit([], [])

    def test_mismatched_lengths_raises(self):
        scaler = PlattScaler()
        with pytest.raises(ValueError, match="same length"):
            scaler.fit([0.9, 0.8], [True])


# ---------------------------------------------------------------------------
# IsotonicCalibrator tests
# ---------------------------------------------------------------------------

class TestIsotonicCalibratorBasic:
    def test_calibrate_before_fit_passthrough(self):
        """Before fitting, calibrate(x) returns x (unfitted passthrough)."""
        cal = IsotonicCalibrator()
        assert cal.calibrate(0.8) == 0.8
        assert cal.fitted is False

    def test_monotone_output(self):
        """Isotonic calibration produces monotone non-decreasing output."""
        confidences, correct = _make_overconfident_data(n=300, rng_seed=3)
        cal = IsotonicCalibrator()
        cal.fit(confidences, correct)

        test_confs = np.linspace(0.01, 0.99, 50).tolist()
        calibrated = [cal.calibrate(c) for c in test_confs]

        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1] + 1e-9, (
                f"Non-monotone at index {i}: {calibrated[i]:.4f} > {calibrated[i+1]:.4f}"
            )

    def test_fit_returns_dict(self):
        confidences, correct = _make_overconfident_data(n=200)
        cal = IsotonicCalibrator()
        result = cal.fit(confidences, correct)
        assert set(result.keys()) == {"ece_before", "ece_after", "n_samples"}
        assert result["n_samples"] == 200

    def test_fitted_flag_set_after_fit(self):
        confidences, correct = _make_overconfident_data(n=100)
        cal = IsotonicCalibrator()
        assert cal.fitted is False
        cal.fit(confidences, correct)
        assert cal.fitted is True

    def test_output_clamped_to_range(self):
        confidences, correct = _make_overconfident_data(n=200)
        cal = IsotonicCalibrator()
        cal.fit(confidences, correct)
        for c in [0.0, 0.5, 1.0]:
            out = cal.calibrate(c)
            assert 0.01 <= out <= 0.99


class TestIsotonicCalibratorSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        """Save to JSON, load back, calibrate produces identical values."""
        confidences, correct = _make_overconfident_data(n=200, rng_seed=8)
        cal = IsotonicCalibrator()
        cal.fit(confidences, correct)

        path = str(tmp_path / "isotonic.json")
        cal.save(path)

        loaded = IsotonicCalibrator.load(path)
        assert loaded.fitted is True

        for c in [0.5, 0.7, 0.85, 0.9, 0.95]:
            assert loaded.calibrate(c) == pytest.approx(cal.calibrate(c), abs=1e-9)

    def test_save_produces_valid_json(self, tmp_path):
        confidences, correct = _make_overconfident_data(n=100)
        cal = IsotonicCalibrator()
        cal.fit(confidences, correct)
        path = str(tmp_path / "isotonic.json")
        cal.save(path)

        with open(path) as f:
            data = json.load(f)
        assert data["method"] == "isotonic"
        assert "X_thresholds" in data
        assert "y_thresholds" in data
        assert len(data["X_thresholds"]) == len(data["y_thresholds"])

    def test_save_before_fit_raises(self, tmp_path):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.save(str(tmp_path / "isotonic.json"))
