"""Isotonic regression calibration — non-parametric fallback for Phase 6C.

Fits a monotone non-decreasing step function mapping raw confidence to
calibrated confidence.  More flexible than Platt scaling but requires more
validation data to avoid overfitting.

Serialisation stores the knot points (X_thresholds_, y_thresholds_) as JSON
and reconstructs the mapping via numpy.interp, avoiding any dependence on
sklearn's internal state across library versions.

Reference: Zadrozny & Elkan (2002)
"""

import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Non-parametric calibration via isotonic regression.

    Fits a monotone non-decreasing step function mapping raw confidence
    to calibrated confidence.  More flexible than Platt scaling but
    requires more validation data to avoid overfitting.

    Reference: Zadrozny & Elkan (2002)
    """

    def __init__(self) -> None:
        self._x_thresholds: Optional[np.ndarray] = None
        self._y_thresholds: Optional[np.ndarray] = None
        self.fitted: bool = False

    # ── public API ────────────────────────────────────────────────────────────

    def calibrate(self, confidence: float) -> float:
        """Apply isotonic calibration via np.interp. Clamps to [0.01, 0.99]."""
        if self._x_thresholds is None or self._y_thresholds is None:
            return confidence
        val = float(np.interp(confidence, self._x_thresholds, self._y_thresholds))
        return max(0.01, min(0.99, val))

    def fit(
        self,
        confidences: list[float],
        correct: list[bool],
    ) -> dict:
        """Fit isotonic regression on validation (confidence, correctness) pairs.

        Args:
            confidences: List of raw VLM confidence scores in [0, 1].
            correct:     List of bool indicating whether each prediction was correct.

        Returns:
            dict with keys: ece_before, ece_after, n_samples.

        Raises:
            ValueError: If inputs are empty or have mismatched lengths.
        """
        if not confidences or not correct:
            raise ValueError("confidences and correct must be non-empty")
        if len(confidences) != len(correct):
            raise ValueError(
                f"confidences and correct must have the same length "
                f"(got {len(confidences)} vs {len(correct)})"
            )

        from sklearn.isotonic import IsotonicRegression
        from radiology_vqa.evaluation.calibration import expected_calibration_error

        confs = np.array(confidences, dtype=np.float64)
        labels = np.array([1.0 if bool(c) else 0.0 for c in correct], dtype=np.float64)

        ece_before = expected_calibration_error(
            list(confidences), [bool(c) for c in correct]
        )

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(confs, labels)

        # Store knot points for JSON-serialisable reconstruction via np.interp.
        self._x_thresholds = ir.X_thresholds_.copy()
        self._y_thresholds = ir.y_thresholds_.copy()
        self.fitted = True

        calibrated = [self.calibrate(c) for c in confidences]
        ece_after = expected_calibration_error(calibrated, [bool(c) for c in correct])

        logger.info(
            "Isotonic calibration fitted: ECE %.4f → %.4f  (n=%d, knots=%d)",
            ece_before,
            ece_after,
            len(confidences),
            len(self._x_thresholds),
        )

        return {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "n_samples": len(confidences),
        }

    def save(self, path: str) -> None:
        """Save knot points to JSON for auditability and version control."""
        if self._x_thresholds is None:
            raise RuntimeError("IsotonicCalibrator not fitted yet; call fit() first.")
        data = {
            "method": "isotonic",
            "fitted": self.fitted,
            "X_thresholds": self._x_thresholds.tolist(),
            "y_thresholds": self._y_thresholds.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "IsotonicCalibrator saved to %s (%d knots)",
            path,
            len(self._x_thresholds),
        )

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        """Load knot points from JSON and reconstruct np.interp mapping."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calibrator = cls()
        calibrator._x_thresholds = np.array(data["X_thresholds"], dtype=np.float64)
        calibrator._y_thresholds = np.array(data["y_thresholds"], dtype=np.float64)
        calibrator.fitted = bool(data.get("fitted", True))
        logger.info(
            "IsotonicCalibrator loaded from %s (%d knots)",
            path,
            len(calibrator._x_thresholds),
        )
        return calibrator
