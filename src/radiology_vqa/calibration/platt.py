"""Platt scaling for post-hoc confidence calibration.

Fits ``calibrated_conf = sigmoid(a * raw_conf + b)`` on validation
(confidence, correctness) pairs.  Two parameters fitted via L-BFGS-B
minimising negative log-likelihood.

Reference: Platt (1999), Guo et al. (ICML 2017)
"""

import json
import logging
import math
from typing import Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for a scalar."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class PlattScaler:
    """Post-hoc confidence calibration via Platt scaling.

    Fits ``calibrated_conf = sigmoid(a * raw_conf + b)`` on a validation set.
    Two parameters (a, b) fitted via L-BFGS-B minimising NLL.

    Before fitting:  a=1.0, b=0.0  →  calibrate(x) = sigmoid(x).
    After fitting:   a < 1 compresses overconfident scores toward 0.5;
                     b shifts the inflection point.

    Reference: Platt (1999), Guo et al. (ICML 2017)
    """

    def __init__(self) -> None:
        self.a: float = 1.0   # identity before fitting
        self.b: float = 0.0
        self.fitted: bool = False

    # ── public API ────────────────────────────────────────────────────────────

    def calibrate(self, confidence: float) -> float:
        """Apply calibration: sigmoid(a * conf + b). Clamps to [0.01, 0.99]."""
        raw = _sigmoid(self.a * confidence + self.b)
        return max(0.01, min(0.99, raw))

    def fit(
        self,
        confidences: list[float],
        correct: list[bool],
    ) -> dict:
        """Fit a, b on validation (confidence, correctness) pairs.

        Minimises NLL = -Σ[y_i * log(σ(a*c_i + b)) + (1-y_i) * log(1 - σ(a*c_i + b))]
        via scipy.optimize.minimize with L-BFGS-B.

        Args:
            confidences: List of raw VLM confidence scores in [0, 1].
            correct:     List of bool indicating whether each prediction was correct.

        Returns:
            dict with keys: a, b, ece_before, ece_after, n_samples.

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

        from radiology_vqa.evaluation.calibration import expected_calibration_error

        confs = np.array(confidences, dtype=np.float64)
        labels = np.array([1.0 if bool(c) else 0.0 for c in correct], dtype=np.float64)

        ece_before = expected_calibration_error(
            list(confidences), [bool(c) for c in correct]
        )

        def nll(params: np.ndarray) -> float:
            a, b = params
            logits = a * confs + b
            # Numerically stable log-sigmoid and log(1 - sigmoid)
            log_p = np.where(
                logits >= 0,
                -np.log1p(np.exp(-logits)),
                logits - np.log1p(np.exp(logits)),
            )
            log_1mp = np.where(
                logits >= 0,
                -logits - np.log1p(np.exp(-logits)),
                -np.log1p(np.exp(logits)),
            )
            return -float(np.mean(labels * log_p + (1.0 - labels) * log_1mp))

        result = minimize(
            nll,
            x0=np.array([1.0, 0.0]),
            method="L-BFGS-B",
        )

        self.a = float(result.x[0])
        self.b = float(result.x[1])
        self.fitted = True

        calibrated = [self.calibrate(c) for c in confidences]
        ece_after = expected_calibration_error(calibrated, [bool(c) for c in correct])

        logger.info(
            "Platt scaling fitted: a=%.4f b=%.4f  ECE %.4f → %.4f  (n=%d)",
            self.a,
            self.b,
            ece_before,
            ece_after,
            len(confidences),
        )

        return {
            "a": self.a,
            "b": self.b,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "n_samples": len(confidences),
        }

    def save(self, path: str) -> None:
        """Save parameters to JSON for auditability and version control."""
        data = {
            "method": "platt",
            "a": self.a,
            "b": self.b,
            "fitted": self.fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("PlattScaler saved to %s (a=%.4f b=%.4f)", path, self.a, self.b)

    @classmethod
    def load(cls, path: str) -> "PlattScaler":
        """Load parameters from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls()
        scaler.a = float(data["a"])
        scaler.b = float(data["b"])
        scaler.fitted = bool(data.get("fitted", True))
        logger.info(
            "PlattScaler loaded from %s (a=%.4f b=%.4f)", path, scaler.a, scaler.b
        )
        return scaler
