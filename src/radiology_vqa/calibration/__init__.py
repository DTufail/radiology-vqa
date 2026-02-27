"""Post-hoc confidence calibration for VLM outputs (Phase 6C).

Provides:
    PlattScaler     — parametric calibration via logistic regression (2 params)
    IsotonicCalibrator — non-parametric calibration via isotonic regression
"""

from radiology_vqa.calibration.platt import PlattScaler
from radiology_vqa.calibration.isotonic import IsotonicCalibrator

__all__ = ["PlattScaler", "IsotonicCalibrator"]
