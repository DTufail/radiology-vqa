"""Confidence calibration analysis for VLM and agent predictions.

Calibration measures whether predicted confidence scores are meaningful.
A model saying "0.9 confidence" should be correct ~90% of the time.
If confidence is poorly calibrated, the supervisor's thresholds (0.85, 0.55)
are arbitrary numbers rather than meaningful decision boundaries.
"""

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Partition predictions into equal-width confidence bins [0, 0.1), [0.1, 0.2), ...
    For each bin:
        bin_accuracy = fraction of correct predictions in bin
        bin_confidence = mean confidence in bin
        bin_weight = fraction of total samples in bin

    ECE = sum(bin_weight * |bin_accuracy - bin_confidence|)

    ECE = 0 means perfectly calibrated.
    ECE = 1 means maximally miscalibrated.

    For context: LLaVA v1.6 on VQA-RAD has mean correct confidence 0.896
    and mean wrong confidence 0.815. If those averages hold per-bin,
    we'd expect moderate ECE (~0.1-0.2).

    Edge cases:
    - Empty inputs → 0.0
    - All samples in one bin → ECE for that single bin
    - Empty bins are skipped (zero weight)
    """
    if not confidences or not correct:
        return 0.0

    n = len(confidences)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins

        # Last bin: include upper boundary (confidence == 1.0).
        if i == n_bins - 1:
            in_bin = [j for j in range(n) if bin_lower <= confidences[j] <= bin_upper]
        else:
            in_bin = [j for j in range(n) if bin_lower <= confidences[j] < bin_upper]

        if not in_bin:
            continue

        bin_acc = sum(bool(correct[j]) for j in in_bin) / len(in_bin)
        bin_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
        bin_weight = len(in_bin) / n
        ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def calibration_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> list[dict]:
    """Detailed per-bin calibration data for visualization.

    Returns list of dicts, one per bin:
    {
        "bin_start": float,         # e.g., 0.0
        "bin_end": float,           # e.g., 0.1
        "count": int,               # samples in this bin
        "mean_confidence": float,   # average confidence in bin
        "accuracy": float,          # fraction correct in bin
        "gap": float,               # |accuracy - mean_confidence|
    }

    Empty bins are included with count=0, all other values 0.0.
    """
    n = len(confidences)
    result: list[dict] = []

    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins

        if i == n_bins - 1:
            in_bin = [j for j in range(n) if bin_lower <= confidences[j] <= bin_upper]
        else:
            in_bin = [j for j in range(n) if bin_lower <= confidences[j] < bin_upper]

        if in_bin:
            mean_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
            accuracy = sum(bool(correct[j]) for j in in_bin) / len(in_bin)
            gap = abs(accuracy - mean_conf)
        else:
            mean_conf = 0.0
            accuracy = 0.0
            gap = 0.0

        result.append(
            {
                "bin_start": bin_lower,
                "bin_end": bin_upper,
                "count": len(in_bin),
                "mean_confidence": mean_conf,
                "accuracy": accuracy,
                "gap": gap,
            }
        )

    return result


def confidence_discrimination(
    confidences: Sequence[float],
    correct: Sequence[bool],
) -> dict[str, float]:
    """Measure how well confidence separates correct from wrong predictions.

    Returns:
    {
        "mean_correct_confidence": float,
        "mean_wrong_confidence": float,
        "confidence_gap": float,       # mean_correct - mean_wrong
        "auroc": float,                # area under ROC curve
    }

    AUROC treats confidence as a binary classifier score:
    - correct predictions = positive class
    - wrong predictions = negative class
    - AUROC > 0.5 means higher confidence → more likely correct
    - AUROC = 1.0 means perfect separation
    - AUROC = 0.5 means confidence is random/uninformative

    Use sklearn.metrics.roc_auc_score. Handle edge cases:
    - All correct or all wrong → AUROC undefined, return 0.5
    - Single sample → return 0.5

    For context: LLaVA has confidence_gap = 0.896 - 0.815 = 0.081.
    This is a small but positive gap, suggesting moderate discrimination.
    AUROC should be > 0.5 but probably not > 0.7.
    """
    if not confidences or not correct:
        return {
            "mean_correct_confidence": 0.0,
            "mean_wrong_confidence": 0.0,
            "confidence_gap": 0.0,
            "auroc": 0.5,
        }

    correct_confs = [c for c, ok in zip(confidences, correct) if bool(ok)]
    wrong_confs = [c for c, ok in zip(confidences, correct) if not bool(ok)]

    mean_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    mean_wrong = sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0.0
    gap = mean_correct - mean_wrong

    auroc = 0.5
    if correct_confs and wrong_confs:
        try:
            from sklearn.metrics import roc_auc_score

            labels = [1 if bool(ok) else 0 for ok in correct]
            auroc = float(roc_auc_score(labels, list(confidences)))
        except ValueError:
            # Only one class present (should not happen since we checked above).
            logger.warning(
                "confidence_discrimination: only one class present; AUROC undefined"
            )
            auroc = 0.5
        except ImportError:
            logger.warning("scikit-learn not installed; AUROC returning 0.5")
            auroc = 0.5

    return {
        "mean_correct_confidence": mean_correct,
        "mean_wrong_confidence": mean_wrong,
        "confidence_gap": gap,
        "auroc": auroc,
    }


def threshold_analysis(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float] = (
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ),
) -> list[dict]:
    """Evaluate accuracy at different confidence thresholds.

    For each threshold t:
    - Filter to samples with confidence >= t
    - Compute accuracy on filtered set
    - Count how many samples remain (coverage)

    Returns list of dicts:
    {
        "threshold": float,
        "coverage": float,           # fraction of samples above threshold
        "accuracy": float,           # accuracy on samples above threshold
        "count": int,                # absolute count above threshold
    }

    This directly informs the supervisor's threshold settings.
    If accuracy at threshold=0.85 is 65% but accuracy at threshold=0.90
    is 80%, then HIGH_CONFIDENCE should be 0.90, not 0.85.

    The optimal threshold is where accuracy-when-answered is maximized
    while keeping coverage acceptably high.
    """
    n = len(confidences)
    result: list[dict] = []

    for t in thresholds:
        above = [j for j in range(n) if confidences[j] >= t]
        count = len(above)
        coverage = count / n if n > 0 else 0.0
        accuracy = (
            sum(bool(correct[j]) for j in above) / count if count > 0 else 0.0
        )
        result.append(
            {
                "threshold": t,
                "coverage": coverage,
                "accuracy": accuracy,
                "count": count,
            }
        )

    return result
