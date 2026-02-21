"""Evaluation metrics for VLM benchmark runs."""

import logging
import string

logger = logging.getLogger(__name__)

_STRIP_CHARS = string.punctuation + string.whitespace


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Steps:
    1. Strip leading/trailing whitespace
    2. Lowercase
    3. Remove trailing punctuation (., !, ?)

    Args:
        answer: Raw answer string from model or ground truth.

    Returns:
        Normalized answer string.
    """
    text = answer.strip().lower().rstrip(_STRIP_CHARS).strip()
    return text


def is_match(predicted: str, ground_truth: str, answer_type: str) -> bool:
    """Determine if a prediction matches the ground truth.

    Both closed and open questions use exact match after normalization.
    This gives a conservative lower bound — no fuzzy matching at baseline.
    Relaxed metrics can be added in Phase 5.

    Args:
        predicted: Raw predicted answer string.
        ground_truth: Raw ground truth answer string.
        answer_type: "closed" or "open" (treated the same at this phase).

    Returns:
        True if normalized strings are equal.
    """
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def compute_metrics(per_sample: list[dict]) -> dict:
    """Compute aggregate metrics from per-sample prediction records.

    Args:
        per_sample: List of dicts, each containing at minimum
            ``"answer_type"`` (str) and ``"correct"`` (bool).

    Returns:
        Dict with keys:
            overall_accuracy, closed_accuracy, open_accuracy,
            total, total_closed, total_open,
            correct_total, correct_closed, correct_open.
    """
    total = len(per_sample)
    correct_total = sum(1 for s in per_sample if s["correct"])

    closed = [s for s in per_sample if s.get("answer_type") == "closed"]
    open_ = [s for s in per_sample if s.get("answer_type") == "open"]

    total_closed = len(closed)
    correct_closed = sum(1 for s in closed if s["correct"])

    total_open = len(open_)
    correct_open = sum(1 for s in open_ if s["correct"])

    return {
        "overall_accuracy": correct_total / total if total > 0 else 0.0,
        "closed_accuracy": correct_closed / total_closed if total_closed > 0 else 0.0,
        "open_accuracy": correct_open / total_open if total_open > 0 else 0.0,
        "total": total,
        "total_closed": total_closed,
        "total_open": total_open,
        "correct_total": correct_total,
        "correct_closed": correct_closed,
        "correct_open": correct_open,
    }
