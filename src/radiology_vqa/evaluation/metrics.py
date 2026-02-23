"""Core evaluation metrics for medical VQA.

All functions are pure: no side effects, no file I/O, no model loading.
All functions operate on batched inputs (lists).
All string comparisons use normalized answers (lowercased, stripped,
punctuation removed, articles removed).
"""

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer data is available (required by bleu_1).
try:
    import nltk

    for _resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(_resource)
        except LookupError:
            nltk.download(_resource.split("/")[-1], quiet=True)
except ImportError:
    pass  # nltk not installed; bleu_1 will raise ImportError at call time

_ARTICLES = frozenset({"a", "an", "the"})


# ── Answer Normalization ────────────────────────────────────────────────────────


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Steps (in order):
    1. Convert to lowercase
    2. Strip leading/trailing whitespace
    3. Remove articles: "a", "an", "the"
    4. Remove punctuation (except hyphens within words)
    5. Collapse multiple whitespace to single space
    6. Strip again

    Examples:
        "The Left Lung" → "left lung"
        "Yes." → "yes"
        "X-ray" → "x-ray"
        "  CT scan  " → "ct scan"

    This matches the standard VQA evaluation normalization used in
    VQA-RAD, SLAKE, and PathVQA papers.
    """
    if not answer:
        return ""

    text = answer.lower().strip()

    # Replace punctuation (except hyphens) with space.
    # [^\w\s-] matches anything that is not a word char, whitespace, or hyphen.
    text = re.sub(r"[^\w\s-]", " ", text)

    # Tokenize; strip leading/trailing hyphens from each token; drop empties.
    tokens = [t.strip("-") for t in text.split()]
    tokens = [t for t in tokens if t]

    # Remove articles.
    tokens = [t for t in tokens if t not in _ARTICLES]

    return " ".join(tokens)


# ── Closed-Ended Metrics (yes/no) ──────────────────────────────────────────────


def exact_match_accuracy(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> float:
    """Fraction of predictions that exactly match ground truth after normalization.

    This is the primary metric reported in VQA-RAD literature.
    Returns 0.0 if inputs are empty.
    """
    if not predictions or not ground_truths:
        return 0.0
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )
    correct = sum(
        1
        for p, g in zip(predictions, ground_truths)
        if normalize_answer(p) == normalize_answer(g)
    )
    return correct / len(predictions)


def closed_precision_recall_f1(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> dict[str, float]:
    """Binary classification metrics for yes/no questions.

    Treats "yes" as positive class, "no" as negative class.
    Predictions and ground truths that normalize to neither "yes" nor "no"
    are excluded with a logged warning.

    Returns:
        {"precision": float, "recall": float, "f1": float}

    Why this matters: LLaVA v1.6 predicts "yes" 165/251 times on closed
    questions. If ground truth is 60% "yes", a yes-biased model gets 60%
    accuracy but poor precision on "no". F1 exposes this.

    Edge cases:
    - All predictions same class → precision or recall = 0 for other class
    - Empty input → all values 0.0
    - Use zero_division=0 behavior (return 0.0, not NaN)
    """
    if not predictions or not ground_truths:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = tn = fp = fn = excluded = 0
    for pred, gt in zip(predictions, ground_truths):
        p = normalize_answer(pred)
        g = normalize_answer(gt)
        if p not in ("yes", "no") or g not in ("yes", "no"):
            excluded += 1
            continue
        if g == "yes" and p == "yes":
            tp += 1
        elif g == "no" and p == "no":
            tn += 1
        elif g == "no" and p == "yes":
            fp += 1
        else:  # g == "yes" and p == "no"
            fn += 1

    if excluded > 0:
        logger.warning(
            "closed_precision_recall_f1: excluded %d samples with non-binary answers",
            excluded,
        )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def closed_confusion_matrix(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> dict[str, int]:
    """Confusion matrix for yes/no classification.

    Returns: {"tp": int, "tn": int, "fp": int, "fn": int}
    where positive = "yes", negative = "no".

    tp = predicted "yes" and ground truth "yes"
    fp = predicted "yes" but ground truth "no"
    fn = predicted "no" but ground truth "yes"
    tn = predicted "no" and ground truth "no"
    """
    tp = tn = fp = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        p = normalize_answer(pred)
        g = normalize_answer(gt)
        if p not in ("yes", "no") or g not in ("yes", "no"):
            continue
        if g == "yes" and p == "yes":
            tp += 1
        elif g == "no" and p == "no":
            tn += 1
        elif g == "no" and p == "yes":
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ── Open-Ended Metrics ─────────────────────────────────────────────────────────


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between a single prediction and ground truth.

    Steps:
    1. Normalize both strings
    2. Tokenize by whitespace into word sets
    3. Compute:
       - precision = |pred_tokens ∩ gt_tokens| / |pred_tokens|
       - recall = |pred_tokens ∩ gt_tokens| / |gt_tokens|
       - f1 = 2 * precision * recall / (precision + recall)

    Edge cases:
    - Empty prediction → 0.0
    - Empty ground truth → 0.0
    - Identical strings → 1.0
    - No overlap → 0.0

    Example:
        "left lung" vs "left lower lung"
        → precision = 2/2 = 1.0, recall = 2/3 = 0.67, f1 ≈ 0.8

    This is the standard SQuAD token F1 metric.
    """
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = set(normalize_answer(ground_truth).split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    intersection = pred_tokens & gt_tokens
    if not intersection:
        return 0.0

    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def batch_token_f1(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> float:
    """Mean token F1 across all samples."""
    if not predictions or not ground_truths:
        return 0.0
    scores = [token_f1(p, g) for p, g in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)


def bleu_1(prediction: str, ground_truth: str) -> float:
    """Unigram BLEU score for a single prediction.

    Uses nltk.translate.bleu_score.sentence_bleu with:
    - weights=(1.0, 0, 0, 0) for unigram only
    - SmoothingFunction().method1 to handle short sentences

    Why BLEU-1: Medical VQA answers are 1-3 words. Higher-order BLEU
    (2,3,4-gram) is meaningless for such short answers. BLEU-1 measures
    unigram precision — did the model produce the right medical terms?

    Edge cases:
    - Empty prediction → 0.0
    - Empty reference → 0.0
    - Identical → 1.0
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        logger.warning("nltk not installed; bleu_1 returning 0.0")
        return 0.0

    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    if not pred_norm or not gt_norm:
        return 0.0

    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()

    smoothing = SmoothingFunction().method1
    try:
        return float(
            sentence_bleu(
                [gt_tokens],
                pred_tokens,
                weights=(1.0, 0, 0, 0),
                smoothing_function=smoothing,
            )
        )
    except Exception as exc:
        logger.warning("bleu_1 error: %s; returning 0.0", exc)
        return 0.0


def batch_bleu_1(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> float:
    """Mean BLEU-1 across all samples."""
    if not predictions or not ground_truths:
        return 0.0
    scores = [bleu_1(p, g) for p, g in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)


def bert_score_f1(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
) -> dict[str, float]:
    """BERTScore using contextual embeddings.

    Returns: {"precision": float, "recall": float, "f1": float}
    All values are means across samples.

    Uses the bert-score library. Handles:
    - Empty strings: replace with "[EMPTY]" placeholder before scoring
    - GPU memory: if deberta-xlarge-mnli causes OOM, catch the error,
      log a warning, and retry with "bert-base-uncased" as fallback
    - Batch processing: score all samples in one call (not per-sample)

    IMPORTANT: This function WILL load a transformer model into GPU memory.
    On T4 with VLM loaded (~4.5 GB), deberta-xlarge needs ~1.5 GB additional.
    If this is a problem, the caller should pass model_type="bert-base-uncased".
    """
    try:
        import bert_score as bs
    except ImportError:
        logger.warning("bert_score not installed; returning -1.0 placeholders")
        return {"precision": -1.0, "recall": -1.0, "f1": -1.0}

    if not predictions or not ground_truths:
        logger.warning("bert_score_f1: empty input; returning 0.0")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Replace empty strings: bert_score crashes on empty input.
    preds = [p if p.strip() else "[EMPTY]" for p in predictions]
    refs = [r if r.strip() else "[EMPTY]" for r in ground_truths]

    def _run(mt: str) -> dict[str, float]:
        P, R, F1 = bs.score(preds, refs, model_type=mt, batch_size=batch_size, verbose=False)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }

    try:
        return _run(model_type)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning(
                "BERTScore OOM with %s; falling back to bert-base-uncased", model_type
            )
            return _run("bert-base-uncased")
        raise


# ── Compound Metrics ───────────────────────────────────────────────────────────


def compute_all_metrics(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    answer_types: Sequence[str],
    confidences: Sequence[float] | None = None,
    compute_bertscore: bool = True,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
) -> dict:
    """Compute all metrics in one call, split by answer type.

    Splits samples by answer_type, computes:
    - Overall: exact_match_accuracy
    - Closed: exact_match_accuracy, precision, recall, f1, confusion_matrix
    - Open: exact_match_accuracy, token_f1, bleu_1, bertscore_f1

    Returns a flat dict:
    {
        "overall_accuracy": float,
        "closed_accuracy": float,
        "closed_precision": float,
        "closed_recall": float,
        "closed_f1": float,
        "closed_confusion": {"tp": int, "tn": int, "fp": int, "fn": int},
        "closed_count": int,
        "open_accuracy": float,
        "open_token_f1": float,
        "open_bleu_1": float,
        "open_bertscore_f1": float,     # -1.0 if compute_bertscore=False
        "open_bertscore_precision": float,
        "open_bertscore_recall": float,
        "open_count": int,
        "total_count": int,
    }
    """
    total = len(predictions)
    closed_idx = [i for i, t in enumerate(answer_types) if t == "closed"]
    open_idx = [i for i, t in enumerate(answer_types) if t == "open"]

    closed_preds = [predictions[i] for i in closed_idx]
    closed_gts = [ground_truths[i] for i in closed_idx]
    open_preds = [predictions[i] for i in open_idx]
    open_gts = [ground_truths[i] for i in open_idx]

    # Overall
    overall_acc = exact_match_accuracy(list(predictions), list(ground_truths))

    # Closed metrics
    closed_acc = exact_match_accuracy(closed_preds, closed_gts) if closed_preds else 0.0
    prf = (
        closed_precision_recall_f1(closed_preds, closed_gts)
        if closed_preds
        else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    )
    cm = (
        closed_confusion_matrix(closed_preds, closed_gts)
        if closed_preds
        else {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    )

    # Open metrics
    open_acc = exact_match_accuracy(open_preds, open_gts) if open_preds else 0.0
    open_tf1 = batch_token_f1(open_preds, open_gts) if open_preds else 0.0
    open_b1 = batch_bleu_1(open_preds, open_gts) if open_preds else 0.0

    if compute_bertscore and open_preds:
        bscore = bert_score_f1(open_preds, open_gts, model_type=bertscore_model)
    else:
        bscore = {"precision": -1.0, "recall": -1.0, "f1": -1.0}

    return {
        "overall_accuracy": overall_acc,
        "closed_accuracy": closed_acc,
        "closed_precision": prf["precision"],
        "closed_recall": prf["recall"],
        "closed_f1": prf["f1"],
        "closed_confusion": cm,
        "closed_count": len(closed_preds),
        "open_accuracy": open_acc,
        "open_token_f1": open_tf1,
        "open_bleu_1": open_b1,
        "open_bertscore_f1": bscore["f1"],
        "open_bertscore_precision": bscore["precision"],
        "open_bertscore_recall": bscore["recall"],
        "open_count": len(open_preds),
        "total_count": total,
    }
