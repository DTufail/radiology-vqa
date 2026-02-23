"""Phase 5: Evaluation metrics for medical VQA.

Five modules:
- metrics: Standard VQA metrics (accuracy, token F1, BLEU, BERTScore)
- agent_metrics: Multi-agent specific (abstention, grounding, citations)
- calibration: Confidence analysis (ECE, AUROC, threshold analysis)
- result: Pydantic data models (PerSampleResult, EvaluationResult, ComparisonResult)
- evaluator: AgentEvaluator orchestrator
- comparator: BaselineComparator with McNemar's test
- report: generate_report → markdown + JSON deliverable
"""

from radiology_vqa.evaluation.agent_metrics import (
    abstention_rate,
    accuracy_when_answered,
    citation_relevance,
    correct_abstention_rate,
    grounding_improvement,
    re_query_rate,
    re_query_rate_from_counts,
)
from radiology_vqa.evaluation.calibration import (
    calibration_bins,
    confidence_discrimination,
    expected_calibration_error,
    threshold_analysis,
)
from radiology_vqa.evaluation.comparator import BaselineComparator
from radiology_vqa.evaluation.evaluator import AgentEvaluator
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

from radiology_vqa.evaluation.report import generate_report
from radiology_vqa.evaluation.result import (
    ComparisonResult,
    EvaluationResult,
    PerSampleResult,
)

__all__ = [
    # metrics
    "normalize_answer",
    "exact_match_accuracy",
    "closed_precision_recall_f1",
    "closed_confusion_matrix",
    "token_f1",
    "batch_token_f1",
    "bleu_1",
    "batch_bleu_1",
    "bert_score_f1",
    "compute_all_metrics",
    # agent_metrics
    "abstention_rate",
    "accuracy_when_answered",
    "correct_abstention_rate",
    "re_query_rate",
    "re_query_rate_from_counts",
    "grounding_improvement",
    "citation_relevance",
    # calibration
    "expected_calibration_error",
    "calibration_bins",
    "confidence_discrimination",
    "threshold_analysis",
    # result models
    "PerSampleResult",
    "EvaluationResult",
    "ComparisonResult",
    # orchestration
    "AgentEvaluator",
    "BaselineComparator",
    "generate_report",
]
