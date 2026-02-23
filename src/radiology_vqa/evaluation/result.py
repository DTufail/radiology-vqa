"""Structured result models for evaluation outputs.

These are the data contracts between the evaluator, comparator, and report
generator. All are Pydantic models, serializable to JSON.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class PerSampleResult(BaseModel):
    """Evaluation result for a single sample."""

    sample_id: str
    question: str
    ground_truth: str
    prediction: str
    correct: bool
    answer_type: str  # "closed" or "open"
    confidence: float
    latency_seconds: float

    # Agent-specific (empty/default for VLM-only mode)
    decision: str = ""
    citations: list[dict] = Field(default_factory=list)
    reasoning: str = ""
    retrieval_query: str = ""
    visual_answer: str = ""
    retry_count: int = 0


class EvaluationResult(BaseModel):
    """Complete evaluation output for one run (agent or VLM-only)."""

    # ── Metadata ────────────────────────────────────────────────────
    model_name: str
    dataset: str
    split: str
    total_samples: int
    evaluation_mode: str  # "agent" or "vlm_only"
    timestamp: str
    config_snapshot: dict = Field(default_factory=dict)

    # ── Core Accuracy ───────────────────────────────────────────────
    overall_accuracy: float
    closed_accuracy: float
    open_accuracy: float

    # ── Closed Detail ───────────────────────────────────────────────
    closed_precision: float
    closed_recall: float
    closed_f1: float
    closed_confusion: dict  # {"tp", "tn", "fp", "fn"}
    closed_count: int

    # ── Open Detail ─────────────────────────────────────────────────
    open_token_f1: float
    open_bleu_1: float
    open_bertscore_f1: float  # -1.0 if not computed
    open_bertscore_precision: float = -1.0
    open_bertscore_recall: float = -1.0
    open_count: int

    # ── Agent-Specific (defaults for VLM-only) ──────────────────────
    abstention_rate: float = 0.0
    accuracy_when_answered: float = 0.0
    re_query_rate: float = 0.0
    citation_relevance_hit_rate: float = 0.0

    # ── Confidence Calibration ──────────────────────────────────────
    ece: float = 0.0
    mean_correct_confidence: float = 0.0
    mean_wrong_confidence: float = 0.0
    confidence_auroc: float = 0.0
    calibration_bins: list[dict] = Field(default_factory=list)
    threshold_analysis: list[dict] = Field(default_factory=list)

    # ── Runtime ─────────────────────────────────────────────────────
    total_seconds: float
    mean_latency_seconds: float
    median_latency_seconds: float = 0.0

    # ── Per-Sample ──────────────────────────────────────────────────
    per_sample: list[PerSampleResult] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save as JSON. Creates parent directories if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "EvaluationResult":
        """Load from JSON file."""
        return cls.model_validate_json(path.read_text())


class ComparisonResult(BaseModel):
    """Side-by-side comparison of agent vs baseline."""

    agent_name: str
    baseline_name: str
    dataset: str
    split: str
    total_samples: int

    # ── Metric Deltas (agent - baseline, positive = agent better) ───
    accuracy_delta: float
    closed_accuracy_delta: float
    open_accuracy_delta: float
    open_token_f1_delta: float
    open_bertscore_delta: float

    # ── Grounding Analysis ──────────────────────────────────────────
    improved: int  # agent correct, baseline wrong
    degraded: int  # agent wrong, baseline correct
    both_correct: int
    both_wrong: int
    agent_abstained: int
    abstain_vlm_correct: int  # over-abstention
    abstain_vlm_wrong: int  # justified abstention
    net_improvement: int  # improved - degraded

    # ── Abstention Analysis ─────────────────────────────────────────
    correct_abstention_rate: float  # of abstentions, fraction VLM would've missed

    # ── Statistical Significance ────────────────────────────────────
    mcnemar_statistic: float = 0.0
    mcnemar_p_value: float = 1.0
    is_significant: bool = False  # p < 0.05

    # ── Formatted Tables ────────────────────────────────────────────
    comparison_table_md: str = ""
    grounding_table_md: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "ComparisonResult":
        return cls.model_validate_json(path.read_text())
