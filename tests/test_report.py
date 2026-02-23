"""Tests for evaluation/report.py — report generation, no GPU."""

import pytest

from radiology_vqa.evaluation.report import (
    _format_threshold_recommendation,
    generate_report,
)
from radiology_vqa.evaluation.result import (
    ComparisonResult,
    EvaluationResult,
    PerSampleResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _per_sample(
    sample_id: str,
    correct: bool = True,
    decision: str = "answer",
    question: str = "Is there a tumor?",
    ground_truth: str = "yes",
    prediction: str = "yes",
) -> PerSampleResult:
    return PerSampleResult(
        sample_id=sample_id,
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
        correct=correct,
        answer_type="closed",
        confidence=0.9 if correct else 0.4,
        latency_seconds=0.1,
        decision=decision,
        citations=[{"text": "liver info"}] if correct else [],
    )


def _eval_result(n: int = 20, evaluation_mode: str = "agent") -> EvaluationResult:
    per_sample = [_per_sample(f"s{i}", correct=(i % 2 == 0)) for i in range(n)]
    acc = sum(r.correct for r in per_sample) / n
    return EvaluationResult(
        model_name="llava-v1.6-test",
        dataset="vqa_rad",
        split="test",
        total_samples=n,
        evaluation_mode=evaluation_mode,
        timestamp="2026-02-23T12:00:00",
        overall_accuracy=acc,
        closed_accuracy=acc,
        open_accuracy=acc,
        closed_precision=acc,
        closed_recall=acc,
        closed_f1=acc,
        closed_confusion={"tp": 5, "tn": 5, "fp": 5, "fn": 5},
        closed_count=n,
        open_token_f1=acc,
        open_bleu_1=0.4,
        open_bertscore_f1=-1.0,
        open_count=0,
        abstention_rate=0.1,
        accuracy_when_answered=0.55,
        ece=0.12,
        mean_correct_confidence=0.85,
        mean_wrong_confidence=0.55,
        confidence_auroc=0.68,
        calibration_bins=[
            {
                "bin_start": 0.3,
                "bin_end": 0.5,
                "count": 5,
                "mean_confidence": 0.42,
                "accuracy": 0.4,
                "gap": 0.02,
            }
        ],
        threshold_analysis=[
            {"threshold": 0.5, "coverage": 1.0, "accuracy": 0.5, "count": n},
            {"threshold": 0.7, "coverage": 0.6, "accuracy": 0.6, "count": 12},
            {"threshold": 0.9, "coverage": 0.3, "accuracy": 0.7, "count": 6},
        ],
        total_seconds=20.0,
        mean_latency_seconds=1.0,
        per_sample=per_sample,
    )


def _comparison_result(agent: EvaluationResult, baseline: EvaluationResult) -> ComparisonResult:
    return ComparisonResult(
        agent_name=agent.model_name,
        baseline_name=baseline.model_name,
        dataset="vqa_rad",
        split="test",
        total_samples=agent.total_samples,
        accuracy_delta=0.04,
        closed_accuracy_delta=0.03,
        open_accuracy_delta=0.05,
        open_token_f1_delta=0.02,
        open_bertscore_delta=0.0,
        improved=4,
        degraded=2,
        both_correct=8,
        both_wrong=4,
        agent_abstained=2,
        abstain_vlm_correct=0,
        abstain_vlm_wrong=2,
        net_improvement=2,
        correct_abstention_rate=0.5,
        mcnemar_statistic=4.0,
        mcnemar_p_value=0.045,
        is_significant=True,
        comparison_table_md="| Metric | VLM | Agent | Delta |\n|---|---|---|---|",
        grounding_table_md="| Category | Count | % |\n|---|---|---|",
    )


# ── generate_report ───────────────────────────────────────────────────────────


class TestGenerateReport:
    def test_agent_only_creates_report_md(self, tmp_path):
        agent = _eval_result()
        result_dir = generate_report(agent_result=agent, output_dir=tmp_path)
        assert (tmp_path / "report.md").exists()
        assert result_dir == tmp_path

    def test_agent_only_creates_agent_result_json(self, tmp_path):
        agent = _eval_result()
        generate_report(agent_result=agent, output_dir=tmp_path)
        assert (tmp_path / "agent_result.json").exists()

    def test_creates_report_json(self, tmp_path):
        agent = _eval_result()
        generate_report(agent_result=agent, output_dir=tmp_path)
        assert (tmp_path / "report.json").exists()

    def test_with_baseline_creates_baseline_json(self, tmp_path):
        agent = _eval_result()
        baseline = _eval_result(evaluation_mode="vlm_only")
        generate_report(agent_result=agent, baseline_result=baseline, output_dir=tmp_path)
        assert (tmp_path / "baseline_result.json").exists()

    def test_with_comparison_creates_comparison_json(self, tmp_path):
        agent = _eval_result()
        baseline = _eval_result(evaluation_mode="vlm_only")
        comparison = _comparison_result(agent, baseline)
        generate_report(
            agent_result=agent,
            baseline_result=baseline,
            comparison=comparison,
            output_dir=tmp_path,
        )
        assert (tmp_path / "comparison.json").exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        agent = _eval_result()
        nested = tmp_path / "deep" / "nested" / "dir"
        generate_report(agent_result=agent, output_dir=nested)
        assert (nested / "report.md").exists()

    def test_report_contains_expected_sections(self, tmp_path):
        agent = _eval_result()
        baseline = _eval_result(evaluation_mode="vlm_only")
        comparison = _comparison_result(agent, baseline)
        generate_report(
            agent_result=agent,
            baseline_result=baseline,
            comparison=comparison,
            output_dir=tmp_path,
        )
        content = (tmp_path / "report.md").read_text()
        assert "## Executive Summary" in content
        assert "## Closed-Ended Analysis" in content
        assert "## Open-Ended Analysis" in content
        assert "## Confidence Calibration" in content
        assert "## Error Analysis" in content
        assert "## Recommendations" in content

    def test_report_executive_summary_nonempty(self, tmp_path):
        agent = _eval_result()
        generate_report(agent_result=agent, output_dir=tmp_path)
        content = (tmp_path / "report.md").read_text()
        # Find summary section
        start = content.index("## Executive Summary")
        end = content.index("##", start + 5)
        summary = content[start:end].strip()
        assert len(summary) > 50  # non-trivial content


# ── _format_threshold_recommendation ─────────────────────────────────────────


class TestFormatThresholdRecommendation:
    def test_returns_recommendation_when_viable(self):
        data = [
            {"threshold": 0.5, "coverage": 1.0, "accuracy": 0.5, "count": 100},
            {"threshold": 0.7, "coverage": 0.6, "accuracy": 0.7, "count": 60},
            {"threshold": 0.9, "coverage": 0.3, "accuracy": 0.8, "count": 30},
        ]
        result = _format_threshold_recommendation(data)
        # Best viable (coverage >= 0.5) with highest accuracy is 0.7 threshold
        assert "0.70" in result
        assert "70.0%" in result

    def test_returns_note_when_no_viable_threshold(self):
        data = [
            {"threshold": 0.5, "coverage": 0.3, "accuracy": 0.9, "count": 30},
            {"threshold": 0.7, "coverage": 0.2, "accuracy": 0.95, "count": 20},
        ]
        result = _format_threshold_recommendation(data)
        assert "Insufficient" in result
