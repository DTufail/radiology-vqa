"""Tests for evaluation/comparator.py — BaselineComparator, no GPU."""

import pytest

from radiology_vqa.evaluation.comparator import BaselineComparator
from radiology_vqa.evaluation.result import (
    ComparisonResult,
    EvaluationResult,
    PerSampleResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _per_sample(
    sample_id: str,
    ground_truth: str = "yes",
    prediction: str = "yes",
    correct: bool | None = None,
    decision: str = "answer",
    confidence: float = 0.8,
    answer_type: str = "closed",
) -> PerSampleResult:
    if correct is None:
        from radiology_vqa.evaluation.metrics import normalize_answer

        correct = normalize_answer(prediction) == normalize_answer(ground_truth)
    return PerSampleResult(
        sample_id=sample_id,
        question=f"Q {sample_id}?",
        ground_truth=ground_truth,
        prediction=prediction,
        correct=correct,
        answer_type=answer_type,
        confidence=confidence,
        latency_seconds=0.1,
        decision=decision,
    )


def _eval_result(
    per_sample: list[PerSampleResult],
    evaluation_mode: str = "agent",
    **kwargs,
) -> EvaluationResult:
    n = len(per_sample)
    correct = sum(r.correct for r in per_sample)
    acc = correct / n if n else 0.0
    defaults = dict(
        model_name="test-model",
        dataset="vqa_rad",
        split="test",
        total_samples=n,
        evaluation_mode=evaluation_mode,
        timestamp="2026-02-23T00:00:00",
        overall_accuracy=acc,
        closed_accuracy=acc,
        open_accuracy=acc,
        closed_precision=acc,
        closed_recall=acc,
        closed_f1=acc,
        closed_confusion={"tp": correct, "tn": 0, "fp": 0, "fn": n - correct},
        closed_count=n,
        open_token_f1=acc,
        open_bleu_1=acc,
        open_bertscore_f1=-1.0,
        open_count=0,
        total_seconds=1.0,
        mean_latency_seconds=0.1,
        per_sample=per_sample,
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


# ── Basic comparison ──────────────────────────────────────────────────────────


class TestBaselineComparatorBasic:
    def test_compare_returns_comparison_result(self):
        agent = _eval_result(
            [_per_sample("s0", prediction="yes"), _per_sample("s1", prediction="yes")]
        )
        baseline = _eval_result(
            [_per_sample("s0", prediction="yes"), _per_sample("s1", prediction="no", correct=False)],
            evaluation_mode="vlm_only",
        )
        cmp = BaselineComparator().compare(agent, baseline)
        assert isinstance(cmp, ComparisonResult)
        assert cmp.total_samples == 2

    def test_accuracy_delta_computed_correctly(self):
        agent = _eval_result(
            [_per_sample("s0"), _per_sample("s1"), _per_sample("s2")],
            overall_accuracy=0.6,
        )
        baseline = _eval_result(
            [_per_sample("s0"), _per_sample("s1"), _per_sample("s2")],
            overall_accuracy=0.5,
            evaluation_mode="vlm_only",
        )
        cmp = BaselineComparator().compare(agent, baseline)
        assert cmp.accuracy_delta == pytest.approx(0.1)

    def test_negative_delta_when_baseline_better(self):
        agent = _eval_result(
            [_per_sample("s0"), _per_sample("s1")],
            overall_accuracy=0.3,
        )
        baseline = _eval_result(
            [_per_sample("s0"), _per_sample("s1")],
            overall_accuracy=0.5,
            evaluation_mode="vlm_only",
        )
        cmp = BaselineComparator().compare(agent, baseline)
        assert cmp.accuracy_delta == pytest.approx(-0.2)

    def test_all_fields_populated(self):
        per_s = [_per_sample(f"s{i}") for i in range(4)]
        agent = _eval_result(per_s)
        baseline = _eval_result(per_s, evaluation_mode="vlm_only")
        cmp = BaselineComparator().compare(agent, baseline)
        # All grounding fields present
        assert hasattr(cmp, "improved")
        assert hasattr(cmp, "degraded")
        assert hasattr(cmp, "both_correct")
        assert hasattr(cmp, "both_wrong")
        assert hasattr(cmp, "agent_abstained")
        assert hasattr(cmp, "net_improvement")

    def test_grounding_counts_sum_to_total(self):
        agent_per = [
            _per_sample("s0", prediction="yes"),  # both correct
            _per_sample("s1", prediction="yes"),  # agent correct, baseline wrong → improved
            _per_sample("s2", prediction="no", correct=False),  # agent wrong, baseline correct → degraded
            _per_sample("s3", prediction="no", correct=False),  # both wrong
            _per_sample("s4", decision="abstain", prediction="ABSTAIN", correct=False),  # abstained
        ]
        baseline_per = [
            _per_sample("s0", prediction="yes"),  # both correct
            _per_sample("s1", prediction="no", correct=False),  # baseline wrong
            _per_sample("s2", prediction="yes"),  # baseline correct
            _per_sample("s3", prediction="no", correct=False),  # both wrong
            _per_sample("s4", prediction="no", correct=False),  # vlm wrong
        ]
        agent = _eval_result(agent_per)
        baseline = _eval_result(baseline_per, evaluation_mode="vlm_only")
        cmp = BaselineComparator().compare(agent, baseline)

        total = (
            cmp.improved + cmp.degraded + cmp.both_correct + cmp.both_wrong + cmp.agent_abstained
        )
        assert total == cmp.total_samples

    def test_net_improvement_equals_improved_minus_degraded(self):
        agent_per = [
            _per_sample("s0", prediction="yes"),  # improved
            _per_sample("s1", prediction="yes"),  # improved
            _per_sample("s2", prediction="no", correct=False),  # degraded
        ]
        baseline_per = [
            _per_sample("s0", prediction="no", correct=False),
            _per_sample("s1", prediction="no", correct=False),
            _per_sample("s2", prediction="yes"),
        ]
        agent = _eval_result(agent_per)
        baseline = _eval_result(baseline_per, evaluation_mode="vlm_only")
        cmp = BaselineComparator().compare(agent, baseline)
        assert cmp.net_improvement == cmp.improved - cmp.degraded

    def test_mismatched_sample_counts_uses_intersection(self):
        agent_per = [_per_sample(f"s{i}") for i in range(5)]
        baseline_per = [_per_sample(f"s{i}") for i in range(3)]  # only s0,s1,s2
        agent = _eval_result(agent_per)
        baseline = _eval_result(baseline_per, evaluation_mode="vlm_only")
        cmp = BaselineComparator().compare(agent, baseline)
        assert cmp.total_samples == 3  # intersection

    def test_correct_abstention_rate_computed(self):
        agent_per = [
            _per_sample("s0", decision="abstain", prediction="ABSTAIN", correct=False),
            _per_sample("s1", decision="abstain", prediction="ABSTAIN", correct=False),
        ]
        baseline_per = [
            _per_sample("s0", prediction="no", correct=False),  # VLM wrong → justified abstain
            _per_sample("s1", prediction="yes", correct=True),  # VLM correct → over-abstain
        ]
        agent = _eval_result(agent_per)
        baseline = _eval_result(baseline_per, evaluation_mode="vlm_only")
        cmp = BaselineComparator().compare(agent, baseline)
        assert cmp.correct_abstention_rate == pytest.approx(0.5)


# ── McNemar's test ────────────────────────────────────────────────────────────


class TestMcNemarTest:
    def test_fewer_than_5_discordant_returns_p_one(self):
        comparator = BaselineComparator()
        # All the same → 0 discordant pairs
        agent_correct = [True, True, True, True]
        baseline_correct = [True, True, True, True]
        stat, p = comparator._mcnemar_test(agent_correct, baseline_correct)
        assert p == pytest.approx(1.0)
        assert stat == pytest.approx(0.0)

    def test_sufficient_discordant_gives_valid_p(self):
        comparator = BaselineComparator()
        # 30 samples: agent better on 20, baseline better on 10 → large discordance
        agent_correct = [True] * 20 + [False] * 10 + [True] * 20 + [False] * 20
        baseline_correct = [False] * 20 + [True] * 10 + [True] * 20 + [False] * 20
        stat, p = comparator._mcnemar_test(agent_correct, baseline_correct)
        assert 0.0 <= p <= 1.0

    def test_all_discordant_large_p_when_symmetric(self):
        comparator = BaselineComparator()
        # Exactly half improved, half degraded → no effect → p should be high
        agent_correct = [True] * 15 + [False] * 15
        baseline_correct = [False] * 15 + [True] * 15
        stat, p = comparator._mcnemar_test(agent_correct, baseline_correct)
        assert p > 0.05  # symmetric → not significant


# ── Markdown tables ───────────────────────────────────────────────────────────


class TestMarkdownTables:
    def _make_pair(self):
        per_s = [_per_sample(f"s{i}") for i in range(10)]
        agent = _eval_result(per_s, overall_accuracy=0.6)
        baseline = _eval_result(per_s, overall_accuracy=0.5, evaluation_mode="vlm_only")
        return agent, baseline

    def test_comparison_table_is_markdown(self):
        agent, baseline = self._make_pair()
        cmp = BaselineComparator().compare(agent, baseline)
        assert "|" in cmp.comparison_table_md
        assert "Metric" in cmp.comparison_table_md
        assert "Agent" in cmp.comparison_table_md

    def test_grounding_table_is_markdown(self):
        agent, baseline = self._make_pair()
        cmp = BaselineComparator().compare(agent, baseline)
        assert "|" in cmp.grounding_table_md
        assert "Count" in cmp.grounding_table_md
