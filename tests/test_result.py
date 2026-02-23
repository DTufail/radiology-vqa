"""Tests for evaluation/result.py — Pydantic data models, no GPU."""

import json

import pytest

from radiology_vqa.evaluation.result import (
    ComparisonResult,
    EvaluationResult,
    PerSampleResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _per_sample(**kwargs) -> PerSampleResult:
    defaults = dict(
        sample_id="vqa_rad_test_0",
        question="Is there any disease?",
        ground_truth="yes",
        prediction="yes",
        correct=True,
        answer_type="closed",
        confidence=0.9,
        latency_seconds=1.5,
    )
    defaults.update(kwargs)
    return PerSampleResult(**defaults)


def _eval_result(**kwargs) -> EvaluationResult:
    defaults = dict(
        model_name="llava-v1.6",
        dataset="vqa_rad",
        split="test",
        total_samples=100,
        evaluation_mode="agent",
        timestamp="2026-02-23T00:00:00",
        overall_accuracy=0.5,
        closed_accuracy=0.6,
        open_accuracy=0.4,
        closed_precision=0.6,
        closed_recall=0.6,
        closed_f1=0.6,
        closed_confusion={"tp": 6, "tn": 4, "fp": 4, "fn": 4},
        closed_count=50,
        open_token_f1=0.4,
        open_bleu_1=0.3,
        open_bertscore_f1=-1.0,
        open_count=50,
        total_seconds=100.0,
        mean_latency_seconds=1.0,
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


def _comparison_result(**kwargs) -> ComparisonResult:
    defaults = dict(
        agent_name="agent-llava",
        baseline_name="vlm-llava",
        dataset="vqa_rad",
        split="test",
        total_samples=100,
        accuracy_delta=0.05,
        closed_accuracy_delta=0.04,
        open_accuracy_delta=0.06,
        open_token_f1_delta=0.03,
        open_bertscore_delta=0.0,
        improved=20,
        degraded=8,
        both_correct=42,
        both_wrong=22,
        agent_abstained=8,
        abstain_vlm_correct=2,
        abstain_vlm_wrong=6,
        net_improvement=12,
        correct_abstention_rate=0.75,
    )
    defaults.update(kwargs)
    return ComparisonResult(**defaults)


# ── PerSampleResult ────────────────────────────────────────────────────────────


class TestPerSampleResult:
    def test_required_fields_validate(self):
        r = _per_sample()
        assert r.sample_id == "vqa_rad_test_0"
        assert r.correct is True
        assert r.confidence == pytest.approx(0.9)

    def test_agent_defaults(self):
        r = _per_sample()
        assert r.decision == ""
        assert r.citations == []
        assert r.reasoning == ""
        assert r.retrieval_query == ""
        assert r.visual_answer == ""
        assert r.retry_count == 0

    def test_agent_fields_explicit(self):
        r = _per_sample(
            decision="answer",
            citations=[{"text": "liver info"}],
            reasoning="high confidence",
            retrieval_query="liver anatomy",
            visual_answer="yes",
            retry_count=1,
        )
        assert r.decision == "answer"
        assert r.citations == [{"text": "liver info"}]
        assert r.retry_count == 1

    def test_serialization_roundtrip(self):
        r = _per_sample(decision="abstain", confidence=0.42)
        data = r.model_dump()
        r2 = PerSampleResult(**data)
        assert r2.decision == "abstain"
        assert r2.confidence == pytest.approx(0.42)


# ── EvaluationResult ──────────────────────────────────────────────────────────


class TestEvaluationResult:
    def test_required_fields_validate(self):
        r = _eval_result()
        assert r.model_name == "llava-v1.6"
        assert r.evaluation_mode == "agent"
        assert r.total_samples == 100

    def test_defaults_applied(self):
        r = _eval_result()
        assert r.abstention_rate == pytest.approx(0.0)
        assert r.ece == pytest.approx(0.0)
        assert r.calibration_bins == []
        assert r.per_sample == []
        assert r.config_snapshot == {}

    def test_save_creates_file_and_parent_dirs(self, tmp_path):
        r = _eval_result()
        path = tmp_path / "nested" / "sub" / "result.json"
        r.save(path)
        assert path.exists()

    def test_save_valid_json(self, tmp_path):
        r = _eval_result()
        path = tmp_path / "result.json"
        r.save(path)
        data = json.loads(path.read_text())
        assert data["model_name"] == "llava-v1.6"
        assert data["evaluation_mode"] == "agent"

    def test_load_roundtrip(self, tmp_path):
        r = _eval_result(
            overall_accuracy=0.412,
            per_sample=[_per_sample(sample_id="s0"), _per_sample(sample_id="s1")],
        )
        path = tmp_path / "result.json"
        r.save(path)
        r2 = EvaluationResult.load(path)
        assert r2.overall_accuracy == pytest.approx(0.412)
        assert len(r2.per_sample) == 2
        assert r2.per_sample[0].sample_id == "s0"

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            EvaluationResult(
                # missing model_name and many other required fields
                dataset="vqa_rad",
                split="test",
            )

    def test_large_per_sample_serializes(self, tmp_path):
        samples = [_per_sample(sample_id=f"s{i}") for i in range(1000)]
        r = _eval_result(per_sample=samples, total_samples=1000)
        path = tmp_path / "large.json"
        r.save(path)
        r2 = EvaluationResult.load(path)
        assert len(r2.per_sample) == 1000


# ── ComparisonResult ──────────────────────────────────────────────────────────


class TestComparisonResult:
    def test_required_fields(self):
        c = _comparison_result()
        assert c.net_improvement == 12
        assert c.is_significant is False

    def test_save_load_roundtrip(self, tmp_path):
        c = _comparison_result(
            accuracy_delta=0.05,
            mcnemar_p_value=0.03,
            is_significant=True,
            comparison_table_md="| a | b |\n|---|---|",
        )
        path = tmp_path / "comparison.json"
        c.save(path)
        c2 = ComparisonResult.load(path)
        assert c2.accuracy_delta == pytest.approx(0.05)
        assert c2.is_significant is True
        assert "| a | b |" in c2.comparison_table_md

    def test_grounding_fields(self):
        c = _comparison_result()
        assert c.improved + c.degraded + c.both_correct + c.both_wrong + c.agent_abstained == 100
