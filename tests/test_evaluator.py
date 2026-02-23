"""Tests for evaluation/evaluator.py — AgentEvaluator, no GPU.

All tests use mock runners/datasets so no models are loaded.
"""

import json

import pytest

from radiology_vqa.evaluation.evaluator import AgentEvaluator
from radiology_vqa.evaluation.result import EvaluationResult, PerSampleResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_samples(n: int = 5, answers: list[str] | None = None):
    """Create synthetic VQASample-like objects without PIL images."""
    from unittest.mock import MagicMock

    samples = []
    for i in range(n):
        s = MagicMock()
        s.sample_id = f"vqa_rad_test_{i}"
        s.question = f"Question {i}?"
        s.answer = answers[i] if answers else "yes"
        s.answer_type = "closed"
        s.image = MagicMock()  # mock PIL image
        samples.append(s)
    return samples


def _agent_run_info(prediction="yes", confidence=0.9, decision="answer"):
    return {
        "prediction": prediction,
        "confidence": confidence,
        "decision": decision,
        "citations": [],
        "reasoning": "test reasoning",
        "retrieval_query": "test query",
        "visual_answer": prediction,
        "retry_count": 0,
        "latency_seconds": 0.05,
    }


def _vlm_run_info(prediction="yes", confidence=0.8):
    return {
        "prediction": prediction,
        "confidence": confidence,
        "decision": "answer",
        "citations": [],
        "reasoning": "",
        "retrieval_query": "",
        "visual_answer": prediction,
        "retry_count": 0,
        "latency_seconds": 0.03,
    }


# ── Lazy initialization ───────────────────────────────────────────────────────


class TestAgentEvaluatorInit:
    def test_init_does_not_load_models(self):
        evaluator = AgentEvaluator()
        assert evaluator._agent_runner is None
        assert evaluator._vlm is None

    def test_init_with_config(self):
        from radiology_vqa.config import Settings

        cfg = Settings()
        evaluator = AgentEvaluator(cfg)
        assert evaluator._config is cfg
        assert evaluator._agent_runner is None


# ── Intermediate save / load ──────────────────────────────────────────────────


class TestIntermediateSaveLoad:
    def test_save_creates_valid_json(self, tmp_path):
        evaluator = AgentEvaluator()
        per_sample = [
            PerSampleResult(
                sample_id="s0",
                question="Q?",
                ground_truth="yes",
                prediction="yes",
                correct=True,
                answer_type="closed",
                confidence=0.9,
                latency_seconds=0.1,
            )
        ]
        path = tmp_path / "intermediate.json"
        evaluator._save_intermediate(per_sample, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert data[0]["sample_id"] == "s0"

    def test_load_intermediate_roundtrip(self, tmp_path):
        evaluator = AgentEvaluator()
        per_sample = [
            PerSampleResult(
                sample_id=f"s{i}",
                question=f"Q{i}?",
                ground_truth="yes",
                prediction="yes",
                correct=True,
                answer_type="closed",
                confidence=0.9,
                latency_seconds=0.1,
            )
            for i in range(3)
        ]
        path = tmp_path / "intermediate.json"
        evaluator._save_intermediate(per_sample, path)
        loaded = evaluator._load_intermediate(path)
        assert len(loaded) == 3
        assert loaded[1]["sample_id"] == "s1"

    def test_save_creates_parent_dirs(self, tmp_path):
        evaluator = AgentEvaluator()
        per_sample = []
        path = tmp_path / "nested" / "dir" / "intermediate.json"
        evaluator._save_intermediate(per_sample, path)
        assert path.exists()


# ── evaluate() with mock runner ──────────────────────────────────────────────


class TestEvaluateWithMock:
    def test_agent_mode_returns_evaluation_result(self):
        from unittest.mock import patch

        evaluator = AgentEvaluator()
        mock_samples = _make_mock_samples(5)

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_agent", side_effect=lambda img, q, at="": _agent_run_info()):
                result = evaluator.evaluate(mode="agent", save_intermediate=False)

        assert isinstance(result, EvaluationResult)
        assert result.evaluation_mode == "agent"
        assert result.total_samples == 5

    def test_vlm_only_mode_returns_evaluation_result(self):
        from unittest.mock import patch

        evaluator = AgentEvaluator()
        mock_samples = _make_mock_samples(5)

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_vlm_only", side_effect=lambda img, q: _vlm_run_info()):
                result = evaluator.evaluate(mode="vlm_only", save_intermediate=False)

        assert result.evaluation_mode == "vlm_only"
        assert result.abstention_rate == pytest.approx(0.0)

    def test_all_correct_gives_accuracy_one(self):
        from unittest.mock import patch

        evaluator = AgentEvaluator()
        mock_samples = _make_mock_samples(4, answers=["yes", "no", "yes", "no"])

        def mock_agent(img, q, at=""):
            idx = int(q.split()[1].rstrip("?"))
            preds = ["yes", "no", "yes", "no"]
            return _agent_run_info(prediction=preds[idx])

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_agent", side_effect=mock_agent):
                result = evaluator.evaluate(mode="agent", save_intermediate=False, compute_bertscore=False)

        assert result.overall_accuracy == pytest.approx(1.0)

    def test_per_sample_has_correct_count(self):
        from unittest.mock import patch

        evaluator = AgentEvaluator()
        mock_samples = _make_mock_samples(7)

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_agent", side_effect=lambda img, q, at="": _agent_run_info()):
                result = evaluator.evaluate(mode="agent", save_intermediate=False)

        assert len(result.per_sample) == 7

    def test_per_sample_error_does_not_stop_evaluation(self):
        from unittest.mock import patch

        evaluator = AgentEvaluator()
        mock_samples = _make_mock_samples(4)

        call_count = [0]

        def flaky_agent(img, q, at=""):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated VLM crash")
            return _agent_run_info()

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_agent", side_effect=flaky_agent):
                result = evaluator.evaluate(mode="agent", save_intermediate=False)

        # All 4 samples processed — the crash was caught and recorded
        assert len(result.per_sample) == 4
        # The crashed sample (index 1) should have prediction="" and correct=False
        assert result.per_sample[1].prediction == ""
        assert result.per_sample[1].correct is False

    def test_abstained_samples_counted(self):
        from unittest.mock import patch

        evaluator = AgentEvaluator()
        mock_samples = _make_mock_samples(6)

        def mock_agent(img, q, at=""):
            idx = int(q.split()[1].rstrip("?"))
            # samples 2, 4 abstain
            if idx in (2, 4):
                return _agent_run_info(prediction="ABSTAIN: uncertain", decision="abstain", confidence=0.3)
            return _agent_run_info()

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_agent", side_effect=mock_agent):
                result = evaluator.evaluate(mode="agent", save_intermediate=False)

        assert result.abstention_rate == pytest.approx(2 / 6)

    def test_intermediate_save_triggered_every_50(self, tmp_path):
        from unittest.mock import patch

        from radiology_vqa.config import Settings

        cfg = Settings(eval_output_dir=tmp_path)
        evaluator = AgentEvaluator(cfg)
        mock_samples = _make_mock_samples(50)

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_agent", side_effect=lambda img, q, at="": _agent_run_info()):
                result = evaluator.evaluate(mode="agent", save_intermediate=True)

        intermediate = tmp_path / "intermediate_agent_vqa_rad_test.json"
        assert intermediate.exists()
        assert len(result.per_sample) == 50

    def test_resume_skips_completed_samples(self, tmp_path):
        from unittest.mock import patch

        from radiology_vqa.config import Settings

        cfg = Settings(eval_output_dir=tmp_path)
        evaluator = AgentEvaluator(cfg)
        mock_samples = _make_mock_samples(5)

        # Pre-save 3 completed samples
        completed = [
            PerSampleResult(
                sample_id=f"vqa_rad_test_{i}",
                question=f"Question {i}?",
                ground_truth="yes",
                prediction="yes",
                correct=True,
                answer_type="closed",
                confidence=0.9,
                latency_seconds=0.1,
            )
            for i in range(3)
        ]
        resume_path = tmp_path / "intermediate.json"
        evaluator._save_intermediate(completed, resume_path)

        call_count = [0]

        def counting_agent(img, q, at=""):
            call_count[0] += 1
            return _agent_run_info()

        with patch.object(evaluator, "_load_dataset", return_value=mock_samples):
            with patch.object(evaluator, "_run_agent", side_effect=counting_agent):
                result = evaluator.evaluate(
                    mode="agent",
                    save_intermediate=False,
                    resume_from=resume_path,
                )

        # Only 2 new samples should have been processed (samples 3 and 4)
        assert call_count[0] == 2
        assert len(result.per_sample) == 5
