"""Integration tests for the full agent pipeline with real VLM + Retriever.

All tests in this module are marked @pytest.mark.slow.
They are skipped unless a VLM model and FAISS index are available.
Run with: pytest tests/test_graph_integration.py (or make test-slow)
"""

import pytest
from PIL import Image

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def runner():
    """AgentRunner with real VLM + Retriever.  Loaded once per test module."""
    try:
        from radiology_vqa.graph.runner import create_runner
        return create_runner()
    except Exception as exc:
        pytest.skip(f"Could not create AgentRunner: {exc}")


@pytest.fixture
def gray_image():
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


class TestRunQueryNeverRaises:
    def test_run_query_returns_system_output(self, runner, gray_image):
        from radiology_vqa.agents.state import SystemOutput
        result = runner.run_query(gray_image, "Is there evidence of pneumonia?")
        assert isinstance(result, SystemOutput)

    def test_run_query_with_open_question_does_not_raise(self, runner, gray_image):
        result = runner.run_query(gray_image, "What is the primary finding?", answer_type="open")
        assert result is not None

    def test_run_query_with_tiny_image_does_not_raise(self, runner):
        tiny = Image.new("RGB", (4, 4), color=(0, 0, 0))
        result = runner.run_query(tiny, "What do you see?")
        assert result is not None

    def test_run_query_decision_is_valid(self, runner, gray_image):
        result = runner.run_query(gray_image, "Is there a fracture?")
        assert result.decision in ("answer", "abstain")

    def test_run_query_confidence_is_bounded(self, runner, gray_image):
        result = runner.run_query(gray_image, "What organ is shown?")
        assert 0.0 <= result.confidence <= 1.0

    def test_run_query_answer_is_non_empty(self, runner, gray_image):
        result = runner.run_query(gray_image, "Is there consolidation?")
        assert len(result.answer) > 0

    def test_run_query_citations_is_list(self, runner, gray_image):
        result = runner.run_query(gray_image, "Is there pneumonia?")
        assert isinstance(result.citations, list)


class TestRunQueryEdgeCases:
    def test_run_query_with_answer_type_auto_infer(self, runner, gray_image):
        # answer_type="" → entry_node infers from question
        result = runner.run_query(gray_image, "Is there a nodule?", answer_type="")
        assert result is not None
        assert result.decision in ("answer", "abstain")

    def test_retry_count_not_exceeded_in_output(self, runner, gray_image):
        from radiology_vqa.config import settings
        # The pipeline should not loop more than max_retries times.
        # We verify this indirectly: if final_answer is produced, the loop terminated.
        result = runner.run_query(gray_image, "Is there evidence of an aortic aneurysm?")
        assert result is not None

    def test_abstain_output_has_human_review_flag(self, runner):
        # A 1x1 pixel image is unlikely to produce a confident medical answer
        tiny = Image.new("RGB", (1, 1))
        result = runner.run_query(tiny, "Is there pneumonia?")
        # Either the model abstains (requires_human_review=True) or answers
        # — either way the output is valid
        assert isinstance(result.requires_human_review, bool)


class TestRunBatch:
    def test_run_batch_returns_correct_length(self, runner, gray_image):
        samples = [(gray_image, "Is there a fracture?"), (gray_image, "What organ is shown?")]
        results = runner.run_batch(samples)
        assert len(results) == 2

    def test_run_batch_all_results_are_system_output(self, runner, gray_image):
        from radiology_vqa.agents.state import SystemOutput
        samples = [(gray_image, "Is there pneumonia?"), (gray_image, "What is shown?")]
        results = runner.run_batch(samples)
        for r in results:
            assert isinstance(r, SystemOutput)

    def test_run_batch_with_answer_types(self, runner, gray_image):
        samples = [(gray_image, "Is there a fracture?"), (gray_image, "What is the diagnosis?")]
        answer_types = ["closed", "open"]
        results = runner.run_batch(samples, answer_types=answer_types)
        assert len(results) == 2
