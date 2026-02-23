"""Tests for the Visual Agent node."""

import pytest
from PIL import Image

from radiology_vqa.agents.visual_agent import visual_agent_node


class TestVisualAgentNode:
    def test_returns_visual_fields_on_success(self, base_state, mock_vlm):
        result = visual_agent_node(base_state, mock_vlm)

        assert result["visual_answer"] == "yes"
        assert result["visual_confidence"] == 0.92
        assert result["visual_raw_output"] == "yes"
        assert result["visual_model"] == "mock-vlm"
        assert result["visual_error"] == ""

    def test_passes_image_and_question_to_vlm(self, base_state, mock_vlm):
        visual_agent_node(base_state, mock_vlm)

        assert mock_vlm.last_image is base_state["image"]
        assert mock_vlm.last_question == base_state["question"]

    def test_error_sets_visual_error_and_zero_confidence(self, base_state):
        class FailingVLM:
            def predict(self, image, question):
                raise RuntimeError("GPU OOM")

            def predict_batch(self, samples):
                return []

            @property
            def model_name(self):
                return "failing-vlm"

        result = visual_agent_node(base_state, FailingVLM())

        assert result["visual_error"] == "GPU OOM"
        assert result["visual_confidence"] == 0.0
        assert result["visual_answer"] == ""
        assert result["visual_raw_output"] == ""

    def test_error_does_not_raise(self, base_state):
        class FailingVLM:
            def predict(self, image, question):
                raise ValueError("Unexpected model error")

            def predict_batch(self, samples):
                return []

            @property
            def model_name(self):
                return "failing-vlm"

        # Should not raise — error goes to state
        result = visual_agent_node(base_state, FailingVLM())
        assert "visual_error" in result

    def test_preserves_existing_state_keys(self, base_state, mock_vlm):
        base_state["retry_count"] = 1
        result = visual_agent_node(base_state, mock_vlm)

        # Existing keys must survive
        assert result["retry_count"] == 1
        assert result["question"] == base_state["question"]
        assert result["answer_type"] == base_state["answer_type"]

    def test_does_not_mutate_input_state(self, base_state, mock_vlm):
        original_keys = set(base_state.keys())
        visual_agent_node(base_state, mock_vlm)

        # Input dict must be unmodified
        assert set(base_state.keys()) == original_keys

    def test_visual_error_empty_on_success(self, base_state, mock_vlm):
        result = visual_agent_node(base_state, mock_vlm)
        assert result["visual_error"] == ""
