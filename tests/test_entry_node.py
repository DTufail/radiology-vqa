"""Tests for the Entry Node — validates and normalises inputs."""

import pytest
from PIL import Image

from radiology_vqa.graph.entry import entry_node


class TestEntryNodeValidImage:
    def test_valid_image_and_question_clears_visual_error(self, base_state):
        result = entry_node(base_state)
        assert result["visual_error"] == ""

    def test_valid_image_question_forwarded(self, base_state):
        result = entry_node(base_state)
        assert result["question"] == base_state["question"]

    def test_retry_count_defaults_to_zero(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "What is shown?"}
        result = entry_node(state)
        assert result["retry_count"] == 0

    def test_existing_retry_count_preserved(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "What is shown?", "retry_count": 2}
        result = entry_node(state)
        assert result["retry_count"] == 2

    def test_retrieval_error_defaults_to_empty_string(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "What is shown?"}
        result = entry_node(state)
        assert result["retrieval_error"] == ""


class TestEntryNodeImageValidation:
    def test_none_image_sets_visual_error(self):
        state = {"image": None, "question": "What is shown?"}
        result = entry_node(state)
        assert result["visual_error"] != ""

    def test_missing_image_sets_visual_error(self):
        state = {"question": "What is shown?"}
        result = entry_node(state)
        assert result["visual_error"] != ""

    def test_non_image_object_sets_visual_error(self):
        state = {"image": "not_an_image", "question": "What is shown?"}
        result = entry_node(state)
        assert result["visual_error"] != ""

    def test_cmyk_image_converted_to_rgb(self):
        img = Image.new("CMYK", (64, 64))
        state = {"image": img, "question": "What is shown?"}
        result = entry_node(state)
        assert result["image"].mode == "RGB"
        assert result["visual_error"] == ""

    def test_l_mode_image_converted_to_rgb(self):
        img = Image.new("L", (64, 64))
        state = {"image": img, "question": "What is shown?"}
        result = entry_node(state)
        assert result["image"].mode == "RGB"

    def test_oversized_image_resized_to_fit_within_4096(self):
        img = Image.new("RGB", (5000, 5000))
        state = {"image": img, "question": "What is shown?"}
        result = entry_node(state)
        w, h = result["image"].size
        assert w <= 4096 and h <= 4096

    def test_oversized_image_aspect_ratio_preserved(self):
        img = Image.new("RGB", (8000, 4000))
        state = {"image": img, "question": "What is shown?"}
        result = entry_node(state)
        w, h = result["image"].size
        assert abs(w / h - 2.0) < 0.01

    def test_normal_sized_image_not_resized(self):
        img = Image.new("RGB", (512, 512))
        state = {"image": img, "question": "What is shown?"}
        result = entry_node(state)
        assert result["image"].size == (512, 512)


class TestEntryNodeQuestionValidation:
    def test_empty_question_sets_visual_error(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": ""}
        result = entry_node(state)
        assert result["visual_error"] != ""

    def test_whitespace_only_question_sets_visual_error(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "   \t\n"}
        result = entry_node(state)
        assert result["visual_error"] != ""

    def test_question_leading_whitespace_stripped(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "  What is shown?  "}
        result = entry_node(state)
        assert result["question"] == "What is shown?"


class TestEntryNodeAnswerTypeInference:
    def test_is_question_inferred_as_closed(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "Is there a fracture?"}
        result = entry_node(state)
        assert result["answer_type"] == "closed"

    def test_are_question_inferred_as_closed(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "Are there signs of pneumonia?"}
        result = entry_node(state)
        assert result["answer_type"] == "closed"

    def test_does_question_inferred_as_closed(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "Does this show consolidation?"}
        result = entry_node(state)
        assert result["answer_type"] == "closed"

    def test_what_question_inferred_as_open(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "What organ is affected?"}
        result = entry_node(state)
        assert result["answer_type"] == "open"

    def test_which_question_inferred_as_open(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "Which modality is this?"}
        result = entry_node(state)
        assert result["answer_type"] == "open"

    def test_existing_answer_type_not_overridden(self):
        img = Image.new("RGB", (64, 64))
        # "Is there?" would infer "closed" but it's already "open"
        state = {"image": img, "question": "Is there a fracture?", "answer_type": "open"}
        result = entry_node(state)
        assert result["answer_type"] == "open"


class TestEntryNodeSafety:
    def test_does_not_raise_on_empty_state(self):
        result = entry_node({})
        assert "visual_error" in result

    def test_does_not_raise_on_none_values(self):
        result = entry_node({"image": None, "question": None})
        assert "visual_error" in result

    def test_does_not_mutate_input_state(self):
        img = Image.new("RGB", (64, 64))
        state = {"image": img, "question": "What is shown?"}
        original_keys = set(state.keys())
        entry_node(state)
        assert set(state.keys()) == original_keys
