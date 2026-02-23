"""Tests for AgentState TypedDict and SystemOutput Pydantic model."""

import pytest
from PIL import Image

from radiology_vqa.agents.state import AgentState, SystemOutput


class TestAgentState:
    def test_accepts_input_fields_only(self):
        state: AgentState = {
            "image": Image.new("RGB", (224, 224), color="gray"),
            "question": "Is there evidence of pneumonia?",
            "answer_type": "closed",
            "retry_count": 0,
        }
        assert state["question"] == "Is there evidence of pneumonia?"
        assert state["answer_type"] == "closed"
        assert state["retry_count"] == 0

    def test_accepts_empty_dict(self):
        # total=False: all keys optional
        state: AgentState = {}
        assert isinstance(state, dict)

    def test_accepts_fully_populated_state(self):
        img = Image.new("RGB", (64, 64))
        state: AgentState = {
            "image": img,
            "question": "What is shown?",
            "answer_type": "open",
            "retry_count": 0,
            "visual_answer": "consolidation",
            "visual_confidence": 0.92,
            "visual_raw_output": "consolidation",
            "visual_model": "mock-vlm",
            "visual_error": "",
            "retrieval_query": "What is shown? consolidation",
            "retrieved_evidence": [],
            "retrieval_error": "",
            "decision": "answer",
            "decision_reasoning": "Confident and grounded.",
            "agreement_score": 0.8,
            "grounded_answer": "consolidation",
            "grounded_confidence": 0.88,
            "final_answer": "consolidation",
            "final_confidence": 0.88,
            "citations": [],
            "requires_human_review": False,
            "output_reasoning": "Confident and grounded.",
        }
        assert state["grounded_confidence"] == 0.88


class TestSystemOutput:
    def test_validates_with_all_required_fields(self):
        output = SystemOutput(
            answer="pneumonia",
            confidence=0.92,
            citations=[],
            requires_human_review=False,
            reasoning="VLM confident with supporting evidence.",
            decision="answer",
            visual_answer="pneumonia",
            retrieval_query="Is there pneumonia?",
        )
        assert output.answer == "pneumonia"
        assert output.confidence == 0.92
        assert output.requires_human_review is False

    def test_rejects_empty_answer(self):
        with pytest.raises(Exception):
            SystemOutput(
                answer="",  # min_length=1
                confidence=0.5,
                citations=[],
                requires_human_review=False,
                reasoning="test",
                decision="answer",
                visual_answer="yes",
                retrieval_query="test query",
            )

    def test_rejects_confidence_above_one(self):
        with pytest.raises(Exception):
            SystemOutput(
                answer="yes",
                confidence=1.5,  # le=1.0
                citations=[],
                requires_human_review=False,
                reasoning="test",
                decision="answer",
                visual_answer="yes",
                retrieval_query="test",
            )

    def test_rejects_confidence_below_zero(self):
        with pytest.raises(Exception):
            SystemOutput(
                answer="yes",
                confidence=-0.1,  # ge=0.0
                citations=[],
                requires_human_review=False,
                reasoning="test",
                decision="answer",
                visual_answer="yes",
                retrieval_query="test",
            )

    def test_citations_is_list(self):
        output = SystemOutput(
            answer="yes",
            confidence=0.5,
            citations=[{"text": "some evidence", "score": 0.7}],
            requires_human_review=False,
            reasoning="test",
            decision="answer",
            visual_answer="yes",
            retrieval_query="test",
        )
        assert isinstance(output.citations, list)
        assert len(output.citations) == 1
