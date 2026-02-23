"""Tests for the Output Formatter node and format_system_output."""

import pytest

from radiology_vqa.agents.output_formatter import format_system_output, output_formatter_node
from radiology_vqa.agents.state import SystemOutput


SAMPLE_EVIDENCE = [
    {
        "text": "Pneumonia symptoms include fever and cough.",
        "score": 0.72,
        "source_type": "kg_disease",
        "entity_name": "Pneumonia",
        "attribute": "symptom",
        "rank": 1,
    },
    {
        "text": "Pneumonia is caused by bacterial infection.",
        "score": 0.65,
        "source_type": "kg_disease",
        "entity_name": "Pneumonia",
        "attribute": "cause",
        "rank": 2,
    },
    {
        "text": "Treatment for Pneumonia: antibiotics.",
        "score": 0.58,
        "source_type": "kg_disease",
        "entity_name": "Pneumonia",
        "attribute": "treatment",
        "rank": 3,
    },
    {
        "text": "Pneumonia prevention: vaccination.",
        "score": 0.45,
        "source_type": "kg_disease",
        "entity_name": "Pneumonia",
        "attribute": "prevention",
        "rank": 4,
    },
]


def _answer_state(evidence=None) -> dict:
    return {
        "decision": "answer",
        "grounded_answer": "pneumonia",
        "grounded_confidence": 0.88,
        "retrieved_evidence": evidence if evidence is not None else SAMPLE_EVIDENCE,
        "decision_reasoning": "VLM confident with supporting evidence.",
        "visual_answer": "pneumonia",
        "retrieval_query": "What is the diagnosis? pneumonia",
    }


def _abstain_state() -> dict:
    return {
        "decision": "abstain",
        "grounded_answer": "",
        "grounded_confidence": 0.0,
        "retrieved_evidence": [],
        "decision_reasoning": "VLM confidence too low.",
        "visual_answer": "unclear",
        "retrieval_query": "What is the diagnosis?",
    }


class TestOutputFormatterNode:
    def test_answer_decision_sets_final_answer(self):
        result = output_formatter_node(_answer_state())
        assert result["final_answer"] == "pneumonia"

    def test_answer_decision_sets_final_confidence(self):
        result = output_formatter_node(_answer_state())
        assert result["final_confidence"] == 0.88

    def test_answer_decision_does_not_require_human_review(self):
        result = output_formatter_node(_answer_state())
        assert result["requires_human_review"] is False

    def test_answer_decision_includes_citations(self):
        result = output_formatter_node(_answer_state())
        assert isinstance(result["citations"], list)
        assert len(result["citations"]) > 0

    def test_citations_capped_at_three(self):
        # SAMPLE_EVIDENCE has 4 items — should return at most 3
        result = output_formatter_node(_answer_state())
        assert len(result["citations"]) <= 3

    def test_citations_sorted_by_score_descending(self):
        result = output_formatter_node(_answer_state())
        scores = [c["score"] for c in result["citations"]]
        assert scores == sorted(scores, reverse=True)

    def test_citations_have_required_keys(self):
        result = output_formatter_node(_answer_state())
        for citation in result["citations"]:
            for key in ("source_type", "entity_name", "attribute", "text", "score"):
                assert key in citation, f"Missing citation key: {key}"

    def test_abstain_decision_sets_abstain_message(self):
        result = output_formatter_node(_abstain_state())
        assert result["final_answer"].startswith("ABSTAIN:")

    def test_abstain_decision_sets_zero_confidence(self):
        result = output_formatter_node(_abstain_state())
        assert result["final_confidence"] == 0.0

    def test_abstain_decision_requires_human_review(self):
        result = output_formatter_node(_abstain_state())
        assert result["requires_human_review"] is True

    def test_abstain_decision_has_empty_citations(self):
        result = output_formatter_node(_abstain_state())
        assert result["citations"] == []

    def test_re_query_treated_as_abstain(self):
        state = {
            "decision": "re_query",
            "grounded_answer": "",
            "grounded_confidence": 0.0,
            "retrieved_evidence": [],
            "decision_reasoning": "No supporting evidence found.",
        }
        result = output_formatter_node(state)
        assert result["final_answer"].startswith("ABSTAIN:")
        assert result["requires_human_review"] is True

    def test_output_reasoning_matches_decision_reasoning(self):
        result = output_formatter_node(_answer_state())
        assert result["output_reasoning"] == "VLM confident with supporting evidence."

    def test_preserves_existing_state_keys(self):
        state = _answer_state()
        state["retry_count"] = 1
        result = output_formatter_node(state)
        assert result["retry_count"] == 1

    def test_does_not_mutate_input_state(self):
        state = _answer_state()
        original_keys = set(state.keys())
        output_formatter_node(state)
        assert set(state.keys()) == original_keys

    def test_empty_evidence_answer_has_no_citations(self):
        result = output_formatter_node(_answer_state(evidence=[]))
        assert result["citations"] == []


class TestFormatSystemOutput:
    def test_produces_valid_system_output(self):
        state = _answer_state()
        state.update(output_formatter_node(state))
        output = format_system_output(state)
        assert isinstance(output, SystemOutput)

    def test_answer_field_matches_final_answer(self):
        state = _answer_state()
        state.update(output_formatter_node(state))
        output = format_system_output(state)
        assert output.answer == "pneumonia"

    def test_decision_field_preserved(self):
        state = _answer_state()
        state.update(output_formatter_node(state))
        output = format_system_output(state)
        assert output.decision == "answer"

    def test_visual_answer_preserved(self):
        state = _answer_state()
        state.update(output_formatter_node(state))
        output = format_system_output(state)
        assert output.visual_answer == "pneumonia"

    def test_retrieval_query_preserved(self):
        state = _answer_state()
        state.update(output_formatter_node(state))
        output = format_system_output(state)
        assert "pneumonia" in output.retrieval_query

    def test_abstain_output_requires_human_review(self):
        state = _abstain_state()
        state.update(output_formatter_node(state))
        output = format_system_output(state)
        assert output.requires_human_review is True
        assert output.confidence == 0.0

    def test_system_output_confidence_bounded(self):
        state = _answer_state()
        state.update(output_formatter_node(state))
        output = format_system_output(state)
        assert 0.0 <= output.confidence <= 1.0
