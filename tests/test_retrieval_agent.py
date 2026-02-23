"""Tests for the Retrieval Agent node."""

import pytest

from radiology_vqa.agents.retrieval_agent import retrieval_agent_node
from radiology_vqa.rag.document import Document, DocumentMeta, RetrievalResult


def _make_result(text: str, score: float = 0.72, entity: str = "Pneumonia") -> RetrievalResult:
    doc = Document(
        text=text,
        meta=DocumentMeta(
            source_type="kg_disease",
            entity_name=entity,
            attribute="symptom",
            source_file="en_disease.csv",
        ),
        doc_id=f"kg_disease_{entity.lower()}_symptom_0",
    )
    return RetrievalResult(document=doc, score=score, rank=1)


class TestRetrievalAgentNode:
    def test_query_includes_visual_answer_when_present(self, mock_retriever):
        state = {
            "question": "Is there evidence of consolidation?",
            "visual_answer": "consolidation",
            "visual_error": "",
            "retry_count": 0,
        }
        retriever = mock_retriever()
        result = retrieval_agent_node(state, retriever)

        assert "consolidation" in result["retrieval_query"]
        assert "Is there evidence of consolidation?" in result["retrieval_query"]

    def test_query_is_question_only_when_vlm_failed(self, mock_retriever):
        state = {
            "question": "Is there evidence of consolidation?",
            "visual_answer": "",
            "visual_error": "GPU OOM",
            "retry_count": 0,
        }
        retriever = mock_retriever()
        result = retrieval_agent_node(state, retriever)

        assert result["retrieval_query"] == "Is there evidence of consolidation?"

    def test_evidence_is_plain_dicts(self, mock_retriever):
        result_obj = _make_result("Pneumonia symptoms include fever and cough.")
        retriever = mock_retriever(results=[result_obj])

        state = {
            "question": "What is the diagnosis?",
            "visual_answer": "pneumonia",
            "visual_error": "",
        }
        output = retrieval_agent_node(state, retriever)
        evidence = output["retrieved_evidence"]

        assert isinstance(evidence, list)
        assert len(evidence) == 1
        assert isinstance(evidence[0], dict)

    def test_evidence_dict_has_all_required_keys(self, mock_retriever):
        result_obj = _make_result("Pneumonia symptoms include fever and cough.")
        retriever = mock_retriever(results=[result_obj])

        state = {
            "question": "What is the diagnosis?",
            "visual_answer": "pneumonia",
            "visual_error": "",
        }
        output = retrieval_agent_node(state, retriever)
        item = output["retrieved_evidence"][0]

        for key in ("text", "score", "source_type", "entity_name", "attribute", "rank"):
            assert key in item, f"Missing key: {key}"

    def test_evidence_dict_has_correct_values(self, mock_retriever):
        result_obj = _make_result("Pneumonia symptoms include fever.", score=0.85)
        retriever = mock_retriever(results=[result_obj])

        state = {
            "question": "What is shown?",
            "visual_answer": "pneumonia",
            "visual_error": "",
        }
        output = retrieval_agent_node(state, retriever)
        item = output["retrieved_evidence"][0]

        assert item["text"] == "Pneumonia symptoms include fever."
        assert item["score"] == 0.85
        assert item["source_type"] == "kg_disease"
        assert item["entity_name"] == "Pneumonia"

    def test_empty_results_when_retriever_returns_nothing(self, mock_retriever):
        retriever = mock_retriever(results=[])
        state = {
            "question": "What is shown?",
            "visual_answer": "pneumonia",
            "visual_error": "",
        }
        output = retrieval_agent_node(state, retriever)

        assert output["retrieved_evidence"] == []
        assert output["retrieval_error"] == ""

    def test_error_sets_retrieval_error_and_empty_evidence(self, mock_retriever):
        class FailingRetriever:
            def retrieve(self, query, top_k=5, min_score=0.0):
                raise RuntimeError("FAISS index not loaded")

        state = {
            "question": "What is shown?",
            "visual_answer": "pneumonia",
            "visual_error": "",
        }
        output = retrieval_agent_node(state, FailingRetriever())

        assert output["retrieval_error"] == "FAISS index not loaded"
        assert output["retrieved_evidence"] == []

    def test_error_does_not_raise(self, mock_retriever):
        class FailingRetriever:
            def retrieve(self, query, top_k=5, min_score=0.0):
                raise ValueError("Unexpected error")

        state = {
            "question": "test",
            "visual_answer": "",
            "visual_error": "",
        }
        # Must not raise
        output = retrieval_agent_node(state, FailingRetriever())
        assert "retrieval_error" in output

    def test_preserves_existing_state_keys(self, mock_retriever):
        state = {
            "question": "What is shown?",
            "visual_answer": "pneumonia",
            "visual_error": "",
            "retry_count": 1,
            "visual_confidence": 0.9,
        }
        retriever = mock_retriever()
        output = retrieval_agent_node(state, retriever)

        assert output["retry_count"] == 1
        assert output["visual_confidence"] == 0.9

    def test_does_not_mutate_input_state(self, mock_retriever):
        state = {
            "question": "What is shown?",
            "visual_answer": "pneumonia",
            "visual_error": "",
        }
        original_keys = set(state.keys())
        retrieval_agent_node(state, mock_retriever())

        assert set(state.keys()) == original_keys
