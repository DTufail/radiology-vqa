"""Tests for GraphBuilder — fast tests using build_lightweight().

These tests do NOT load a VLM or FAISS index.  The lightweight graph uses
passthrough nodes that forward pre-populated state values.
"""

import pytest
from PIL import Image

from radiology_vqa.config import Settings
from radiology_vqa.graph.builder import GraphBuilder


class TestBuildLightweight:
    def test_build_lightweight_returns_compiled_graph(self):
        builder = GraphBuilder(Settings())
        graph = builder.build_lightweight()
        assert graph is not None

    def test_build_lightweight_is_callable(self):
        builder = GraphBuilder(Settings())
        graph = builder.build_lightweight()
        assert callable(graph.invoke)

    def test_compiled_graph_has_entry_node(self, lightweight_graph):
        node_names = set(lightweight_graph.get_graph().nodes.keys())
        assert "entry" in node_names

    def test_compiled_graph_has_visual_agent_node(self, lightweight_graph):
        node_names = set(lightweight_graph.get_graph().nodes.keys())
        assert "visual_agent" in node_names

    def test_compiled_graph_has_retrieval_agent_node(self, lightweight_graph):
        node_names = set(lightweight_graph.get_graph().nodes.keys())
        assert "retrieval_agent" in node_names

    def test_compiled_graph_has_supervisor_node(self, lightweight_graph):
        node_names = set(lightweight_graph.get_graph().nodes.keys())
        assert "supervisor" in node_names

    def test_compiled_graph_has_output_formatter_node(self, lightweight_graph):
        node_names = set(lightweight_graph.get_graph().nodes.keys())
        assert "output_formatter" in node_names


class TestLightweightGraphExecution:
    def test_empty_state_does_not_raise(self, lightweight_graph):
        # entry_node handles missing image → visual_error → supervisor abstains
        result = lightweight_graph.invoke({})
        assert result is not None

    def test_empty_state_produces_final_answer(self, lightweight_graph):
        result = lightweight_graph.invoke({})
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    def test_empty_state_abstains_and_requires_human_review(self, lightweight_graph):
        result = lightweight_graph.invoke({})
        assert result["requires_human_review"] is True
        assert result["final_answer"].startswith("ABSTAIN:")

    def test_missing_image_abstains(self, lightweight_graph):
        state = {"question": "Is there pneumonia?", "retry_count": 0}
        result = lightweight_graph.invoke(state)
        assert result["requires_human_review"] is True

    def test_prepopulated_state_produces_final_answer(
        self, lightweight_graph, pre_populated_state
    ):
        result = lightweight_graph.invoke(pre_populated_state)
        assert "final_answer" in result
        assert len(result.get("final_answer", "")) > 0

    def test_prepopulated_with_supporting_evidence_answers(
        self, lightweight_graph, pre_populated_state
    ):
        # pre_populated_state: visual_confidence=0.92, pneumonia evidence,
        # closed question "Is there evidence of pneumonia?"
        # → supervisor finds agreement via question keywords → decision="answer"
        result = lightweight_graph.invoke(pre_populated_state)
        assert result.get("decision") == "answer"
        assert result["requires_human_review"] is False

    def test_prepopulated_state_has_citations(self, lightweight_graph, pre_populated_state):
        result = lightweight_graph.invoke(pre_populated_state)
        if result.get("decision") == "answer":
            assert isinstance(result.get("citations"), list)
            assert len(result["citations"]) > 0

    def test_no_evidence_high_confidence_routes_to_requery_then_abstain(
        self, lightweight_graph, base_state
    ):
        # base_state has no visual fields pre-populated.
        # Passthrough visual node returns visual_answer="" → supervisor sees
        # empty answer with low default confidence → abstain (Case E or VLM error)
        # In any case it should not crash and should produce a final answer.
        result = lightweight_graph.invoke(base_state)
        assert "final_answer" in result

    def test_result_has_all_expected_keys(self, lightweight_graph, pre_populated_state):
        result = lightweight_graph.invoke(pre_populated_state)
        for key in ("final_answer", "final_confidence", "citations", "requires_human_review"):
            assert key in result, f"Expected key '{key}' in result"

    def test_final_confidence_is_bounded(self, lightweight_graph, pre_populated_state):
        result = lightweight_graph.invoke(pre_populated_state)
        conf = result.get("final_confidence", 0.0)
        assert 0.0 <= conf <= 1.0


class TestReQueryLoop:
    """Verify the re-query loop (Case B/D) executes and terminates correctly.

    Scenario: high VLM confidence + no supporting evidence.
    Expected path:
        supervisor (pass 1) → re_query, retry_count 0→1
        routing → retrieval_agent (pass 2)
        retrieval_agent (pass 2) → still no evidence
        supervisor (pass 2) → retry_count=1 >= max_retries=1 → abstain
        output_formatter → final_answer starts with "ABSTAIN:"
    """

    def _high_confidence_no_evidence_state(self) -> dict:
        """State with high VLM confidence but no retrieved evidence."""
        return {
            "image": Image.new("RGB", (224, 224), color="gray"),
            "question": "What is the primary finding?",
            "answer_type": "open",
            "retry_count": 0,
            "visual_answer": "consolidation",
            "visual_confidence": 0.92,    # >= HIGH_CONFIDENCE (0.85) → Case A or B
            "visual_raw_output": "consolidation",
            "visual_model": "mock-vlm",
            "visual_error": "",
            "retrieval_query": "",
            "retrieved_evidence": [],     # no evidence → agreement_score=0.0 → Case B
            "retrieval_error": "",
        }

    def test_requery_loop_terminates_with_abstain(self, lightweight_graph):
        result = lightweight_graph.invoke(self._high_confidence_no_evidence_state())
        assert result["decision"] == "abstain"

    def test_requery_loop_requires_human_review(self, lightweight_graph):
        result = lightweight_graph.invoke(self._high_confidence_no_evidence_state())
        assert result["requires_human_review"] is True

    def test_requery_loop_final_answer_is_abstain_message(self, lightweight_graph):
        result = lightweight_graph.invoke(self._high_confidence_no_evidence_state())
        assert result["final_answer"].startswith("ABSTAIN:")

    def test_requery_loop_increments_retry_count(self, lightweight_graph):
        # retry_count=1 in the final state proves the supervisor emitted re_query
        # exactly once (max_retries=1) before escalating to abstain.
        result = lightweight_graph.invoke(self._high_confidence_no_evidence_state())
        assert result.get("retry_count", 0) == 1

    def test_requery_loop_does_not_exceed_max_retries(self, lightweight_graph):
        from radiology_vqa.config import settings
        result = lightweight_graph.invoke(self._high_confidence_no_evidence_state())
        assert result.get("retry_count", 0) <= settings.supervisor_max_retries

    def test_requery_loop_final_confidence_is_zero(self, lightweight_graph):
        # Abstain path always yields 0.0 confidence.
        result = lightweight_graph.invoke(self._high_confidence_no_evidence_state())
        assert result.get("final_confidence", -1.0) == 0.0

    def test_requery_loop_citations_empty(self, lightweight_graph):
        # Abstain path yields no citations.
        result = lightweight_graph.invoke(self._high_confidence_no_evidence_state())
        assert result.get("citations") == []

    def test_moderate_confidence_no_evidence_also_abstains_after_retry(
        self, lightweight_graph
    ):
        # Case D: moderate confidence (>= LOW_CONF but < HIGH_CONF) + no evidence.
        # Same loop behaviour as Case B.
        state = self._high_confidence_no_evidence_state()
        state["visual_confidence"] = 0.70    # moderate (0.55 <= 0.70 < 0.85)
        result = lightweight_graph.invoke(state)
        assert result["decision"] == "abstain"
        assert result.get("retry_count", 0) == 1

    def test_requery_does_not_trigger_when_evidence_present(
        self, lightweight_graph, sample_evidence
    ):
        # Control: high confidence + supporting evidence → answer (no re-query).
        state = self._high_confidence_no_evidence_state()
        state["question"] = "Is there evidence of pneumonia?"
        state["answer_type"] = "open"
        state["visual_answer"] = "pneumonia"        # keyword matches evidence
        state["retrieved_evidence"] = sample_evidence
        result = lightweight_graph.invoke(state)
        assert result["decision"] == "answer"
        assert result.get("retry_count", 0) == 0    # never incremented
