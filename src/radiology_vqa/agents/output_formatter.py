"""Output Formatter node — packages supervisor decision into final structured output."""

import logging

from radiology_vqa.agents.state import AgentState, SystemOutput

logger = logging.getLogger(__name__)

_ABSTAIN_MESSAGE = "ABSTAIN: Unable to provide a reliable answer for this case."


def output_formatter_node(state: AgentState) -> AgentState:
    """Package the supervisor's decision into final structured output.

    Reads from state:
        decision, grounded_answer, grounded_confidence,
        retrieved_evidence, decision_reasoning

    Writes to state:
        final_answer, final_confidence, citations,
        requires_human_review, output_reasoning

    If decision == "answer":
        Returns the grounded answer with top-3 supporting citations.
    If decision == "abstain" (or unexpected "re_query"):
        Returns an ABSTAIN message with requires_human_review=True.
    """
    decision: str = state.get("decision", "abstain")
    grounded_answer: str = state.get("grounded_answer", "")
    grounded_confidence: float = state.get("grounded_confidence", 0.0)
    evidence: list[dict] = state.get("retrieved_evidence", [])
    decision_reasoning: str = state.get("decision_reasoning", "")

    if decision == "answer":
        # Sort evidence by score descending and take top 3 as citations
        sorted_evidence = sorted(
            evidence, key=lambda e: e.get("score", 0.0), reverse=True
        )
        citations = [
            {
                "source_type": e.get("source_type", ""),
                "entity_name": e.get("entity_name", ""),
                "attribute": e.get("attribute", ""),
                "text": e.get("text", ""),
                "score": e.get("score", 0.0),
            }
            for e in sorted_evidence[:3]
        ]
        updates: dict = {
            "final_answer": grounded_answer,
            "final_confidence": grounded_confidence,
            "citations": citations,
            "requires_human_review": False,
            "output_reasoning": decision_reasoning,
        }
    else:
        # "abstain" or unexpected "re_query" reaching the formatter
        updates = {
            "final_answer": _ABSTAIN_MESSAGE,
            "final_confidence": 0.0,
            "citations": [],
            "requires_human_review": True,
            "output_reasoning": decision_reasoning,
        }

    logger.info(
        "Output formatter: decision=%s final_answer=%.60r requires_review=%s",
        decision,
        updates["final_answer"],
        updates["requires_human_review"],
    )

    return {**state, **updates}


def format_system_output(state: AgentState) -> SystemOutput:
    """Convert final AgentState into the SystemOutput Pydantic model.

    This is the clean interface for Phase 7's API.
    """
    return SystemOutput(
        answer=state.get("final_answer") or _ABSTAIN_MESSAGE,
        confidence=state.get("final_confidence", 0.0),
        citations=state.get("citations", []),
        requires_human_review=state.get("requires_human_review", True),
        reasoning=state.get("output_reasoning", ""),
        decision=state.get("decision", "abstain"),
        visual_answer=state.get("visual_answer", ""),
        retrieval_query=state.get("retrieval_query", ""),
    )
