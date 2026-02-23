"""Retrieval Agent node — fetches medical evidence from the FAISS index."""

import logging
from typing import Any

from radiology_vqa.agents.state import AgentState

logger = logging.getLogger(__name__)


def retrieval_agent_node(
    state: AgentState,
    retriever: Any,
    top_k: int = 5,
) -> AgentState:
    """Retrieve medical evidence relevant to the question + visual findings.

    Reads from state:
        question, visual_answer, visual_error

    Writes to state:
        retrieval_query, retrieved_evidence, retrieval_error

    Query construction:
        If visual_answer is present and VLM succeeded:
            query = "{question} {visual_answer}"
        If VLM failed:
            query = question   (question-only fallback)

    Evidence is serialised as plain dicts (not RetrievalResult objects) so that
    AgentState remains JSON-serialisable and LangGraph-compatible.

    The retriever is injected, not created here. Errors go to state, never raised.
    """
    question = state.get("question", "")
    visual_answer = state.get("visual_answer", "")
    visual_error = state.get("visual_error", "")
    retry_count = state.get("retry_count", 0)

    # Build query.
    # First attempt (retry_count == 0): combine question with VLM answer for a focused query.
    # Re-query (retry_count > 0): use question only — different embedding vector, different
    #     retrieval ranking. Avoids the first-attempt query returning identical results.
    if retry_count > 0:
        query = question
    elif visual_answer and not visual_error:
        query = f"{question} {visual_answer}"
    else:
        query = question

    try:
        results = retriever.retrieve(query, top_k=top_k, min_score=0.0)
        evidence = [
            {
                "text": r.document.text,
                "score": r.score,
                "source_type": r.document.meta.source_type,
                "entity_name": r.document.meta.entity_name,
                "attribute": r.document.meta.attribute,
                "rank": r.rank,
            }
            for r in results
        ]
        top_score = results[0].score if results else 0.0
        logger.info(
            "Retrieval agent: query=%.100r → %d results (top_score=%.4f)",
            query,
            len(results),
            top_score,
        )
        updates: dict = {
            "retrieval_query": query,
            "retrieved_evidence": evidence,
            "retrieval_error": "",
        }
    except Exception as e:
        logger.error("Retrieval agent failed: %s", e, exc_info=True)
        updates = {
            "retrieval_query": query,
            "retrieved_evidence": [],
            "retrieval_error": str(e),
        }

    return {**state, **updates}
