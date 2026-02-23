"""Conditional routing functions for the multi-agent LangGraph."""

import logging

from radiology_vqa.agents.state import AgentState

logger = logging.getLogger(__name__)


def route_after_supervisor(state: AgentState) -> str:
    """Determine the next node after the supervisor makes a decision.

    Returns one of: "output_formatter" | "retrieval_agent"

    Decision logic:
        "answer"   → output_formatter
        "re_query" → retrieval_agent  (unless safety bound exceeded)
        "abstain"  → output_formatter
        anything else → output_formatter  (safe default)

    Safety bound:
        If retry_count > supervisor_max_retries with decision="re_query",
        force output_formatter regardless of the supervisor's decision.
        This prevents infinite loops even if the supervisor has a bug that
        keeps emitting "re_query".

    Note: this is the ONLY place that enforces the re-query loop limit.
    """
    from radiology_vqa.config import settings

    max_retries: int = getattr(settings, "supervisor_max_retries", 1)
    decision: str = state.get("decision", "")
    retry_count: int = state.get("retry_count", 0)

    if decision == "answer":
        logger.debug("Routing: answer → output_formatter")
        return "output_formatter"

    if decision == "re_query":
        if retry_count > max_retries:
            logger.warning(
                "Routing: re_query with retry_count=%d > max_retries=%d — "
                "forcing output_formatter (safety bound)",
                retry_count, max_retries,
            )
            return "output_formatter"
        logger.debug("Routing: re_query (retry_count=%d) → retrieval_agent", retry_count)
        return "retrieval_agent"

    if decision == "abstain":
        logger.debug("Routing: abstain → output_formatter")
        return "output_formatter"

    # Unknown decision — safe default
    logger.error(
        "Routing: unknown decision=%r (retry_count=%d) — defaulting to output_formatter",
        decision, retry_count,
    )
    return "output_formatter"
