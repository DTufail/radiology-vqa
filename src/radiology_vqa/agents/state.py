"""Shared state TypedDict and output models for the multi-agent pipeline."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """Shared state that flows through all agent nodes.

    total=False means all keys are optional — nodes progressively populate state.
    Each node reads what it needs and writes what it produces.
    """

    # ── Input (set at entry) ──────────────────────────────────────────
    image: Any                      # PIL.Image.Image
    question: str
    answer_type: str                # "open" or "closed" (if known)

    # ── Visual Agent output ───────────────────────────────────────────
    visual_answer: str              # concise VLM answer
    visual_confidence: float        # 0.0 - 1.0 from VLM logits
    visual_raw_output: str          # full VLM output (for context)
    visual_model: str               # which VLM produced this
    visual_error: str               # empty if no error

    # ── Retrieval Agent output ────────────────────────────────────────
    retrieval_query: str            # the query actually sent to retriever
    retrieved_evidence: list[dict]  # [{text, score, source_type, entity_name, attribute, rank}]
    retrieval_error: str            # empty if no error

    # ── Supervisor output ─────────────────────────────────────────────
    decision: str                   # "answer" | "re_query" | "abstain"
    decision_reasoning: str         # human-readable explanation
    agreement_score: float          # 0.0 - 1.0, how well VLM and RAG agree
    grounded_answer: str            # final answer after RAG grounding
    grounded_confidence: float      # combined confidence after RAG

    # ── Output Formatter output ───────────────────────────────────────
    final_answer: str
    final_confidence: float
    citations: list[dict]           # [{source_type, entity_name, attribute, text, score}]
    requires_human_review: bool
    output_reasoning: str           # explanation of the answer or abstention reason

    # ── Control flow ──────────────────────────────────────────────────
    retry_count: int                # starts at 0, incremented by supervisor on re_query


class SystemOutput(BaseModel):
    """The final structured output returned to the caller.

    This is what Phase 7's API will serialize and return.
    """

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[dict]
    requires_human_review: bool
    reasoning: str
    decision: str                   # "answer" or "abstain"
    visual_answer: str              # what the VLM said (for transparency)
    retrieval_query: str            # what was retrieved (for debugging)
