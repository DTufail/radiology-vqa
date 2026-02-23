"""Supervisor node — fuses VLM output with RAG evidence and decides: answer / re_query / abstain.

Decision logic is entirely rule-based (deterministic). No LLM calls.

Thresholds are calibrated to the Phase 3 LLaVA v1.6 baseline on VQA-RAD (451 samples):
  - Correct answers:  mean confidence 0.896
  - Incorrect answers: mean confidence 0.815
  - HIGH_CONFIDENCE=0.85 sits between these two means.
  - LOW_CONFIDENCE=0.55 is a conservative lower bound for clinical use.

Keyword-based agreement scoring is deliberately simple. Phase 6 can upgrade to
embedding-based agreement (semantic similarity) to catch synonyms like
"tumor" / "neoplasm" or "consolidation" / "opacity".
"""

import logging
import re

from radiology_vqa.agents.state import AgentState

logger = logging.getLogger(__name__)

# Module-level defaults — overridden at runtime by settings
HIGH_CONFIDENCE: float = 0.85
LOW_CONFIDENCE: float = 0.55
EVIDENCE_SUPPORT_THRESHOLD: float = 0.4
MIN_SUPPORTING_EVIDENCE: int = 1

# Function words to skip when extracting medical terms from questions/answers
_STOP_WORDS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been",
    "the", "a", "an", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "by", "from",
    "there", "this", "that", "these", "those",
    "what", "which", "who", "how", "when", "where",
    "does", "do", "did", "has", "have", "had",
    "not", "any", "some", "its",
})


def supervisor_node(state: AgentState) -> AgentState:
    """Fuse VLM output with retrieved evidence and decide: answer, re_query, or abstain.

    Reads from state:
        visual_answer, visual_confidence, visual_error,
        retrieved_evidence, retrieval_error,
        retry_count, question, answer_type

    Writes to state:
        decision, decision_reasoning, agreement_score,
        grounded_answer, grounded_confidence
        (also increments retry_count when decision == "re_query")

    Decision cases:
        A: high confidence + evidence supports  → answer
        B: high confidence + no evidence        → re_query (retry=0) or abstain
        C: moderate confidence + evidence       → answer
        D: moderate confidence + no evidence    → re_query (retry=0) or abstain
        E: low confidence                       → abstain (regardless of evidence)
    """
    from radiology_vqa.config import settings

    high_conf: float = getattr(settings, "supervisor_high_confidence", HIGH_CONFIDENCE)
    low_conf: float = getattr(settings, "supervisor_low_confidence", LOW_CONFIDENCE)
    support_threshold: float = getattr(
        settings, "supervisor_evidence_threshold", EVIDENCE_SUPPORT_THRESHOLD
    )
    max_retries: int = getattr(settings, "supervisor_max_retries", 1)

    visual_error: str = state.get("visual_error", "")
    retrieval_error: str = state.get("retrieval_error", "")
    visual_confidence: float = state.get("visual_confidence", 0.0)
    visual_answer: str = state.get("visual_answer", "")
    evidence: list[dict] = state.get("retrieved_evidence", [])
    retry_count: int = state.get("retry_count", 0)
    question: str = state.get("question", "")
    answer_type: str = state.get("answer_type", "open")

    # ── Step 1: Error handling ─────────────────────────────────────────────────
    if visual_error:
        if retrieval_error:
            reasoning = "Both VLM and retrieval failed — cannot produce a grounded answer."
        else:
            reasoning = "VLM failed — cannot answer visual questions without vision."
        logger.info("Supervisor: abstain (visual_error=%r)", visual_error)
        return {
            **state,
            "decision": "abstain",
            "decision_reasoning": reasoning,
            "agreement_score": 0.0,
            "grounded_answer": "",
            "grounded_confidence": 0.0,
        }

    # ── Step 2: Compute agreement ──────────────────────────────────────────────
    agreement_score, supporting = _compute_agreement(
        visual_answer, evidence, question, answer_type, support_threshold
    )
    updates: dict = {"agreement_score": agreement_score}

    # ── Step 3: Route ──────────────────────────────────────────────────────────
    if visual_confidence >= high_conf:
        if agreement_score > 0:
            # Case A: high confidence + supporting evidence
            updates["decision"] = "answer"
            updates["grounded_answer"] = visual_answer
            updates["grounded_confidence"] = visual_confidence * (0.5 + 0.5 * agreement_score)
            updates["decision_reasoning"] = (
                f"VLM confident (confidence={visual_confidence:.3f}) with "
                f"{len(supporting)} supporting evidence item(s) "
                f"(agreement={agreement_score:.3f})."
            )
            logger.info(
                "Supervisor [Case A]: answer=%r conf=%.3f agreement=%.3f",
                visual_answer, visual_confidence, agreement_score,
            )
        else:
            # Case B: high confidence but no supporting evidence
            if retry_count < max_retries:
                updates["decision"] = "re_query"
                updates["retry_count"] = retry_count + 1
                updates["decision_reasoning"] = (
                    f"VLM confident (confidence={visual_confidence:.3f}) but no supporting "
                    "evidence found. Re-querying with different terms."
                )
                logger.info(
                    "Supervisor [Case B]: re_query conf=%.3f retry=%d",
                    visual_confidence, retry_count + 1,
                )
            else:
                updates["decision"] = "abstain"
                updates["decision_reasoning"] = (
                    f"VLM confident (confidence={visual_confidence:.3f}) but no supporting "
                    "evidence after retry. Cannot ground answer in medical knowledge."
                )
                logger.info(
                    "Supervisor [Case B]: abstain after retry conf=%.3f", visual_confidence
                )
            updates["grounded_answer"] = visual_answer
            updates["grounded_confidence"] = 0.0

    elif visual_confidence >= low_conf:
        if agreement_score > 0:
            # Case C: moderate confidence + supporting evidence
            updates["decision"] = "answer"
            updates["grounded_answer"] = visual_answer
            updates["grounded_confidence"] = visual_confidence * (0.3 + 0.7 * agreement_score)
            updates["decision_reasoning"] = (
                f"VLM moderate confidence ({visual_confidence:.3f}) supported by "
                f"{len(supporting)} evidence item(s) (agreement={agreement_score:.3f})."
            )
            logger.info(
                "Supervisor [Case C]: answer=%r conf=%.3f agreement=%.3f",
                visual_answer, visual_confidence, agreement_score,
            )
        else:
            # Case D: moderate confidence, no supporting evidence
            if retry_count < max_retries:
                updates["decision"] = "re_query"
                updates["retry_count"] = retry_count + 1
                updates["decision_reasoning"] = (
                    f"VLM moderate confidence ({visual_confidence:.3f}) with no supporting "
                    "evidence. Re-querying."
                )
                logger.info(
                    "Supervisor [Case D]: re_query conf=%.3f retry=%d",
                    visual_confidence, retry_count + 1,
                )
            else:
                updates["decision"] = "abstain"
                updates["decision_reasoning"] = (
                    f"VLM moderate confidence ({visual_confidence:.3f}) with no supporting "
                    "evidence after retry. Abstaining."
                )
                logger.info(
                    "Supervisor [Case D]: abstain after retry conf=%.3f", visual_confidence
                )
            updates["grounded_answer"] = visual_answer
            updates["grounded_confidence"] = 0.0

    else:
        # Case E: low confidence — abstain regardless of evidence
        updates["decision"] = "abstain"
        updates["grounded_answer"] = visual_answer
        updates["grounded_confidence"] = 0.0
        updates["decision_reasoning"] = (
            f"VLM confidence too low for clinical answer "
            f"({visual_confidence:.3f} < {low_conf})."
        )
        logger.info("Supervisor [Case E]: abstain low_conf=%.3f", visual_confidence)

    return {**state, **updates}


def _compute_agreement(
    visual_answer: str,
    evidence: list[dict],
    question: str,
    answer_type: str,
    support_threshold: float,
) -> tuple[float, list[dict]]:
    """Compute keyword-based agreement between VLM answer and retrieved evidence.

    Returns (agreement_score, supporting_evidence_items).

    Matching strategy:
        Open questions: tokenise visual_answer (words > 2 chars), check if any word
            appears (case-insensitive) in evidence text or entity_name.
        Closed (yes/no) questions: extract medical terms from the question instead
            (the visual_answer is "yes"/"no" which carries no medical content).

    Limitation: keyword matching misses semantic equivalents ("tumor" vs "neoplasm").
    Phase 6 can replace this with embedding cosine similarity.
    """
    if not evidence:
        return 0.0, []

    # Build keyword set
    if answer_type == "closed":
        # yes/no answers have no medical content — extract terms from the question
        words = re.findall(r"[a-z]+", question.lower())
        keywords = {w for w in words if len(w) > 2 and w not in _STOP_WORDS}
    else:
        words = re.findall(r"[a-z]+", visual_answer.lower())
        keywords = {w for w in words if len(w) > 2}

    if not keywords:
        return 0.0, []

    supporting: list[dict] = []
    for item in evidence:
        if item.get("score", 0.0) < support_threshold:
            continue
        text_lower = item.get("text", "").lower()
        entity_lower = item.get("entity_name", "").lower()
        if any(kw in text_lower or kw in entity_lower for kw in keywords):
            supporting.append(item)

    # Normalise by total evidence count (not just above-threshold candidates)
    score = len(supporting) / len(evidence)
    return score, supporting
