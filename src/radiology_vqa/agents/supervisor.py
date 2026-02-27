"""Supervisor node — fuses VLM output with RAG evidence and decides: answer / re_query / abstain.

Decision logic is entirely rule-based (deterministic). No LLM calls.

Thresholds are calibrated to the Phase 3 LLaVA v1.6 baseline on VQA-RAD (451 samples):
  - Correct answers:  mean confidence 0.896
  - Incorrect answers: mean confidence 0.815
  - HIGH_CONFIDENCE=0.85 sits between these two means.
  - LOW_CONFIDENCE=0.55 is a conservative lower bound for clinical use.

Agreement scoring (Phase 6B-3) uses PubMedBERT cosine similarity instead of keyword
matching. The same S-PubMedBert-MS-MARCO model used for FAISS retrieval is reused here,
meaning no additional model is loaded. Cosine similarity handles synonyms and paraphrases
that keyword matching misses ("tumor"/"neoplasm", "consolidation"/"opacity",
"cardiac silhouette"/"cardiomegaly"), directly targeting the 61 over-abstentions
observed in the Phase 6A evaluation.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from radiology_vqa.agents.state import AgentState

if TYPE_CHECKING:
    from radiology_vqa.rag.embedder import Embedder

logger = logging.getLogger(__name__)

# Module-level defaults — overridden at runtime by settings
HIGH_CONFIDENCE: float = 0.85
LOW_CONFIDENCE: float = 0.55
EVIDENCE_SUPPORT_THRESHOLD: float = 0.4
MIN_SUPPORTING_EVIDENCE: int = 1

# Module-level embedder singleton — loaded lazily on first agreement computation.
# Reuses the same model (S-PubMedBert-MS-MARCO) already loaded by the Retriever,
# so in production the process-level model cache means no duplicate weights.
_embedder = None


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
    agreement_method: str = getattr(settings, "agreement_method", "embedding")
    if agreement_method == "keyword":
        agreement_score, supporting = _compute_agreement_keyword(
            visual_answer, evidence, question, answer_type, support_threshold
        )
    else:
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


_STOP_WORDS = frozenset({
    "is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "with", "this",
    "that", "what", "which", "where", "how", "why", "when", "who", "there",
    "any", "show", "can", "be", "it", "from",
})


def _keyword_in_text(kw: str, text: str) -> bool:
    """Check if keyword (or its singular stem) appears in text."""
    text_lower = text.lower()
    if kw in text_lower:
        return True
    if kw.endswith("s") and len(kw) >= 5 and kw[:-1] in text_lower:
        return True
    return False


def _extract_keywords(text: str) -> list[str]:
    """Extract non-stopword tokens of length >= 3."""
    tokens = []
    for word in text.lower().split():
        token = word.strip("?.!,;:")
        if token and token not in _STOP_WORDS and len(token) >= 3:
            tokens.append(token)
    return tokens


def _compute_agreement_keyword(
    visual_answer: str,
    evidence: list[dict],
    question: str,
    answer_type: str,
    support_threshold: float,
) -> tuple[float, list[dict]]:
    """Phase 5 keyword-based agreement (used for ablation configs 2 and 4).

    Dual-signal: closed/yes-no → extract from question; open → extract from visual_answer.
    Falls back to visual_answer tokens if question yields empty keyword set.
    """
    if not evidence:
        return 0.0, []

    va_norm = visual_answer.strip().lower()
    use_question = (answer_type == "closed") or (va_norm in ("yes", "no"))
    keywords = _extract_keywords(question if use_question else visual_answer)
    if not keywords and use_question:
        keywords = _extract_keywords(visual_answer)
    if not keywords:
        return 0.0, []

    candidates = [item for item in evidence if item.get("score", 0.0) >= support_threshold]
    if not candidates:
        return 0.0, []

    supporting = []
    for item in candidates:
        item_text = f"{item.get('text', '')} {item.get('entity_name', '')}".lower()
        if any(_keyword_in_text(kw, item_text) for kw in keywords):
            supporting.append(item)

    score = len(supporting) / len(evidence)
    logger.debug(
        "Keyword agreement: keywords=%r supporting=%d/%d score=%.3f",
        keywords[:5], len(supporting), len(evidence), score,
    )
    return score, supporting


def _get_embedder():
    """Lazy-load and cache the sentence embedding model.

    Reuses the same model identifier (S-PubMedBert-MS-MARCO) as the FAISS
    retriever, so in production the sentence-transformers process-level cache
    avoids loading duplicate weights into GPU memory.
    """
    global _embedder
    if _embedder is None:
        from radiology_vqa.rag.embedder import Embedder
        _embedder = Embedder()
    return _embedder


def _compute_agreement(
    visual_answer: str,
    evidence: list[dict],
    question: str,
    answer_type: str,
    support_threshold: float,
    embedder=None,
) -> tuple[float, list[dict]]:
    """Compute semantic agreement using PubMedBERT cosine similarity.

    Replaces Phase 5 keyword matching. Handles synonyms and paraphrases that
    keyword matching misses ("tumor"/"neoplasm", "consolidation"/"opacity",
    "cardiac silhouette"/"cardiomegaly").

    Query construction (same dual-signal strategy as the former keyword approach):
        Closed / yes-no: embed the full question — the visual_answer carries no
            medical signal when it is "yes" or "no".
        Open: embed the visual_answer — the actual medical term predicted by the VLM.

    An evidence item is "supporting" when:
        cosine_sim(query_embedding, evidence_embedding) >= semantic_threshold
        AND item.score >= support_threshold (retrieval quality filter, unchanged).

    The evidence embedding concatenates text + entity_name so that short entity-name
    fields ("Pneumonia") and full sentence fields both contribute to the match.

    Agreement score = n_supporting / n_total_evidence, preserving the same
    normalisation formula and the `> 0` routing contract used by supervisor_node.

    Args:
        visual_answer:    VLM prediction string.
        evidence:         List of evidence dicts from retrieval_agent_node.
        question:         Original question string.
        answer_type:      "open" or "closed".
        support_threshold: Minimum retrieval score (0–1) to consider an item.
        embedder:         Optional Embedder instance (injected in tests / benchmarks).
                          If None, the module-level singleton is used.

    Returns:
        (agreement_score, supporting_evidence_items)
    """
    if not evidence:
        return 0.0, []

    # Dual-signal: closed questions and yes/no answers → embed the question
    va_norm = visual_answer.strip().lower()
    use_question = (answer_type == "closed") or (va_norm in ("yes", "no"))
    query_text = question.strip() if use_question else visual_answer.strip()

    if not query_text:
        return 0.0, []

    from radiology_vqa.config import settings

    semantic_threshold: float = getattr(
        settings, "supervisor_semantic_threshold", 0.5
    )

    # Filter by retrieval quality first (same gate as keyword approach)
    candidates = [item for item in evidence if item.get("score", 0.0) >= support_threshold]
    if not candidates:
        return 0.0, []

    emb = embedder if embedder is not None else _get_embedder()

    # query_vec: shape (dim,), L2-normalised
    # embed_query may return (1, dim) or (dim,) depending on the backend — flatten to 1D
    query_vec = emb.embed_query(query_text).flatten()

    # Concatenate text + entity_name for richer evidence representation.
    # Both fields are present in every evidence dict produced by retrieval_agent_node.
    evidence_texts = [
        f"{item.get('text', '')} {item.get('entity_name', '')}".strip()
        for item in candidates
    ]
    # evidence_vecs: shape (n, dim), L2-normalised
    evidence_vecs = emb.embed_texts(evidence_texts)

    # Cosine similarity = dot product of L2-normalised vectors
    sims: np.ndarray = evidence_vecs @ query_vec  # shape (n,)

    supporting = [
        item
        for item, sim in zip(candidates, sims)
        if float(sim) >= semantic_threshold
    ]

    # Normalise by total evidence count (preserves routing contract: score ∈ [0, 1])
    score = len(supporting) / len(evidence)

    logger.debug(
        "Embedding agreement: query=%r sims=[%s] supporting=%d/%d score=%.3f",
        query_text[:60],
        ", ".join(f"{s:.3f}" for s in sims),
        len(supporting),
        len(evidence),
        score,
    )

    return score, supporting
