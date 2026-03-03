"""Metrics specific to the multi-agent pipeline with selective prediction.

These metrics evaluate the AGENT BEHAVIOR — not just answer quality, but
decision quality (when to answer, when to abstain, when to re-query) and
grounding quality (are citations relevant?).

Standard VQA papers don't report these. They're unique to our architecture.
"""

import logging
from typing import Sequence

from radiology_vqa.evaluation.metrics import normalize_answer

logger = logging.getLogger(__name__)


def abstention_rate(decisions: Sequence[str]) -> float:
    """Fraction of samples where the agent abstained.

    A high abstention rate means the system is conservative.
    A low rate means it's answering most questions.
    Neither is inherently good — it depends on accuracy-when-answered.

    Returns 0.0 if input is empty.
    """
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d == "abstain") / len(decisions)


def accuracy_when_answered(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    decisions: Sequence[str],
) -> float:
    """Accuracy computed ONLY on samples where the agent answered.

    Filters to decision != "abstain", then computes exact match accuracy.

    This is the selective prediction metric. The key insight:
    - VLM-only overall accuracy: 41.2%  (no abstention option)
    - Agent accuracy-when-answered: should be HIGHER than 41.2%
      because the agent abstains on uncertain cases.

    If accuracy-when-answered < VLM-only accuracy, the agent is making
    WORSE predictions on the cases it chooses to answer. That means the
    supervisor's routing logic is broken.

    Returns 0.0 if agent answered zero questions (all abstained).
    """
    answered_idx = [i for i, d in enumerate(decisions) if d != "abstain"]
    if not answered_idx:
        logger.warning("accuracy_when_answered: no answered samples; returning 0.0")
        return 0.0
    correct = sum(
        1
        for i in answered_idx
        if normalize_answer(predictions[i]) == normalize_answer(ground_truths[i])
    )
    return correct / len(answered_idx)


def correct_abstention_rate(
    vlm_only_predictions: Sequence[str],
    ground_truths: Sequence[str],
    decisions: Sequence[str],
) -> float:
    """Of samples the agent abstained on, what fraction would the VLM have gotten wrong?

    This measures abstention quality. High value = the system correctly
    identifies cases where the VLM would fail.

    Calculation:
    1. Filter to samples where decision == "abstain"
    2. Check if vlm_only_prediction matches ground_truth for each
    3. Return fraction where VLM would have been WRONG

    Edge cases:
    - Zero abstentions → return 0.0 with logged warning
    - VLM-only predictions must be aligned by sample index
    """
    abstained_idx = [i for i, d in enumerate(decisions) if d == "abstain"]
    if not abstained_idx:
        logger.warning("correct_abstention_rate: no abstentions; returning 0.0")
        return 0.0
    vlm_wrong = sum(
        1
        for i in abstained_idx
        if normalize_answer(vlm_only_predictions[i]) != normalize_answer(ground_truths[i])
    )
    return vlm_wrong / len(abstained_idx)


def re_query_rate(decisions: Sequence[str]) -> float:
    """Fraction of samples with decision == 're_query' (intermediate decisions).

    Note: this function receives a pre-computed list of whether each sample
    went through a re-query path. Use re_query_rate_from_counts for
    retry_count-based computation.

    Returns 0.0 if input is empty.
    """
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d == "re_query") / len(decisions)


def re_query_rate_from_counts(retry_counts: Sequence[int]) -> float:
    """Fraction of samples with retry_count > 0."""
    if not retry_counts:
        return 0.0
    return sum(1 for c in retry_counts if c > 0) / len(retry_counts)


def grounding_improvement(
    agent_predictions: Sequence[str],
    vlm_only_predictions: Sequence[str],
    ground_truths: Sequence[str],
    agent_decisions: Sequence[str],
) -> dict[str, int]:
    """Side-by-side comparison of agent vs VLM-only predictions.

    For each sample, categorize into one of:
    - "improved": agent correct AND vlm_only wrong (RAG helped)
    - "degraded": agent wrong AND vlm_only correct (RAG hurt)
    - "both_correct": both right (RAG didn't matter)
    - "both_wrong": both wrong (neither approach works)
    - "agent_abstained": agent abstained (separate category)

    For agent_abstained samples, additionally check vlm_only:
    - "abstain_vlm_correct": abstained but VLM would have been right (over-abstention)
    - "abstain_vlm_wrong": abstained and VLM also wrong (justified abstention)

    Returns:
    {
        "improved": int,
        "degraded": int,
        "both_correct": int,
        "both_wrong": int,
        "agent_abstained": int,
        "abstain_vlm_correct": int,
        "abstain_vlm_wrong": int,
        "net_improvement": int,  # improved - degraded
        "total": int,
    }

    The net_improvement number is the headline: positive = RAG grounding
    helps, negative = RAG grounding hurts, zero = wash.
    """
    improved = degraded = both_correct = both_wrong = 0
    agent_abstained = abstain_vlm_correct = abstain_vlm_wrong = 0

    for pred_a, pred_v, gt, dec in zip(
        agent_predictions, vlm_only_predictions, ground_truths, agent_decisions
    ):
        gt_norm = normalize_answer(gt)
        vlm_correct = normalize_answer(pred_v) == gt_norm

        if dec == "abstain":
            agent_abstained += 1
            if vlm_correct:
                abstain_vlm_correct += 1
            else:
                abstain_vlm_wrong += 1
            continue

        agent_correct = normalize_answer(pred_a) == gt_norm
        if agent_correct and not vlm_correct:
            improved += 1
        elif not agent_correct and vlm_correct:
            degraded += 1
        elif agent_correct and vlm_correct:
            both_correct += 1
        else:
            both_wrong += 1

    total = improved + degraded + both_correct + both_wrong + agent_abstained
    return {
        "improved": improved,
        "degraded": degraded,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "agent_abstained": agent_abstained,
        "abstain_vlm_correct": abstain_vlm_correct,
        "abstain_vlm_wrong": abstain_vlm_wrong,
        "net_improvement": improved - degraded,
        "total": total,
    }


def citation_relevance(
    per_sample_results: list,
    min_token_overlap: int = 1,
) -> float:
    """Fraction of answered samples that have at least one relevant citation.

    A citation is relevant if the retrieved document text shares at least
    min_token_overlap tokens with the question (case-insensitive, after
    stripping punctuation and stop words).

    This definition is source_type-agnostic — it works for KG documents,
    RadLex definitions, and QA pseudo-documents equally. It replaces the
    Phase 6 source_type allowlist which became stale when qa_vqarad and
    qa_slake were added to the index in Phase 8A.

    Backward-compatible with the Phase 5 calling convention:
        citation_relevance(citations_per_sample, ground_truths)
    where citations_per_sample is a list of list-of-dicts and ground_truths
    is a list of strings. In that case the legacy dict result is returned.

    Args:
        per_sample_results: List of PerSampleResult objects (new style), OR
                            list of list-of-dicts for legacy callers.
        min_token_overlap:  Minimum number of shared tokens to count as
                            relevant. Default 1 (at least one shared term).
                            When called in legacy style this is the ground-
                            truth list instead (type is detected at runtime).

    Returns:
        Float in [0, 1] (new style). 0.0 if no samples were answered.
        dict with citation_hit_rate etc. (legacy style, for backward compat).
    """
    # ── Legacy call detection ────────────────────────────────────────────────
    # Legacy signature: citation_relevance(citations_per_sample, ground_truths)
    # where ground_truths is a Sequence[str]. Detect by checking min_token_overlap.
    if isinstance(min_token_overlap, (list, tuple)):
        return _citation_relevance_legacy(
            citations_per_sample=per_sample_results,
            ground_truths=min_token_overlap,
        )

    # ── New implementation (Phase 8A) ────────────────────────────────────────
    # Minimal stop-word list — clinical terms must not be filtered
    _STOP = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "have", "has", "had", "will", "would",
        "can", "could", "should", "may", "might", "this", "that",
        "there", "in", "on", "at", "of", "to", "for", "and", "or",
        "not", "what", "which", "where", "how", "any", "with",
    })

    import re
    _PUNCT_RE = re.compile(r"[^a-z0-9\s]")

    def _tokens(text: str) -> frozenset[str]:
        cleaned = _PUNCT_RE.sub(" ", text.lower())
        return frozenset(t for t in cleaned.split() if t not in _STOP and len(t) > 1)

    answered = [r for r in per_sample_results if getattr(r, "decision", "answer") == "answer"]
    if not answered:
        return 0.0

    relevant_count = 0
    for result in answered:
        question_tokens = _tokens(result.question)
        citations = getattr(result, "citations", []) or []
        for citation in citations:
            # citation may be a string (text) or an object with .text attribute
            cite_text = citation if isinstance(citation, str) else getattr(citation, "text", "")
            cite_tokens = _tokens(cite_text)
            if len(question_tokens & cite_tokens) >= min_token_overlap:
                relevant_count += 1
                break  # only need one relevant citation per sample

    return relevant_count / len(answered)


def _citation_relevance_legacy(
    citations_per_sample: Sequence[list[dict]],
    ground_truths: Sequence[str],
) -> dict[str, float]:
    """Phase 5 citation relevance implementation — kept for backward compatibility.

    Checks whether any citation text contains ground-truth tokens.
    Called automatically by citation_relevance() when the second argument is
    a list, preserving backward compatibility with existing callers and tests.
    """
    total_cited = 0
    total_uncited = 0
    hit_count = 0
    relevant_per_sample: list[int] = []

    for citations, gt in zip(citations_per_sample, ground_truths):
        if not citations:
            total_uncited += 1
            continue

        total_cited += 1
        # GT tokens longer than 2 chars (exclude short tokens like "CT", "is", etc.)
        gt_tokens = {t for t in normalize_answer(gt).split() if len(t) > 2}

        relevant = 0
        for cit in citations:
            cit_text = cit.get("text", "").lower()
            if gt_tokens and any(tok in cit_text for tok in gt_tokens):
                relevant += 1

        relevant_per_sample.append(relevant)
        if relevant > 0:
            hit_count += 1

    citation_hit_rate = hit_count / total_cited if total_cited > 0 else 0.0
    mean_relevant = (
        sum(relevant_per_sample) / len(relevant_per_sample)
        if relevant_per_sample
        else 0.0
    )
    return {
        "citation_hit_rate": citation_hit_rate,
        "mean_relevant_citations": mean_relevant,
        "total_cited_samples": total_cited,
        "total_uncited_samples": total_uncited,
    }
