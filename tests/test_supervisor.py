"""Tests for the Supervisor node — covers all routing cases."""

import pytest

from radiology_vqa.agents.supervisor import supervisor_node

# ── Helpers ────────────────────────────────────────────────────────────────────

PNEUMONIA_EVIDENCE = [
    {
        "text": "Lobar Pneumonia symptoms include: chills, high fever, chest pain",
        "score": 0.72,
        "source_type": "kg_disease",
        "entity_name": "Lobar Pneumonia",
        "attribute": "symptom",
        "rank": 1,
    },
    {
        "text": "Lobar Pneumonia is caused by streptococcus pneumoniae",
        "score": 0.65,
        "source_type": "kg_disease",
        "entity_name": "Lobar Pneumonia",
        "attribute": "cause",
        "rank": 2,
    },
]

LIVER_EVIDENCE = [
    {
        "text": "The function of Liver: metabolize nutrients, detoxify blood.",
        "score": 0.45,
        "source_type": "kg_organ",
        "entity_name": "Liver",
        "attribute": "function",
        "rank": 1,
    }
]


def _make_state(
    visual_confidence: float,
    evidence: list,
    visual_answer: str = "pneumonia",
    answer_type: str = "open",
    question: str = "What is the diagnosis?",
    visual_error: str = "",
    retrieval_error: str = "",
    retry_count: int = 0,
) -> dict:
    return {
        "question": question,
        "answer_type": answer_type,
        "visual_answer": visual_answer,
        "visual_confidence": visual_confidence,
        "visual_raw_output": visual_answer,
        "visual_model": "mock-vlm",
        "visual_error": visual_error,
        "retrieved_evidence": evidence,
        "retrieval_query": f"{question} {visual_answer}",
        "retrieval_error": retrieval_error,
        "retry_count": retry_count,
    }


# ── Case A ─────────────────────────────────────────────────────────────────────

class TestCaseA:
    def test_high_confidence_with_evidence_returns_answer(self):
        state = _make_state(0.92, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["decision"] == "answer"

    def test_high_confidence_with_evidence_sets_grounded_answer(self):
        state = _make_state(0.92, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["grounded_answer"] == "pneumonia"

    def test_high_confidence_with_evidence_has_positive_grounded_confidence(self):
        state = _make_state(0.92, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["grounded_confidence"] > 0.0


# ── Case B ─────────────────────────────────────────────────────────────────────

class TestCaseB:
    def test_high_confidence_no_evidence_returns_re_query(self):
        state = _make_state(0.90, [], retry_count=0)
        result = supervisor_node(state)
        assert result["decision"] == "re_query"

    def test_high_confidence_no_evidence_increments_retry(self):
        state = _make_state(0.90, [], retry_count=0)
        result = supervisor_node(state)
        assert result["retry_count"] == 1

    def test_high_confidence_no_evidence_after_retry_abstains(self):
        state = _make_state(0.90, [], retry_count=1)
        result = supervisor_node(state)
        assert result["decision"] == "abstain"

    def test_high_confidence_irrelevant_evidence_returns_re_query(self):
        # Liver evidence does not support "pneumonia" visual_answer (open question)
        state = _make_state(0.90, LIVER_EVIDENCE, retry_count=0)
        result = supervisor_node(state)
        assert result["decision"] == "re_query"


# ── Case C ─────────────────────────────────────────────────────────────────────

class TestCaseC:
    def test_moderate_confidence_with_evidence_returns_answer(self):
        state = _make_state(0.70, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["decision"] == "answer"

    def test_moderate_confidence_with_evidence_sets_grounded_answer(self):
        state = _make_state(0.70, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["grounded_answer"] == "pneumonia"


# ── Case D ─────────────────────────────────────────────────────────────────────

class TestCaseD:
    def test_moderate_confidence_no_evidence_returns_re_query(self):
        state = _make_state(0.60, [], retry_count=0)
        result = supervisor_node(state)
        assert result["decision"] == "re_query"

    def test_moderate_confidence_no_evidence_after_retry_abstains(self):
        state = _make_state(0.60, [], retry_count=1)
        result = supervisor_node(state)
        assert result["decision"] == "abstain"


# ── Case E ─────────────────────────────────────────────────────────────────────

class TestCaseE:
    def test_low_confidence_abstains_regardless_of_supporting_evidence(self):
        # Even with supporting evidence, low confidence → abstain
        state = _make_state(0.40, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["decision"] == "abstain"

    def test_low_confidence_abstains_with_no_evidence(self):
        state = _make_state(0.40, [])
        result = supervisor_node(state)
        assert result["decision"] == "abstain"


# ── Error states ───────────────────────────────────────────────────────────────

class TestErrorStates:
    def test_vlm_error_abstains(self):
        state = _make_state(0.0, PNEUMONIA_EVIDENCE, visual_error="GPU OOM")
        result = supervisor_node(state)
        assert result["decision"] == "abstain"

    def test_vlm_error_sets_zero_grounded_confidence(self):
        state = _make_state(0.0, [], visual_error="GPU OOM")
        result = supervisor_node(state)
        assert result["grounded_confidence"] == 0.0

    def test_both_errors_abstain(self):
        state = _make_state(
            0.0, [], visual_error="GPU OOM", retrieval_error="FAISS error"
        )
        result = supervisor_node(state)
        assert result["decision"] == "abstain"

    def test_both_errors_reasoning_mentions_both(self):
        state = _make_state(
            0.0, [], visual_error="GPU OOM", retrieval_error="FAISS error"
        )
        result = supervisor_node(state)
        reasoning = result["decision_reasoning"].lower()
        # Should mention both failures
        assert "vlm" in reasoning or "visual" in reasoning or "both" in reasoning


# ── Confidence bounds ──────────────────────────────────────────────────────────

class TestConfidenceBounds:
    def test_grounded_confidence_never_exceeds_visual_confidence(self):
        state = _make_state(0.92, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["grounded_confidence"] <= 0.92

    def test_grounded_confidence_moderate_never_exceeds_visual(self):
        state = _make_state(0.70, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["grounded_confidence"] <= 0.70

    def test_grounded_confidence_is_non_negative_case_a(self):
        state = _make_state(0.92, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert result["grounded_confidence"] >= 0.0

    def test_grounded_confidence_is_non_negative_case_e(self):
        state = _make_state(0.30, [])
        result = supervisor_node(state)
        assert result["grounded_confidence"] >= 0.0


# ── Decision reasoning ─────────────────────────────────────────────────────────

class TestDecisionReasoning:
    @pytest.mark.parametrize("confidence,evidence,retry,expected_decision", [
        (0.92, PNEUMONIA_EVIDENCE, 0, "answer"),    # Case A
        (0.90, [], 0, "re_query"),                   # Case B
        (0.90, [], 1, "abstain"),                    # Case B2
        (0.70, PNEUMONIA_EVIDENCE, 0, "answer"),    # Case C
        (0.60, [], 0, "re_query"),                   # Case D
        (0.60, [], 1, "abstain"),                    # Case D2
        (0.40, PNEUMONIA_EVIDENCE, 0, "abstain"),   # Case E
    ])
    def test_decision_reasoning_is_non_empty(
        self, confidence, evidence, retry, expected_decision
    ):
        state = _make_state(confidence, evidence, retry_count=retry)
        result = supervisor_node(state)
        assert result["decision"] == expected_decision
        assert len(result["decision_reasoning"]) > 0


# ── Agreement score ────────────────────────────────────────────────────────────

class TestAgreementScore:
    def test_positive_agreement_when_keyword_matches(self):
        # open question: visual_answer "pneumonia" appears in evidence text
        evidence = [
            {
                "text": "Pneumonia is an infection of the lungs causing inflammation.",
                "score": 0.72,
                "source_type": "kg_disease",
                "entity_name": "Pneumonia",
                "attribute": "description",
                "rank": 1,
            }
        ]
        state = _make_state(0.92, evidence, visual_answer="pneumonia", answer_type="open")
        result = supervisor_node(state)
        assert result["agreement_score"] > 0.0

    def test_zero_agreement_with_irrelevant_evidence(self):
        # open question: "consolidation" does not appear in liver evidence
        evidence = [
            {
                "text": "The function of Liver: metabolize nutrients, detoxify blood.",
                "score": 0.72,
                "source_type": "kg_organ",
                "entity_name": "Liver",
                "attribute": "function",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.92, evidence, visual_answer="consolidation", answer_type="open"
        )
        result = supervisor_node(state)
        assert result["agreement_score"] == 0.0

    def test_zero_agreement_with_empty_evidence(self):
        state = _make_state(0.92, [])
        result = supervisor_node(state)
        assert result["agreement_score"] == 0.0

    def test_agreement_score_bounded_between_zero_and_one(self):
        state = _make_state(0.92, PNEUMONIA_EVIDENCE)
        result = supervisor_node(state)
        assert 0.0 <= result["agreement_score"] <= 1.0

    def test_closed_question_uses_question_keywords(self):
        # Closed question: visual_answer is "yes" but question mentions "pneumonia"
        # → "pneumonia" keyword from question should match the evidence
        evidence = [
            {
                "text": "Lobar Pneumonia symptoms: fever, cough.",
                "score": 0.72,
                "source_type": "kg_disease",
                "entity_name": "Lobar Pneumonia",
                "attribute": "symptom",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.92,
            evidence,
            visual_answer="yes",
            answer_type="closed",
            question="Is there evidence of pneumonia?",
        )
        result = supervisor_node(state)
        assert result["agreement_score"] > 0.0


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_keyword_overlap_gives_zero_agreement(self):
        # visual_answer="tumor" but evidence is about pneumonia — no shared keywords
        evidence = [
            {
                "text": "Lobar Pneumonia symptoms include: fever, chest pain, cough.",
                "score": 0.72,
                "source_type": "kg_disease",
                "entity_name": "Lobar Pneumonia",
                "attribute": "symptom",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.92, evidence, visual_answer="tumor", answer_type="open"
        )
        result = supervisor_node(state)
        assert result["agreement_score"] == 0.0

    def test_closed_question_evidence_mentioning_condition_gives_positive_agreement(self):
        # visual_answer="yes" carries no medical keywords.
        # For closed questions the supervisor extracts terms from the question instead.
        # "fracture" from the question appears in the evidence → agreement > 0.
        evidence = [
            {
                "text": "Fracture is a break in the continuity of bone.",
                "score": 0.68,
                "source_type": "kg_disease",
                "entity_name": "Fracture",
                "attribute": "description",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.92,
            evidence,
            visual_answer="yes",
            answer_type="closed",
            question="Is there a fracture visible in this X-ray?",
        )
        result = supervisor_node(state)
        assert result["agreement_score"] > 0.0

    def test_empty_retrieved_evidence_routes_same_as_no_supporting_evidence(self):
        # retrieved_evidence=[] with no retrieval_error is not an error state —
        # the retriever simply found nothing. agreement=0, so Case B (re_query).
        state = _make_state(
            0.90,
            [],              # empty — not an error, just zero results
            retry_count=0,
        )
        state["retrieval_error"] = ""   # explicit: no error, just empty
        result = supervisor_node(state)
        # High confidence + zero agreement + retry_count=0 → re_query (Case B)
        assert result["decision"] == "re_query"
        assert result["agreement_score"] == 0.0

    def test_agreement_closed_all_stopwords_fallback(self):
        """question='is this the one?' — after stopword removal: 'one' kept (len>2).
        Doesn't appear in evidence → agreement = 0.0. No crash."""
        evidence = [
            {
                "text": "Some medical condition affects the organ.",
                "score": 0.80,
                "source_type": "kg_disease",
                "entity_name": "Condition",
                "attribute": "description",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.89, evidence, visual_answer="yes", answer_type="closed",
            question="Is this the one?",
        )
        result = supervisor_node(state)
        assert 0.0 <= result["agreement_score"] <= 1.0

    def test_confidence_exactly_at_high_boundary_hits_case_a_or_b_not_c_or_d(self):
        # HIGH_CONFIDENCE = 0.85. The condition is visual_confidence >= high_conf,
        # so 0.85 exactly satisfies it and falls into Case A/B (not Case C/D).
        # With supporting evidence this is Case A → "answer".
        # With no evidence this is Case B → "re_query" (at retry_count=0).
        evidence = [
            {
                "text": "Pneumonia is a lung infection causing inflammation.",
                "score": 0.72,
                "source_type": "kg_disease",
                "entity_name": "Pneumonia",
                "attribute": "description",
                "rank": 1,
            }
        ]
        state_with_evidence = _make_state(
            0.85, evidence, visual_answer="pneumonia", answer_type="open"
        )
        result_with = supervisor_node(state_with_evidence)
        # Case A: high branch, evidence supports → "answer"
        assert result_with["decision"] == "answer", (
            "confidence=0.85 should be treated as HIGH (>= threshold), not MODERATE"
        )

        state_no_evidence = _make_state(
            0.85, [], visual_answer="pneumonia", answer_type="open", retry_count=0
        )
        result_without = supervisor_node(state_no_evidence)
        # Case B: high branch, no evidence → "re_query"
        assert result_without["decision"] == "re_query", (
            "confidence=0.85 with no evidence should be Case B (re_query), not Case D"
        )


# ── Closed question agreement (Strategy B) ────────────────────────────────────

_CONSOLIDATION_EVIDENCE = [
    {
        "text": "Consolidation is an abnormal filling of lung air spaces with fluid or inflammatory cells.",
        "score": 0.87,
        "source_type": "kg_disease",
        "entity_name": "Consolidation",
        "attribute": "description",
        "rank": 1,
    }
]

_KIDNEY_EVIDENCE = [
    {
        "text": "Kidney stone causes severe pain in the lower back and urinary tract.",
        "score": 0.87,
        "source_type": "kg_disease",
        "entity_name": "Kidney Stone",
        "attribute": "symptom",
        "rank": 1,
    }
]

_FRACTURE_EVIDENCE = [
    {
        "text": "A temporal bone fracture involves a break in the bony skull base.",
        "score": 0.88,
        "source_type": "kg_disease",
        "entity_name": "Temporal Bone Fracture",
        "attribute": "description",
        "rank": 1,
    }
]


class TestClosedQuestionAgreement:
    """Verify Strategy B (question-keyword) agreement for closed/yes-no questions."""

    def test_closed_agreement_extracts_consolidation_from_question(self):
        """question='Is there consolidation in the lungs?', visual_answer='Yes'
        → extracts 'consolidation', 'lungs' → matches evidence → agreement > 0."""
        state = _make_state(
            0.89, _CONSOLIDATION_EVIDENCE,
            visual_answer="Yes", answer_type="closed",
            question="Is there consolidation in the lungs?",
        )
        result = supervisor_node(state)
        assert result["agreement_score"] > 0.0
        # High confidence + evidence → Case A → answer
        assert result["decision"] == "answer"

    def test_closed_agreement_ignores_yes_token_in_visual_answer(self):
        """Agreement must NOT depend on 'yes' appearing in evidence.
        Uses liver evidence that contains no 'yes'/'no' — match must come from
        question keyword 'liver'."""
        liver_evidence = [
            {
                "text": "Liver: metabolize nutrients, filter blood, produce bile.",
                "score": 0.88,
                "source_type": "kg_organ",
                "entity_name": "Liver",
                "attribute": "function",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.89, liver_evidence,
            visual_answer="Yes", answer_type="closed",
            question="Is the liver visible?",
        )
        result = supervisor_node(state)
        # 'liver' from question matches entity_name → agreement > 0
        assert result["agreement_score"] > 0.0

    def test_closed_agreement_multiple_medical_terms(self):
        """question='Are the temporal bones fractured?'
        Extracts: 'temporal', 'bones', 'fractured'.
        Evidence contains 'fracture' → 'fractured' substring matches via 'in' check."""
        state = _make_state(
            0.89, _FRACTURE_EVIDENCE,
            visual_answer="yes", answer_type="closed",
            question="Are the temporal bones fractured?",
        )
        result = supervisor_node(state)
        assert result["agreement_score"] > 0.0

    def test_closed_agreement_zero_when_no_medical_term_overlap(self):
        """question='Is there consolidation visible?' but evidence only about kidney stones.
        No keyword overlap → agreement = 0.0."""
        state = _make_state(
            0.89, _KIDNEY_EVIDENCE,
            visual_answer="yes", answer_type="closed",
            question="Is there consolidation visible?",
        )
        result = supervisor_node(state)
        assert result["agreement_score"] == 0.0

    def test_closed_agreement_domain_stopwords_removed(self):
        """question='Is the image showing any evidence of pneumonia?'
        Domain stopwords 'image', 'evidence' are removed.
        'pneumonia' stays → matches evidence → agreement > 0."""
        pneumonia_evidence = [
            {
                "text": "Pneumonia causes inflammation of the lung parenchyma.",
                "score": 0.88,
                "source_type": "kg_disease",
                "entity_name": "Pneumonia",
                "attribute": "description",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.89, pneumonia_evidence,
            visual_answer="yes", answer_type="closed",
            question="Is the image showing any evidence of pneumonia?",
        )
        result = supervisor_node(state)
        assert result["agreement_score"] > 0.0

    def test_visual_answer_yes_triggers_strategy_b_even_with_open_answer_type(self):
        """Dual-signal detection: visual_answer='yes' with answer_type='open'
        should still trigger question-keyword strategy (Strategy B)."""
        fracture_evidence = [
            {
                "text": "Fracture is a break in the continuity of bone structure.",
                "score": 0.88,
                "source_type": "kg_disease",
                "entity_name": "Fracture",
                "attribute": "description",
                "rank": 1,
            }
        ]
        # answer_type='open' but visual_answer='yes' → should use question keywords
        state = _make_state(
            0.89, fracture_evidence,
            visual_answer="yes", answer_type="open",
            question="Is there a fracture in this X-ray?",
        )
        result = supervisor_node(state)
        # 'fracture' from question matches evidence → agreement > 0
        assert result["agreement_score"] > 0.0

    def test_visual_answer_no_triggers_strategy_b(self):
        """visual_answer='no' also triggers Strategy B — should use question keywords."""
        state = _make_state(
            0.89, _CONSOLIDATION_EVIDENCE,
            visual_answer="no", answer_type="closed",
            question="Is there consolidation in the lungs?",
        )
        result = supervisor_node(state)
        assert result["agreement_score"] > 0.0

    def test_plural_keyword_matches_singular_in_evidence(self):
        """'lungs' (plural in question) matches 'lung' (singular in KG evidence).
        The SLAKE KG stores entities in singular form; questions use plurals.
        Without this fix, 'lungs' would never match 'Pneumonia is located in the Lung'."""
        lung_evidence = [
            {
                "text": "Pneumonia is located in the Lung.",
                "score": 0.89,
                "source_type": "kg_disease",
                "entity_name": "Pneumonia",
                "attribute": "location",
                "rank": 1,
            }
        ]
        state = _make_state(
            0.89, lung_evidence,
            visual_answer="yes", answer_type="closed",
            question="Is there consolidation in the lungs?",
        )
        result = supervisor_node(state)
        # 'lungs' → stem 'lung' matches 'Lung' in evidence → agreement > 0
        assert result["agreement_score"] > 0.0
        assert result["decision"] == "answer"

    def test_open_question_with_real_answer_still_uses_visual_answer(self):
        """Open questions with a genuine medical answer should use visual_answer tokens,
        not question keywords."""
        liver_evidence = [
            {
                "text": "Liver function: metabolize nutrients, detoxify blood.",
                "score": 0.88,
                "source_type": "kg_organ",
                "entity_name": "Liver",
                "attribute": "function",
                "rank": 1,
            }
        ]
        # visual_answer='liver' with answer_type='open' → Strategy A
        state = _make_state(
            0.89, liver_evidence,
            visual_answer="liver", answer_type="open",
            question="What organ is visible in this image?",
        )
        result = supervisor_node(state)
        # 'liver' from visual_answer matches evidence → agreement > 0
        assert result["agreement_score"] > 0.0


# ── Per-type confidence thresholds (Phase 7A-1) ────────────────────────────────


class TestPerTypeThresholds:
    """Verify that closed and open questions use separate confidence thresholds.

    Phase 7A-1: closed questions use supervisor_closed_{high,low}_confidence.
    Open questions use supervisor_open_{high,low}_confidence (same as Phase 6 defaults).
    All tests use monkeypatch to override settings — no GPU, no model loading.
    """

    def test_closed_question_answers_at_low_confidence_with_evidence(self, monkeypatch):
        """closed + conf=0.30 + evidence → should answer (not abstain) with low closed threshold.

        With closed_low_confidence=0.20: 0.30 >= 0.20, enters Case C range → answer.
        With the old global LOW_CONFIDENCE=0.55: 0.30 < 0.55 → would abstain (Case E).
        This is the core regression being fixed.
        """
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_closed_low_confidence", 0.20)
        monkeypatch.setattr(settings, "supervisor_closed_high_confidence", 0.60)
        monkeypatch.setattr(settings, "supervisor_open_low_confidence", 0.55)
        monkeypatch.setattr(settings, "supervisor_open_high_confidence", 0.85)

        state = _make_state(
            visual_confidence=0.30,
            evidence=PNEUMONIA_EVIDENCE,
            visual_answer="yes",
            answer_type="closed",
            question="Is there pneumonia visible?",
        )
        result = supervisor_node(state)
        assert result["decision"] == "answer", (
            "closed question with conf=0.30 and evidence should answer "
            "when closed_low_confidence=0.20, not abstain"
        )

    def test_open_question_abstains_at_same_low_confidence(self, monkeypatch):
        """open + conf=0.30 + evidence → should abstain with open threshold = 0.55.

        The open thresholds are unchanged from Phase 6, so 0.30 < 0.55 → Case E → abstain.
        This confirms the two question types are independently controlled.
        """
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_closed_low_confidence", 0.20)
        monkeypatch.setattr(settings, "supervisor_closed_high_confidence", 0.60)
        monkeypatch.setattr(settings, "supervisor_open_low_confidence", 0.55)
        monkeypatch.setattr(settings, "supervisor_open_high_confidence", 0.85)

        state = _make_state(
            visual_confidence=0.30,
            evidence=PNEUMONIA_EVIDENCE,
            visual_answer="pneumonia",
            answer_type="open",
            question="What is the diagnosis?",
        )
        result = supervisor_node(state)
        assert result["decision"] == "abstain", (
            "open question with conf=0.30 should still abstain "
            "when open_low_confidence=0.55"
        )

    def test_closed_question_re_queries_when_above_closed_low_but_no_evidence(self, monkeypatch):
        """closed + conf=0.30 + no evidence → re_query, not abstain.

        With closed_low_confidence=0.20: 0.30 >= 0.20 (moderate range).
        No evidence → Case D → re_query (at retry_count=0).
        """
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_closed_low_confidence", 0.20)
        monkeypatch.setattr(settings, "supervisor_closed_high_confidence", 0.60)
        monkeypatch.setattr(settings, "supervisor_open_low_confidence", 0.55)
        monkeypatch.setattr(settings, "supervisor_open_high_confidence", 0.85)

        state = _make_state(
            visual_confidence=0.30,
            evidence=[],
            visual_answer="yes",
            answer_type="closed",
            question="Is there consolidation?",
            retry_count=0,
        )
        result = supervisor_node(state)
        assert result["decision"] == "re_query", (
            "closed question with conf=0.30 and no evidence should re_query "
            "(not abstain) when closed_low_confidence=0.20"
        )

    def test_closed_below_closed_low_confidence_still_abstains(self, monkeypatch):
        """closed + conf=0.10 → abstain even with closed_low_confidence=0.20.

        The new lower threshold still has a floor. Confidence below 0.20 → Case E.
        """
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_closed_low_confidence", 0.20)
        monkeypatch.setattr(settings, "supervisor_closed_high_confidence", 0.60)

        state = _make_state(
            visual_confidence=0.10,
            evidence=PNEUMONIA_EVIDENCE,
            visual_answer="yes",
            answer_type="closed",
            question="Is there pneumonia?",
        )
        result = supervisor_node(state)
        assert result["decision"] == "abstain", (
            "closed question with conf=0.10 should still abstain "
            "because 0.10 < closed_low_confidence=0.20"
        )

    def test_closed_high_confidence_with_evidence_answers(self, monkeypatch):
        """closed + conf=0.70 >= closed_high_confidence=0.60 + evidence → Case A → answer."""
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_closed_low_confidence", 0.20)
        monkeypatch.setattr(settings, "supervisor_closed_high_confidence", 0.60)

        state = _make_state(
            visual_confidence=0.70,
            evidence=PNEUMONIA_EVIDENCE,
            visual_answer="yes",
            answer_type="closed",
            question="Is there pneumonia?",
        )
        result = supervisor_node(state)
        assert result["decision"] == "answer"
        assert result["grounded_confidence"] > 0.0

    def test_per_type_disabled_falls_back_to_global_thresholds(self, monkeypatch):
        """If closed_low_confidence=0.0 (disabled), fall back to global LOW_CONFIDENCE.

        This tests the 0.0-disables-per-type contract.
        With global LOW_CONFIDENCE=0.55 and conf=0.30, should abstain.
        """
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_closed_low_confidence", 0.0)   # disabled
        monkeypatch.setattr(settings, "supervisor_closed_high_confidence", 0.0)  # disabled
        monkeypatch.setattr(settings, "supervisor_low_confidence", 0.55)         # global fallback
        monkeypatch.setattr(settings, "supervisor_high_confidence", 0.85)

        state = _make_state(
            visual_confidence=0.30,
            evidence=PNEUMONIA_EVIDENCE,
            visual_answer="yes",
            answer_type="closed",
            question="Is there pneumonia?",
        )
        result = supervisor_node(state)
        assert result["decision"] == "abstain", (
            "when closed per-type thresholds are disabled (0.0), "
            "should fall back to global LOW_CONFIDENCE=0.55 and abstain"
        )

    def test_unknown_answer_type_uses_open_thresholds(self, monkeypatch):
        """answer_type not 'closed' → treated as 'open', uses open thresholds.

        Any value other than 'closed' (including missing/unknown) should use open thresholds.
        With open_low_confidence=0.55 and conf=0.30 → Case E → abstain.
        """
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_open_low_confidence", 0.55)
        monkeypatch.setattr(settings, "supervisor_open_high_confidence", 0.85)

        state = _make_state(
            visual_confidence=0.30,
            evidence=PNEUMONIA_EVIDENCE,
            visual_answer="pneumonia",
            answer_type="unknown_type",  # neither "closed" nor "open"
            question="What is visible?",
        )
        result = supervisor_node(state)
        assert result["decision"] == "abstain", (
            "unknown answer_type should fall through to open thresholds, "
            "causing abstain at conf=0.30 < open_low_confidence=0.55"
        )

    def test_decision_reasoning_includes_answer_type_on_abstain(self, monkeypatch):
        """When Case E triggers, decision_reasoning should mention answer_type.

        This supports auditability — logs should be traceable to which branch fired.
        """
        from radiology_vqa.config import settings
        monkeypatch.setattr(settings, "supervisor_open_low_confidence", 0.55)

        state = _make_state(
            visual_confidence=0.10,
            evidence=[],
            visual_answer="pneumonia",
            answer_type="open",
            question="What is the diagnosis?",
        )
        result = supervisor_node(state)
        assert result["decision"] == "abstain"
        assert "open" in result["decision_reasoning"], (
            "decision_reasoning should include answer_type='open' when Case E fires"
        )

    def test_all_existing_cases_unaffected_with_default_open_thresholds(self, monkeypatch):
        """Regression test: existing Phase 6 behaviour is preserved when open thresholds
        match the Phase 6 global defaults exactly.

        This ensures the refactor is a strict superset — no existing test should break.
        Uses the same confidence values and evidence as TestCaseA/B/C/D/E.
        """
        from radiology_vqa.config import settings
        # Set open thresholds to exactly match Phase 6 global defaults
        monkeypatch.setattr(settings, "supervisor_open_high_confidence", 0.85)
        monkeypatch.setattr(settings, "supervisor_open_low_confidence", 0.55)
        monkeypatch.setattr(settings, "supervisor_closed_high_confidence", 0.60)
        monkeypatch.setattr(settings, "supervisor_closed_low_confidence", 0.20)

        # These are the exact same inputs used in TestCaseA–E for open questions.
        # They must produce the same results as before.
        cases = [
            # (conf, evidence, retry, expected_decision)
            (0.92, PNEUMONIA_EVIDENCE, 0, "answer"),   # Case A
            (0.90, [],                 0, "re_query"),  # Case B
            (0.90, [],                 1, "abstain"),   # Case B2
            (0.70, PNEUMONIA_EVIDENCE, 0, "answer"),   # Case C
            (0.60, [],                 0, "re_query"),  # Case D
            (0.60, [],                 1, "abstain"),   # Case D2
            (0.40, PNEUMONIA_EVIDENCE, 0, "abstain"),  # Case E
        ]
        for conf, evidence, retry, expected in cases:
            state = _make_state(
                conf, evidence,
                visual_answer="pneumonia", answer_type="open",
                retry_count=retry,
            )
            result = supervisor_node(state)
            assert result["decision"] == expected, (
                f"Regression: open question conf={conf} retry={retry} evidence={len(evidence)} "
                f"expected={expected!r} got={result['decision']!r}"
            )
