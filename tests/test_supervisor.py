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
