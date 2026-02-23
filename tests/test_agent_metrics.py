"""Tests for evaluation/agent_metrics.py — agent behavior metrics, no GPU."""

import pytest

from radiology_vqa.evaluation.agent_metrics import (
    abstention_rate,
    accuracy_when_answered,
    citation_relevance,
    correct_abstention_rate,
    grounding_improvement,
    re_query_rate,
    re_query_rate_from_counts,
)


# ── abstention_rate ───────────────────────────────────────────────────────────


class TestAbstentionRate:
    def test_partial_abstention(self):
        decisions = ["answer"] * 7 + ["abstain"] * 3
        assert abstention_rate(decisions) == pytest.approx(0.3)

    def test_all_answered(self):
        assert abstention_rate(["answer", "answer"]) == 0.0

    def test_all_abstained(self):
        assert abstention_rate(["abstain", "abstain"]) == 1.0

    def test_empty(self):
        assert abstention_rate([]) == 0.0

    def test_single_abstain(self):
        assert abstention_rate(["abstain"]) == pytest.approx(1.0)


# ── accuracy_when_answered ────────────────────────────────────────────────────


class TestAccuracyWhenAnswered:
    def test_partial_abstention(self):
        preds = ["yes", "no", "yes", "liver", "no"]
        gts = ["yes", "yes", "yes", "liver", "no"]
        decisions = ["answer", "answer", "answer", "abstain", "answer"]
        # answered: 0,1,2,4 → correct: 0,2,4 → 3/4
        result = accuracy_when_answered(preds, gts, decisions)
        assert result == pytest.approx(3 / 4)

    def test_all_abstained_returns_zero(self):
        result = accuracy_when_answered(["yes"], ["yes"], ["abstain"])
        assert result == 0.0

    def test_none_abstained(self):
        preds = ["yes", "no"]
        gts = ["yes", "no"]
        decisions = ["answer", "answer"]
        assert accuracy_when_answered(preds, gts, decisions) == pytest.approx(1.0)

    def test_perfect_on_answered(self):
        preds = ["yes", "yes", "no"]
        gts = ["yes", "yes", "yes"]
        decisions = ["answer", "answer", "abstain"]
        assert accuracy_when_answered(preds, gts, decisions) == pytest.approx(1.0)

    def test_zero_correct_on_answered(self):
        preds = ["no", "no"]
        gts = ["yes", "yes"]
        decisions = ["answer", "answer"]
        assert accuracy_when_answered(preds, gts, decisions) == pytest.approx(0.0)


# ── correct_abstention_rate ───────────────────────────────────────────────────


class TestCorrectAbstentionRate:
    def test_all_justified_abstentions(self):
        # VLM wrong on both abstained samples
        vlm = ["no", "yes"]
        gts = ["yes", "no"]
        decisions = ["abstain", "abstain"]
        assert correct_abstention_rate(vlm, gts, decisions) == pytest.approx(1.0)

    def test_all_unjustified_abstentions(self):
        # VLM correct on both abstained samples
        vlm = ["yes", "no"]
        gts = ["yes", "no"]
        decisions = ["abstain", "abstain"]
        assert correct_abstention_rate(vlm, gts, decisions) == pytest.approx(0.0)

    def test_mixed_abstentions(self):
        # 3 abstentions; VLM wrong on 2 of them
        vlm = ["no", "yes", "yes"]
        gts = ["yes", "no", "yes"]
        decisions = ["abstain", "abstain", "abstain"]
        # index 0: vlm wrong, index 1: vlm wrong, index 2: vlm correct
        result = correct_abstention_rate(vlm, gts, decisions)
        assert result == pytest.approx(2 / 3)

    def test_no_abstentions_returns_zero(self):
        assert correct_abstention_rate(["yes"], ["yes"], ["answer"]) == 0.0

    def test_ignores_answered_samples(self):
        # Only abstained samples count
        vlm = ["yes", "no"]
        gts = ["yes", "yes"]
        decisions = ["answer", "abstain"]
        # Only index 1 is abstained; vlm="no", gt="yes" → wrong → rate=1.0
        assert correct_abstention_rate(vlm, gts, decisions) == pytest.approx(1.0)


# ── re_query_rate ─────────────────────────────────────────────────────────────


class TestReQueryRate:
    def test_partial(self):
        decisions = ["answer", "re_query", "abstain", "re_query"]
        assert re_query_rate(decisions) == pytest.approx(0.5)

    def test_none(self):
        assert re_query_rate(["answer", "answer"]) == 0.0

    def test_empty(self):
        assert re_query_rate([]) == 0.0

    def test_all_re_query(self):
        assert re_query_rate(["re_query", "re_query"]) == pytest.approx(1.0)


# ── re_query_rate_from_counts ─────────────────────────────────────────────────


class TestReQueryRateFromCounts:
    def test_partial(self):
        counts = [0, 1, 0, 2]
        assert re_query_rate_from_counts(counts) == pytest.approx(0.5)

    def test_none(self):
        assert re_query_rate_from_counts([0, 0, 0]) == 0.0

    def test_empty(self):
        assert re_query_rate_from_counts([]) == 0.0

    def test_all_retried(self):
        assert re_query_rate_from_counts([1, 2, 1]) == pytest.approx(1.0)


# ── grounding_improvement ─────────────────────────────────────────────────────


class TestGroundingImprovement:
    def test_all_categories(self):
        # 0: agent correct, vlm wrong → improved
        # 1: agent wrong, vlm correct → degraded
        # 2: both correct → both_correct
        # 3: both wrong → both_wrong
        # 4: agent abstained, vlm wrong → abstain_vlm_wrong
        agent_preds = ["yes", "no", "liver", "lung", "ABSTAIN"]
        vlm_preds = ["no", "yes", "liver", "spleen", "no"]
        gts = ["yes", "yes", "liver", "liver", "yes"]
        decisions = ["answer", "answer", "answer", "answer", "abstain"]
        result = grounding_improvement(agent_preds, vlm_preds, gts, decisions)
        assert result["improved"] == 1
        assert result["degraded"] == 1
        assert result["both_correct"] == 1
        assert result["both_wrong"] == 1
        assert result["agent_abstained"] == 1
        assert result["abstain_vlm_correct"] == 0
        assert result["abstain_vlm_wrong"] == 1

    def test_counts_sum_to_total(self):
        agent_preds = ["yes", "no", "liver"]
        vlm_preds = ["yes", "no", "liver"]
        gts = ["yes", "no", "spleen"]
        decisions = ["answer", "abstain", "answer"]
        result = grounding_improvement(agent_preds, vlm_preds, gts, decisions)
        total = (
            result["improved"]
            + result["degraded"]
            + result["both_correct"]
            + result["both_wrong"]
            + result["agent_abstained"]
        )
        assert total == result["total"]

    def test_net_improvement_equals_improved_minus_degraded(self):
        # 2 improved, 1 degraded → net = 1
        agent_preds = ["yes", "yes", "no"]
        vlm_preds = ["no", "no", "yes"]
        gts = ["yes", "yes", "no"]
        decisions = ["answer", "answer", "answer"]
        result = grounding_improvement(agent_preds, vlm_preds, gts, decisions)
        assert result["net_improvement"] == result["improved"] - result["degraded"]

    def test_abstain_subcategories_sum(self):
        agent_preds = ["yes", "yes"]
        vlm_preds = ["yes", "no"]  # first correct, second wrong
        gts = ["yes", "yes"]
        decisions = ["abstain", "abstain"]
        result = grounding_improvement(agent_preds, vlm_preds, gts, decisions)
        assert result["abstain_vlm_correct"] == 1
        assert result["abstain_vlm_wrong"] == 1
        assert (
            result["abstain_vlm_correct"] + result["abstain_vlm_wrong"]
            == result["agent_abstained"]
        )

    def test_all_improved(self):
        agent_preds = ["yes", "no"]
        vlm_preds = ["no", "yes"]
        gts = ["yes", "no"]
        decisions = ["answer", "answer"]
        result = grounding_improvement(agent_preds, vlm_preds, gts, decisions)
        assert result["improved"] == 2
        assert result["net_improvement"] == 2


# ── citation_relevance ────────────────────────────────────────────────────────


class TestCitationRelevance:
    def test_relevant_citations(self):
        citations = [[{"text": "The liver shows normal attenuation on CT scan."}]]
        gts = ["liver"]
        result = citation_relevance(citations, gts)
        assert result["citation_hit_rate"] == pytest.approx(1.0)
        assert result["total_cited_samples"] == 1
        assert result["total_uncited_samples"] == 0

    def test_no_relevant_citations(self):
        citations = [[{"text": "The patient presents with an abnormality."}]]
        gts = ["spleen"]
        result = citation_relevance(citations, gts)
        # "spleen" not in citation text → no hit
        assert result["citation_hit_rate"] == pytest.approx(0.0)

    def test_empty_citations_counted_as_uncited(self):
        citations = [[], [{"text": "liver disease is common"}]]
        gts = ["liver", "liver"]
        result = citation_relevance(citations, gts)
        assert result["total_uncited_samples"] == 1
        assert result["total_cited_samples"] == 1

    def test_short_gt_tokens_excluded_from_matching(self):
        # "CT" normalizes to "ct" (2 chars) → excluded from matching (len ≤ 2)
        citations = [[{"text": "CT scan shows normal findings."}]]
        gts = ["CT"]
        result = citation_relevance(citations, gts)
        assert result["citation_hit_rate"] == pytest.approx(0.0)

    def test_mean_relevant_citations(self):
        citations = [
            [{"text": "liver anatomy"}, {"text": "liver disease"}],
        ]
        gts = ["liver"]
        result = citation_relevance(citations, gts)
        # Both citations contain "liver" (5 chars > 2)
        assert result["mean_relevant_citations"] == pytest.approx(2.0)

    def test_all_empty_citations(self):
        citations = [[], []]
        gts = ["liver", "spleen"]
        result = citation_relevance(citations, gts)
        assert result["citation_hit_rate"] == pytest.approx(0.0)
        assert result["total_cited_samples"] == 0
        assert result["total_uncited_samples"] == 2
