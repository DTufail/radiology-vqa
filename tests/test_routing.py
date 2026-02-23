"""Tests for the conditional routing function route_after_supervisor."""

import pytest

from radiology_vqa.graph.routing import route_after_supervisor


class TestRouteAfterSupervisor:
    def test_answer_routes_to_output_formatter(self):
        state = {"decision": "answer", "retry_count": 0}
        assert route_after_supervisor(state) == "output_formatter"

    def test_answer_with_high_retry_still_routes_to_output_formatter(self):
        state = {"decision": "answer", "retry_count": 5}
        assert route_after_supervisor(state) == "output_formatter"

    def test_re_query_with_retry_count_zero_routes_to_retrieval_agent(self):
        # retry_count=0: 0 > max_retries(1) is False → retrieval_agent
        state = {"decision": "re_query", "retry_count": 0}
        assert route_after_supervisor(state) == "retrieval_agent"

    def test_re_query_with_retry_count_one_routes_to_retrieval_agent(self):
        # Supervisor sets retry_count=1 when emitting first re_query.
        # 1 > max_retries(1) is False → retrieval_agent (second attempt allowed)
        state = {"decision": "re_query", "retry_count": 1}
        assert route_after_supervisor(state) == "retrieval_agent"

    def test_re_query_above_max_retries_triggers_safety_bound(self):
        # retry_count=2 > max_retries(1) → safety bound → output_formatter
        state = {"decision": "re_query", "retry_count": 2}
        assert route_after_supervisor(state) == "output_formatter"

    def test_re_query_well_above_max_retries_still_safe(self):
        state = {"decision": "re_query", "retry_count": 99}
        assert route_after_supervisor(state) == "output_formatter"

    def test_abstain_routes_to_output_formatter(self):
        state = {"decision": "abstain", "retry_count": 0}
        assert route_after_supervisor(state) == "output_formatter"

    def test_abstain_with_high_retry_routes_to_output_formatter(self):
        state = {"decision": "abstain", "retry_count": 3}
        assert route_after_supervisor(state) == "output_formatter"

    def test_unknown_decision_routes_to_output_formatter(self):
        state = {"decision": "unknown_value", "retry_count": 0}
        assert route_after_supervisor(state) == "output_formatter"

    def test_empty_string_decision_routes_to_output_formatter(self):
        state = {"decision": "", "retry_count": 0}
        assert route_after_supervisor(state) == "output_formatter"

    def test_missing_decision_key_routes_to_output_formatter(self):
        # state has no "decision" key at all
        state = {"retry_count": 0}
        assert route_after_supervisor(state) == "output_formatter"

    def test_missing_retry_count_treated_as_zero(self):
        # state has re_query but no retry_count → defaults to 0 < max_retries(1)
        state = {"decision": "re_query"}
        assert route_after_supervisor(state) == "retrieval_agent"

    def test_return_type_is_string(self):
        for decision in ("answer", "re_query", "abstain", "unknown"):
            result = route_after_supervisor({"decision": decision, "retry_count": 0})
            assert isinstance(result, str)
