"""Compare agent pipeline results against VLM-only baseline.

Produces the ablation study: did RAG grounding help?
"""

import logging
from typing import Sequence

from radiology_vqa.evaluation.agent_metrics import (
    correct_abstention_rate as compute_correct_abstention_rate,
    grounding_improvement,
)
from radiology_vqa.evaluation.result import ComparisonResult, EvaluationResult

logger = logging.getLogger(__name__)


class BaselineComparator:
    """Compare two EvaluationResults (agent vs baseline).

    Usage::

        comparator = BaselineComparator()
        comparison = comparator.compare(agent_result, vlm_result)
        comparison.save(Path("data/evaluation_reports/comparison.json"))
    """

    def compare(
        self,
        agent_result: EvaluationResult,
        baseline_result: EvaluationResult,
    ) -> ComparisonResult:
        """Produce full comparison between agent and baseline.

        Steps:
        1. Align per-sample results by sample_id.
        2. Compute metric deltas (agent - baseline).
        3. Compute grounding_improvement from agent_metrics.
        4. Compute correct_abstention_rate.
        5. Run McNemar's test for statistical significance.
        6. Generate formatted markdown tables.
        7. Package into ComparisonResult.
        """
        # ── Align per-sample results by sample_id ───────────────────
        agent_by_id = {r.sample_id: r for r in agent_result.per_sample}
        baseline_by_id = {r.sample_id: r for r in baseline_result.per_sample}

        common_ids = sorted(set(agent_by_id) & set(baseline_by_id))
        n_agent = len(agent_by_id)
        n_baseline = len(baseline_by_id)

        if n_agent != n_baseline:
            logger.warning(
                "Sample count mismatch: agent=%d, baseline=%d. "
                "Using intersection of %d samples.",
                n_agent,
                n_baseline,
                len(common_ids),
            )
        if not common_ids:
            logger.warning("No common sample_ids found between agent and baseline results.")

        aligned_agent = [agent_by_id[sid] for sid in common_ids]
        aligned_baseline = [baseline_by_id[sid] for sid in common_ids]
        n = len(common_ids)

        # ── Metric deltas ────────────────────────────────────────────
        deltas = {
            "accuracy_delta": agent_result.overall_accuracy - baseline_result.overall_accuracy,
            "closed_accuracy_delta": agent_result.closed_accuracy - baseline_result.closed_accuracy,
            "open_accuracy_delta": agent_result.open_accuracy - baseline_result.open_accuracy,
            "open_token_f1_delta": agent_result.open_token_f1 - baseline_result.open_token_f1,
            "open_bertscore_delta": (
                agent_result.open_bertscore_f1 - baseline_result.open_bertscore_f1
                if agent_result.open_bertscore_f1 >= 0 and baseline_result.open_bertscore_f1 >= 0
                else 0.0
            ),
        }

        # ── Grounding improvement ────────────────────────────────────
        agent_preds = [r.prediction for r in aligned_agent]
        vlm_preds = [r.prediction for r in aligned_baseline]
        ground_truths = [r.ground_truth for r in aligned_agent]
        agent_decisions = [r.decision for r in aligned_agent]

        grounding = grounding_improvement(agent_preds, vlm_preds, ground_truths, agent_decisions)

        # ── Correct abstention rate ──────────────────────────────────
        vlm_preds_full = [baseline_by_id.get(sid, aligned_baseline[0]).prediction for sid in common_ids]
        car = compute_correct_abstention_rate(vlm_preds_full, ground_truths, agent_decisions)

        # ── McNemar's test ───────────────────────────────────────────
        agent_correct = [r.correct for r in aligned_agent]
        baseline_correct = [r.correct for r in aligned_baseline]
        stat, p_val = self._mcnemar_test(agent_correct, baseline_correct)
        is_sig = p_val < 0.05

        # ── Markdown tables ──────────────────────────────────────────
        comparison_md = self._format_comparison_table(agent_result, baseline_result, deltas)
        grounding_md = self._format_grounding_table(grounding, n)

        return ComparisonResult(
            agent_name=agent_result.model_name,
            baseline_name=baseline_result.model_name,
            dataset=agent_result.dataset,
            split=agent_result.split,
            total_samples=n,
            accuracy_delta=deltas["accuracy_delta"],
            closed_accuracy_delta=deltas["closed_accuracy_delta"],
            open_accuracy_delta=deltas["open_accuracy_delta"],
            open_token_f1_delta=deltas["open_token_f1_delta"],
            open_bertscore_delta=deltas["open_bertscore_delta"],
            improved=grounding["improved"],
            degraded=grounding["degraded"],
            both_correct=grounding["both_correct"],
            both_wrong=grounding["both_wrong"],
            agent_abstained=grounding["agent_abstained"],
            abstain_vlm_correct=grounding["abstain_vlm_correct"],
            abstain_vlm_wrong=grounding["abstain_vlm_wrong"],
            net_improvement=grounding["net_improvement"],
            correct_abstention_rate=car,
            mcnemar_statistic=stat,
            mcnemar_p_value=p_val,
            is_significant=is_sig,
            comparison_table_md=comparison_md,
            grounding_table_md=grounding_md,
        )

    def _mcnemar_test(
        self,
        agent_correct: Sequence[bool],
        baseline_correct: Sequence[bool],
    ) -> tuple[float, float]:
        """McNemar's test for comparing two classifiers on the same dataset.

        Uses only discordant pairs:
        - b = agent correct, baseline wrong (improved)
        - c = agent wrong, baseline correct (degraded)

        Test statistic: (|b - c| - 1)² / (b + c)  [chi-squared, continuity corrected]

        Returns (statistic, p_value).

        Edge cases:
        - b + c < 5 → insufficient discordant pairs, return (0.0, 1.0)
        - b + c < 25 → exact binomial test
        - b + c >= 25 → chi-squared approximation
        """
        b = sum(1 for a, v in zip(agent_correct, baseline_correct) if a and not v)
        c = sum(1 for a, v in zip(agent_correct, baseline_correct) if not a and v)
        n_disc = b + c

        if n_disc < 5:
            logger.warning(
                "McNemar: only %d discordant pairs — insufficient for reliable test "
                "(b=%d, c=%d). Returning p_value=1.0.",
                n_disc,
                b,
                c,
            )
            return 0.0, 1.0

        try:
            from scipy import stats as scipy_stats

            if n_disc < 25:
                # Exact binomial test: H0: P(agent better) = 0.5
                binom_result = scipy_stats.binomtest(b, n_disc, p=0.5, alternative="two-sided")
                return 0.0, float(binom_result.pvalue)
            else:
                # Chi-squared approximation with continuity correction
                statistic = (abs(b - c) - 1) ** 2 / n_disc
                p_value = float(scipy_stats.chi2.sf(statistic, df=1))
                return float(statistic), p_value

        except ImportError:
            logger.warning("scipy not installed; McNemar returning p_value=1.0")
            return 0.0, 1.0

    def _format_comparison_table(
        self,
        agent_result: EvaluationResult,
        baseline_result: EvaluationResult,
        deltas: dict,
    ) -> str:
        """Generate markdown comparison table."""

        def fmt_pct(v: float) -> str:
            return f"{100*v:.1f}%"

        def fmt_delta(v: float) -> str:
            sign = "+" if v >= 0 else ""
            return f"{sign}{100*v:.1f}%"

        def fmt_f1(v: float) -> str:
            return f"{v:.3f}" if v >= 0 else "N/A"

        def fmt_f1_delta(v: float) -> str:
            if v == 0.0:
                return "N/A"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.3f}"

        rows = [
            (
                "Overall Accuracy",
                fmt_pct(baseline_result.overall_accuracy),
                fmt_pct(agent_result.overall_accuracy),
                fmt_delta(deltas["accuracy_delta"]),
            ),
            (
                "Closed Accuracy",
                fmt_pct(baseline_result.closed_accuracy),
                fmt_pct(agent_result.closed_accuracy),
                fmt_delta(deltas["closed_accuracy_delta"]),
            ),
            (
                "Open Accuracy",
                fmt_pct(baseline_result.open_accuracy),
                fmt_pct(agent_result.open_accuracy),
                fmt_delta(deltas["open_accuracy_delta"]),
            ),
            (
                "Open Token F1",
                fmt_f1(baseline_result.open_token_f1),
                fmt_f1(agent_result.open_token_f1),
                fmt_f1_delta(deltas["open_token_f1_delta"]),
            ),
            (
                "Open BERTScore F1",
                fmt_f1(baseline_result.open_bertscore_f1),
                fmt_f1(agent_result.open_bertscore_f1),
                fmt_f1_delta(deltas["open_bertscore_delta"]),
            ),
            (
                "Abstention Rate",
                "—",
                fmt_pct(agent_result.abstention_rate),
                "—",
            ),
            (
                "Accuracy (Answered)",
                fmt_pct(baseline_result.overall_accuracy),
                fmt_pct(agent_result.accuracy_when_answered),
                "—",
            ),
            (
                "ECE (↓ better)",
                f"{baseline_result.ece:.3f}",
                f"{agent_result.ece:.3f}",
                f"{agent_result.ece - baseline_result.ece:+.3f}",
            ),
            (
                "Confidence AUROC",
                f"{baseline_result.confidence_auroc:.3f}",
                f"{agent_result.confidence_auroc:.3f}",
                f"{agent_result.confidence_auroc - baseline_result.confidence_auroc:+.3f}",
            ),
        ]

        header = "| Metric | VLM-Only | Agent | Delta |"
        separator = "|--------|----------|-------|-------|"
        table_rows = [f"| {m} | {v} | {a} | {d} |" for m, v, a, d in rows]

        return "\n".join([header, separator] + table_rows)

    def _format_grounding_table(self, grounding: dict, total: int) -> str:
        """Generate markdown grounding breakdown table."""

        def fmt_row(label: str, count: int) -> str:
            pct = 100 * count / total if total > 0 else 0.0
            return f"| {label} | {count} | {pct:.1f}% |"

        rows = [
            fmt_row("Improved (agent ✓, VLM ✗)", grounding["improved"]),
            fmt_row("Degraded (agent ✗, VLM ✓)", grounding["degraded"]),
            fmt_row("Both Correct", grounding["both_correct"]),
            fmt_row("Both Wrong", grounding["both_wrong"]),
            fmt_row("Agent Abstained — VLM Correct (over-abstention)", grounding["abstain_vlm_correct"]),
            fmt_row("Agent Abstained — VLM Wrong (justified)", grounding["abstain_vlm_wrong"]),
        ]

        net = grounding["net_improvement"]
        net_sign = "+" if net >= 0 else ""

        header = "| Category | Count | % of Total |"
        separator = "|----------|-------|------------|"
        net_row = f"| **Net Improvement** | **{net_sign}{net}** | — |"

        return "\n".join([header, separator] + rows + [net_row])
