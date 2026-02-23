"""Generate evaluation reports in Markdown and JSON.

The markdown report is the Phase 5 deliverable. It goes directly into
the technical report with minimal editing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from radiology_vqa.evaluation.result import (
    ComparisonResult,
    EvaluationResult,
    PerSampleResult,
)

logger = logging.getLogger(__name__)


def generate_report(
    agent_result: EvaluationResult,
    baseline_result: EvaluationResult | None = None,
    comparison: ComparisonResult | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate comprehensive evaluation report.

    Creates output_dir/ with:
    - report.md: human-readable markdown
    - report.json: machine-readable (all three results combined)
    - agent_result.json: raw agent evaluation
    - baseline_result.json: raw baseline evaluation (if provided)
    - comparison.json: comparison result (if provided)

    Returns path to output_dir.
    """
    if output_dir is None:
        try:
            from radiology_vqa.config import settings

            output_dir = settings.eval_output_dir
        except Exception:
            output_dir = Path("data/evaluation_reports")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw result files
    agent_result.save(output_dir / "agent_result.json")
    if baseline_result is not None:
        baseline_result.save(output_dir / "baseline_result.json")
    if comparison is not None:
        comparison.save(output_dir / "comparison.json")

    # Build markdown report
    sections = []
    sections.append(_format_header(agent_result))
    sections.append(_format_executive_summary(agent_result, baseline_result, comparison))
    sections.append(_format_baselines_table(agent_result, baseline_result))

    if comparison is not None and baseline_result is not None:
        sections.append(_format_results_comparison(comparison, agent_result, baseline_result))

    sections.append(_format_closed_analysis(agent_result))
    sections.append(_format_open_analysis(agent_result))

    if agent_result.evaluation_mode == "agent":
        sections.append(_format_agent_behavior(agent_result))

    sections.append(_format_calibration(agent_result))
    sections.append(
        _format_error_analysis(agent_result, baseline_result, comparison)
    )
    sections.append(_format_recommendations(agent_result, comparison))
    sections.append(_format_footer())

    report_md = "\n\n".join(s for s in sections if s)
    (output_dir / "report.md").write_text(report_md)

    # Save combined JSON
    combined = {
        "generated_at": datetime.now().isoformat(),
        "agent": json.loads(agent_result.model_dump_json()),
        "baseline": json.loads(baseline_result.model_dump_json()) if baseline_result else None,
        "comparison": json.loads(comparison.model_dump_json()) if comparison else None,
    }
    (output_dir / "report.json").write_text(json.dumps(combined, indent=2))

    logger.info("Report saved to %s", output_dir)
    print(f"Report saved: {output_dir / 'report.md'}")
    return output_dir


# ── Section formatters ────────────────────────────────────────────────────────


def _format_header(agent_result: EvaluationResult) -> str:
    ts = agent_result.timestamp[:10]
    return (
        f"# Evaluation Report: Grounded Multi-Agent Radiology VQA\n\n"
        f"**Dataset:** {agent_result.dataset.upper()} — {agent_result.split} split  \n"
        f"**Model:** {agent_result.model_name}  \n"
        f"**Date:** {ts}  \n"
        f"**Samples:** {agent_result.total_samples}"
    )


def _format_executive_summary(
    agent_result: EvaluationResult,
    baseline_result: EvaluationResult | None,
    comparison: ComparisonResult | None,
) -> str:
    lines = ["## Executive Summary", ""]
    acc = agent_result.overall_accuracy

    if baseline_result is not None and comparison is not None:
        delta_pct = 100 * comparison.accuracy_delta
        direction = "improvement" if delta_pct >= 0 else "degradation"
        sig_str = " (statistically significant, p < 0.05)" if comparison.is_significant else ""
        net = comparison.net_improvement

        lines.append(
            f"The grounded multi-agent system achieved **{100*acc:.1f}% overall accuracy** on "
            f"{agent_result.dataset.upper()} {agent_result.split}, compared to "
            f"**{100*baseline_result.overall_accuracy:.1f}%** for the VLM-only baseline — "
            f"a **{abs(delta_pct):.1f}% {direction}**{sig_str}. "
            f"Grounding analysis shows {net:+d} net samples improved through RAG retrieval. "
            f"The agent abstained on {100*agent_result.abstention_rate:.1f}% of queries, "
            f"with {100*agent_result.accuracy_when_answered:.1f}% accuracy when it did answer."
        )
    else:
        lines.append(
            f"The multi-agent system achieved **{100*acc:.1f}% overall accuracy** on "
            f"{agent_result.dataset.upper()} {agent_result.split} "
            f"({agent_result.total_samples} samples). "
            f"Closed-ended accuracy: {100*agent_result.closed_accuracy:.1f}%. "
            f"Open-ended accuracy: {100*agent_result.open_accuracy:.1f}%. "
            f"ECE (calibration): {agent_result.ece:.3f}."
        )

    return "\n".join(lines)


def _format_baselines_table(
    agent_result: EvaluationResult,
    baseline_result: EvaluationResult | None,
) -> str:
    lines = ["## Baseline Results", ""]
    lines.append(
        "Prior results on VQA-RAD (exact match, test split) for context:\n"
    )
    lines.append("| System | Overall Acc | Closed Acc | Open Acc | Notes |")
    lines.append("|--------|-------------|------------|----------|-------|")
    lines.append("| BLIP-2 (Salesforce/blip2-opt-2.7b) | ~25–35% | ~40% | ~15% | Zero-shot |")
    lines.append("| LLaVA v1.6 (VLM-only) | ~35–45% | ~50% | ~25% | This project baseline |")

    if baseline_result is not None:
        lines.append(
            f"| **{baseline_result.model_name}** (VLM-only) | "
            f"**{100*baseline_result.overall_accuracy:.1f}%** | "
            f"**{100*baseline_result.closed_accuracy:.1f}%** | "
            f"**{100*baseline_result.open_accuracy:.1f}%** | "
            f"This evaluation |"
        )

    lines.append(
        f"| **{agent_result.model_name}** (Agent + RAG) | "
        f"**{100*agent_result.overall_accuracy:.1f}%** | "
        f"**{100*agent_result.closed_accuracy:.1f}%** | "
        f"**{100*agent_result.open_accuracy:.1f}%** | "
        f"This evaluation |"
    )
    return "\n".join(lines)


def _format_results_comparison(
    comparison: ComparisonResult,
    agent_result: EvaluationResult,
    baseline_result: EvaluationResult,
) -> str:
    lines = ["## Results Comparison", ""]

    lines.append(comparison.comparison_table_md)
    lines.append("")

    sig_str = (
        f"McNemar's test: χ²={comparison.mcnemar_statistic:.3f}, "
        f"p={comparison.mcnemar_p_value:.4f} — "
        + ("**statistically significant** (p < 0.05)" if comparison.is_significant else "not significant (p ≥ 0.05)")
    )
    lines.append(f"*{sig_str}*")

    return "\n".join(lines)


def _format_closed_analysis(agent_result: EvaluationResult) -> str:
    lines = ["## Closed-Ended Analysis", ""]
    cm = agent_result.closed_confusion
    n_closed = agent_result.closed_count

    lines.append(f"**Count:** {n_closed} samples  ")
    lines.append(
        f"**Accuracy:** {100*agent_result.closed_accuracy:.1f}%  "
    )
    lines.append(
        f"**Precision:** {100*agent_result.closed_precision:.1f}%  "
        f"**Recall:** {100*agent_result.closed_recall:.1f}%  "
        f"**F1:** {100*agent_result.closed_f1:.1f}%"
    )
    lines.append("")
    lines.append("### Confusion Matrix (Yes = Positive Class)")
    lines.append("")
    lines.append("| | Predicted Yes | Predicted No |")
    lines.append("|---|---|---|")
    lines.append(f"| **GT Yes** | TP={cm.get('tp',0)} | FN={cm.get('fn',0)} |")
    lines.append(f"| **GT No** | FP={cm.get('fp',0)} | TN={cm.get('tn',0)} |")

    return "\n".join(lines)


def _format_open_analysis(agent_result: EvaluationResult) -> str:
    lines = ["## Open-Ended Analysis", ""]
    n_open = agent_result.open_count

    lines.append(f"**Count:** {n_open} samples  ")
    lines.append(
        f"**Accuracy (EM):** {100*agent_result.open_accuracy:.1f}%  "
        f"**Token F1:** {agent_result.open_token_f1:.3f}  "
        f"**BLEU-1:** {agent_result.open_bleu_1:.3f}"
    )

    if agent_result.open_bertscore_f1 >= 0:
        lines.append(
            f"**BERTScore F1:** {agent_result.open_bertscore_f1:.3f}  "
            f"**BERTScore P:** {agent_result.open_bertscore_precision:.3f}  "
            f"**BERTScore R:** {agent_result.open_bertscore_recall:.3f}"
        )
    else:
        lines.append("**BERTScore:** Not computed (use `--bertscore` flag to enable)")

    return "\n".join(lines)


def _format_agent_behavior(agent_result: EvaluationResult) -> str:
    lines = ["## Agent Behavior Analysis", ""]

    lines.append(
        f"**Abstention Rate:** {100*agent_result.abstention_rate:.1f}%  "
        f"**Accuracy (Answered):** {100*agent_result.accuracy_when_answered:.1f}%  "
        f"**Re-query Rate:** {100*agent_result.re_query_rate:.1f}%  "
        f"**Citation Hit Rate:** {100*agent_result.citation_relevance_hit_rate:.1f}%"
    )

    return "\n".join(lines)


def _format_calibration(agent_result: EvaluationResult) -> str:
    lines = ["## Confidence Calibration", ""]

    lines.append(
        f"**ECE** (↓ better): {agent_result.ece:.4f}  "
        f"**AUROC**: {agent_result.confidence_auroc:.3f}  "
        f"**Mean Confidence (Correct)**: {agent_result.mean_correct_confidence:.3f}  "
        f"**Mean Confidence (Wrong)**: {agent_result.mean_wrong_confidence:.3f}"
    )
    lines.append("")

    if agent_result.calibration_bins:
        lines.append("### Calibration Bins")
        lines.append("")
        lines.append("| Bin | Count | Mean Conf | Accuracy | Gap |")
        lines.append("|-----|-------|-----------|----------|-----|")
        for b in agent_result.calibration_bins:
            if b["count"] > 0:
                lines.append(
                    f"| [{b['bin_start']:.1f}, {b['bin_end']:.1f}) "
                    f"| {b['count']} "
                    f"| {b['mean_confidence']:.3f} "
                    f"| {b['accuracy']:.3f} "
                    f"| {b['gap']:.3f} |"
                )

    if agent_result.threshold_analysis:
        lines.append("")
        lines.append("### Threshold Analysis")
        lines.append("")
        lines.append("| Threshold | Coverage | Accuracy | Count |")
        lines.append("|-----------|----------|----------|-------|")
        for t in agent_result.threshold_analysis:
            lines.append(
                f"| {t['threshold']:.2f} "
                f"| {100*t['coverage']:.1f}% "
                f"| {100*t['accuracy']:.1f}% "
                f"| {t['count']} |"
            )

        lines.append("")
        lines.append(
            "**Recommendation:** "
            + _format_threshold_recommendation(agent_result.threshold_analysis)
        )

    return "\n".join(lines)


def _format_error_analysis(
    agent_result: EvaluationResult,
    baseline_result: EvaluationResult | None,
    comparison: ComparisonResult | None,
) -> str:
    lines = ["## Error Analysis", ""]

    # Wrong answers (answered but incorrect)
    wrong = [
        r
        for r in agent_result.per_sample
        if not r.correct and r.decision != "abstain"
    ][:10]

    if wrong:
        lines.append("### Cases Where Agent Answered Incorrectly (up to 10)")
        lines.append("")
        for i, r in enumerate(wrong, 1):
            lines.append(f"**{i}.** `{r.sample_id}`")
            lines.append(f"- **Q:** {r.question}")
            lines.append(f"- **GT:** {r.ground_truth} | **Pred:** {r.prediction} | **Conf:** {r.confidence:.2f}")
            if r.citations:
                lines.append(f"- **Citations:** {len(r.citations)} retrieved")
            lines.append("")

    # Over-abstentions (abstained but baseline would have been correct)
    if baseline_result is not None:
        baseline_by_id = {r.sample_id: r for r in baseline_result.per_sample}
        over_abstain = [
            r
            for r in agent_result.per_sample
            if r.decision == "abstain" and baseline_by_id.get(r.sample_id, r).correct
        ][:5]

        if over_abstain:
            lines.append("### Over-Abstentions: Agent Abstained But VLM Was Correct (up to 5)")
            lines.append("")
            for i, r in enumerate(over_abstain, 1):
                vlm_pred = baseline_by_id.get(r.sample_id)
                lines.append(f"**{i}.** `{r.sample_id}`")
                lines.append(f"- **Q:** {r.question}")
                lines.append(f"- **GT:** {r.ground_truth} | **VLM Pred:** {vlm_pred.prediction if vlm_pred else '?'} | **Agent Conf:** {r.confidence:.2f}")
                lines.append("")

    # Grounding improvements (agent correct, baseline wrong)
    if baseline_result is not None:
        baseline_by_id = {r.sample_id: r for r in baseline_result.per_sample}
        improved = [
            r
            for r in agent_result.per_sample
            if r.correct and not baseline_by_id.get(r.sample_id, r).correct
            and r.decision == "answer"
        ][:5]

        if improved:
            lines.append("### Grounding Successes: Agent Correct, VLM Wrong (up to 5)")
            lines.append("")
            for i, r in enumerate(improved, 1):
                vlm = baseline_by_id.get(r.sample_id)
                lines.append(f"**{i}.** `{r.sample_id}`")
                lines.append(f"- **Q:** {r.question}")
                lines.append(f"- **GT:** {r.ground_truth} | **Agent:** {r.prediction} (✓) | **VLM:** {vlm.prediction if vlm else '?'} (✗)")
                if r.retrieval_query:
                    lines.append(f"- **Retrieval Query:** {r.retrieval_query}")
                lines.append("")

    return "\n".join(lines)


def _format_recommendations(
    agent_result: EvaluationResult,
    comparison: ComparisonResult | None,
) -> str:
    lines = ["## Recommendations for Phase 6", ""]

    recs = []

    if agent_result.abstention_rate > 0.25:
        recs.append(
            "**High abstention rate** ({:.1f}%): Consider lowering `supervisor_low_confidence` "
            "to reduce unnecessary abstentions that lose recall.".format(
                100 * agent_result.abstention_rate
            )
        )

    if agent_result.citation_relevance_hit_rate < 0.5:
        recs.append(
            "**Low citation relevance** ({:.1f}%): The retrieval index may lack relevant "
            "knowledge. Consider expanding the knowledge base with more SLAKE triples or "
            "adding textbook-style documents.".format(
                100 * agent_result.citation_relevance_hit_rate
            )
        )

    if agent_result.open_accuracy < 0.30:
        recs.append(
            "**Low open-ended accuracy** ({:.1f}%): Consider VLM fine-tuning on medical VQA "
            "data (VQA-RAD, SLAKE) to improve factual answer generation.".format(
                100 * agent_result.open_accuracy
            )
        )

    if agent_result.ece > 0.15:
        recs.append(
            "**High ECE** ({:.3f}): Confidence scores are poorly calibrated. "
            "Consider temperature scaling or Platt scaling post-hoc.".format(agent_result.ece)
        )

    if comparison is not None and comparison.accuracy_delta < 0:
        recs.append(
            "**RAG hurt accuracy** (delta={:+.1f}%): The retrieval step is introducing noise. "
            "Consider raising `supervisor_evidence_threshold` or improving the retrieval index "
            "quality.".format(100 * comparison.accuracy_delta)
        )

    if not recs:
        recs.append(
            "Performance looks reasonable. Focus on expanding the evaluation to additional "
            "datasets (PathVQA, SLAKE) and running ablations on retrieval top-k settings."
        )

    lines.extend(f"- {rec}" for rec in recs)
    return "\n".join(lines)


def _format_footer() -> str:
    return (
        "---\n\n"
        f"*Report generated by radiology-vqa evaluation framework · {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    )


def _format_threshold_recommendation(threshold_data: list[dict]) -> str:
    """Analyze threshold_analysis data and recommend optimal supervisor threshold.

    Find the threshold where accuracy is maximized while coverage >= 50%.
    """
    viable = [t for t in threshold_data if t["coverage"] >= 0.5 and t["count"] > 0]
    if not viable:
        return "Insufficient data to recommend a threshold (all have < 50% coverage)."

    best = max(viable, key=lambda t: t["accuracy"])
    return (
        f"At threshold **{best['threshold']:.2f}**, "
        f"accuracy={100*best['accuracy']:.1f}% with coverage={100*best['coverage']:.1f}% "
        f"({best['count']} samples). "
        f"Consider setting `supervisor_high_confidence={best['threshold']:.2f}`."
    )


def _format_error_samples(
    per_sample: list[PerSampleResult],
    category: str,
    baseline_per_sample: list[PerSampleResult] | None,
    max_samples: int = 10,
) -> str:
    """Format sample-level error analysis as markdown.

    category: "wrong" | "over_abstain" | "improved"
    """
    baseline_by_id: dict = {}
    if baseline_per_sample:
        baseline_by_id = {r.sample_id: r for r in baseline_per_sample}

    if category == "wrong":
        samples = [r for r in per_sample if not r.correct and r.decision != "abstain"]
    elif category == "over_abstain":
        samples = [
            r
            for r in per_sample
            if r.decision == "abstain" and baseline_by_id.get(r.sample_id, r).correct
        ]
    elif category == "improved":
        samples = [
            r
            for r in per_sample
            if r.correct and not baseline_by_id.get(r.sample_id, r).correct
        ]
    else:
        raise ValueError(f"Unknown category: {category!r}")

    samples = samples[:max_samples]
    if not samples:
        return f"*No samples in category '{category}'.*"

    lines = []
    for i, r in enumerate(samples, 1):
        vlm = baseline_by_id.get(r.sample_id)
        lines.append(f"**{i}.** `{r.sample_id}`")
        lines.append(f"- Q: {r.question}")
        lines.append(f"- GT: {r.ground_truth} | Pred: {r.prediction} | Conf: {r.confidence:.2f}")
        if vlm:
            lines.append(f"- VLM: {vlm.prediction}")
        if r.citations:
            lines.append(f"- Citations: {len(r.citations)} retrieved")
        lines.append("")

    return "\n".join(lines)
