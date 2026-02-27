#!/usr/bin/env python3
"""Phase 6D — Generate the 6-configuration ablation report.

Loads evaluation result JSONs for all 6 configs, computes per-component deltas,
runs McNemar's tests for key pairs, and writes a markdown report + JSON summary.

Usage
-----
    # After all 6 evaluation runs are complete:
    python scripts/generate_ablation_report.py \\
        --config1 data/evaluation_reports/vlm_vqa_rad_test_2026-02-27.json \\
        --config2 data/evaluation_reports/config2_baseline_agent/agent_vqa_rad_test_YYYY-MM-DD.json \\
        --config3 data/evaluation_reports/vlm_vqa_rad_test_2026-02-27-2.json \\
        --config4 data/evaluation_reports/config4_finetuned_agent/agent_vqa_rad_test_YYYY-MM-DD.json \\
        --config5 data/evaluation_reports/agent_result-3.json \\
        --config6 data/evaluation_reports/agent_result-4.json \\
        --output data/evaluation_reports/ablation/

    # With some configs missing (will show '—' in table):
    python scripts/generate_ablation_report.py \\
        --config1 data/evaluation_reports/vlm_vqa_rad_test_2026-02-27.json \\
        --config3 data/evaluation_reports/vlm_vqa_rad_test_2026-02-27-2.json \\
        --config5 data/evaluation_reports/agent_result-3.json \\
        --config6 data/evaluation_reports/agent_result-4.json \\
        --output data/evaluation_reports/ablation/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Config metadata ────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "num": 1,
        "label": "baseline_vlm",
        "vlm": "ZS",
        "lora": "—",
        "rag": "—",
        "kg": "—",
        "agreement": "—",
        "cal": "—",
        "arg": "config1",
    },
    {
        "num": 2,
        "label": "baseline_agent",
        "vlm": "ZS",
        "lora": "—",
        "rag": "Yes",
        "kg": "Original",
        "agreement": "Keyword",
        "cal": "—",
        "arg": "config2",
    },
    {
        "num": 3,
        "label": "finetuned_vlm",
        "vlm": "FT",
        "lora": "✓",
        "rag": "—",
        "kg": "—",
        "agreement": "—",
        "cal": "—",
        "arg": "config3",
    },
    {
        "num": 4,
        "label": "finetuned_agent",
        "vlm": "FT",
        "lora": "✓",
        "rag": "Yes",
        "kg": "Original",
        "agreement": "Keyword",
        "cal": "—",
        "arg": "config4",
    },
    {
        "num": 5,
        "label": "full_pipeline",
        "vlm": "FT",
        "lora": "✓",
        "rag": "Yes",
        "kg": "Expanded",
        "agreement": "Embed",
        "cal": "—",
        "arg": "config5",
    },
    {
        "num": 6,
        "label": "full_calibrated",
        "vlm": "FT",
        "lora": "✓",
        "rag": "Yes",
        "kg": "Expanded",
        "agreement": "Embed",
        "cal": "Isotonic",
        "arg": "config6",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_result(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        print(f"  [warn] File not found: {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def pct(val: float | None, digits: int = 1) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.{digits}f}%"


def fmt(val: float | None, digits: int = 3) -> str:
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


def delta(a: float | None, b: float | None) -> str:
    """Format b - a as a signed percentage delta."""
    if a is None or b is None:
        return "—"
    d = (b - a) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f} pp"


def mcnemar_test(result_a: dict, result_b: dict) -> dict | None:
    """Run McNemar's test comparing overall correctness between two evaluation results.

    Uses the per-sample 'correct' field from the samples list if available,
    otherwise approximates from aggregate accuracy and sample counts.
    """
    try:
        from scipy.stats import binom_test
    except ImportError:
        return {"error": "scipy not installed — run: pip install scipy"}

    samples_a = result_a.get("samples", [])
    samples_b = result_b.get("samples", [])

    if not samples_a or not samples_b or len(samples_a) != len(samples_b):
        # Fall back: approximate from aggregate stats only (no McNemar possible)
        return {
            "error": "per-sample data unavailable or mismatched — McNemar requires matched pairs"
        }

    # Build matched correctness vectors
    n_01 = sum(
        1 for a, b in zip(samples_a, samples_b)
        if not a.get("correct", False) and b.get("correct", False)
    )  # A wrong, B correct
    n_10 = sum(
        1 for a, b in zip(samples_a, samples_b)
        if a.get("correct", False) and not b.get("correct", False)
    )  # A correct, B wrong

    n = n_01 + n_10
    if n == 0:
        return {"n_discordant": 0, "p_value": 1.0, "significant": False, "n_01": 0, "n_10": 0}

    # McNemar: exact binomial when n < 25, chi-squared otherwise
    if n < 25:
        p_value = float(2 * binom_test(min(n_01, n_10), n, 0.5))
    else:
        chi2 = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, df=1))

    return {
        "n_discordant": n,
        "n_01": n_01,
        "n_10": n_10,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


# ── Report building ────────────────────────────────────────────────────────────

def build_ablation_table(results: list[dict | None]) -> str:
    lines = []
    lines.append("| # | Config | VLM | LoRA | RAG | KG | Agreement | Cal. | Overall Acc | Acc (ans) | Abstain | ECE | AUROC |")
    lines.append("|---|--------|-----|------|-----|-----|-----------|------|-------------|-----------|---------|-----|-------|")

    for cfg, r in zip(CONFIGS, results):
        if r is None:
            row = (
                f"| {cfg['num']} | {cfg['label']} | {cfg['vlm']} | {cfg['lora']} "
                f"| {cfg['rag']} | {cfg['kg']} | {cfg['agreement']} | {cfg['cal']} "
                f"| — | — | — | — | — |"
            )
        else:
            overall = r.get("overall_accuracy")
            ans = r.get("accuracy_when_answered") or r.get("overall_accuracy")
            abstain = r.get("abstention_rate", 0.0)
            ece = r.get("ece")
            auroc = r.get("confidence_auroc")

            # VLM-only mode: accuracy_when_answered == overall_accuracy, abstention = 0
            if r.get("evaluation_mode") == "vlm_only":
                ans = overall
                abstain = 0.0

            row = (
                f"| {cfg['num']} | {cfg['label']} | {cfg['vlm']} | {cfg['lora']} "
                f"| {cfg['rag']} | {cfg['kg']} | {cfg['agreement']} | {cfg['cal']} "
                f"| {pct(overall)} | {pct(ans)} | {pct(abstain)} | {fmt(ece)} | {fmt(auroc)} |"
            )
        lines.append(row)

    return "\n".join(lines)


def build_deltas(results: list[dict | None]) -> str:
    r = results  # alias: r[0]=config1, r[2]=config3, etc.

    def acc(ri): return ri.get("overall_accuracy") if ri else None
    def ans(ri): return ri.get("accuracy_when_answered") or ri.get("overall_accuracy") if ri else None
    def ece_v(ri): return ri.get("ece") if ri else None
    def auroc(ri): return ri.get("confidence_auroc") if ri else None
    def abstain(ri): return ri.get("abstention_rate", 0.0) if ri else None

    lines = []
    lines.append("| Component | Comparison | Overall Acc Δ | Acc (when answered) Δ | ECE Δ | AUROC Δ |")
    lines.append("|-----------|-----------|--------------|----------------------|-------|---------|")

    # Fine-tuning effect (VLM only): Config 3 vs 1
    lines.append(
        f"| Fine-tuning (VLM) | Config 3 vs 1 "
        f"| {delta(acc(r[0]), acc(r[2]))} | — "
        f"| {delta(ece_v(r[0]), ece_v(r[2]))} | {delta(auroc(r[0]), auroc(r[2]))} |"
    )

    # RAG effect on zero-shot: Config 2 vs 1
    lines.append(
        f"| RAG (zero-shot) | Config 2 vs 1 "
        f"| {delta(acc(r[0]), acc(r[1]))} | {delta(ans(r[0]), ans(r[1]))} "
        f"| {delta(ece_v(r[0]), ece_v(r[1]))} | {delta(auroc(r[0]), auroc(r[1]))} |"
    )

    # RAG effect on fine-tuned: Config 4 vs 3
    lines.append(
        f"| RAG (fine-tuned) | Config 4 vs 3 "
        f"| {delta(acc(r[2]), acc(r[3]))} | {delta(ans(r[2]), ans(r[3]))} "
        f"| {delta(ece_v(r[2]), ece_v(r[3]))} | {delta(auroc(r[2]), auroc(r[3]))} |"
    )

    # KG expansion + embedding agreement: Config 5 vs 4
    lines.append(
        f"| KG expansion + embedding agree. | Config 5 vs 4 "
        f"| {delta(acc(r[3]), acc(r[4]))} | {delta(ans(r[3]), ans(r[4]))} "
        f"| {delta(ece_v(r[3]), ece_v(r[4]))} | {delta(auroc(r[3]), auroc(r[4]))} |"
    )

    # Calibration effect: Config 6 vs 5
    lines.append(
        f"| Calibration (isotonic) | Config 6 vs 5 "
        f"| {delta(acc(r[4]), acc(r[5]))} | {delta(ans(r[4]), ans(r[5]))} "
        f"| {delta(ece_v(r[4]), ece_v(r[5]))} | {delta(auroc(r[4]), auroc(r[5]))} |"
    )

    # Full system vs baseline: Config 6 vs 1
    lines.append(
        f"| Full system vs. baseline | Config 6 vs 1 "
        f"| {delta(acc(r[0]), acc(r[5]))} | {delta(ans(r[0]), ans(r[5]))} "
        f"| {delta(ece_v(r[0]), ece_v(r[5]))} | {delta(auroc(r[0]), auroc(r[5]))} |"
    )

    return "\n".join(lines)


def build_question_type_table(results: list[dict | None]) -> str:
    lines = []
    lines.append("| # | Config | Closed Acc | Open Acc | Closed F1 | Closed Count | Open Count |")
    lines.append("|---|--------|-----------|---------|----------|------------|----------|")

    for cfg, r in zip(CONFIGS, results):
        if r is None:
            lines.append(f"| {cfg['num']} | {cfg['label']} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {cfg['num']} | {cfg['label']} "
                f"| {pct(r.get('closed_accuracy'))} "
                f"| {pct(r.get('open_accuracy'))} "
                f"| {fmt(r.get('closed_f1'), 3)} "
                f"| {r.get('closed_count', '—')} "
                f"| {r.get('open_count', '—')} |"
            )

    return "\n".join(lines)


def build_mcnemar_section(results: list[dict | None]) -> str:
    pairs = [
        ("Config 1 vs 6 (full system vs. baseline)", 0, 5),
        ("Config 3 vs 5 (does 6B pipeline help FT VLM?)", 2, 4),
        ("Config 5 vs 6 (does calibration change predictions?)", 4, 5),
        ("Config 2 vs 4 (fine-tuning effect on agent)", 1, 3),
    ]

    lines = []
    for label, i, j in pairs:
        ri, rj = results[i], results[j]
        if ri is None or rj is None:
            lines.append(f"- **{label}**: data not yet available")
            continue
        result = mcnemar_test(ri, rj)
        if "error" in result:
            lines.append(f"- **{label}**: {result['error']}")
        else:
            sig = "significant at p<0.05" if result["significant"] else "not significant"
            lines.append(
                f"- **{label}**: p={result['p_value']:.4f} ({sig}), "
                f"n_discordant={result['n_discordant']} "
                f"(A→B: {result['n_01']}, B→A: {result['n_10']})"
            )

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6D ablation report.")
    for cfg in CONFIGS:
        parser.add_argument(
            f"--{cfg['arg']}",
            type=Path,
            default=None,
            metavar="PATH",
            help=f"Config {cfg['num']} ({cfg['label']}) evaluation result JSON",
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_reports/ablation"),
        help="Output directory (default: data/evaluation_reports/ablation/)",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation results…")
    results: list[dict | None] = []
    for cfg in CONFIGS:
        path = getattr(args, cfg["arg"])
        r = load_result(path)
        status = "OK" if r else "MISSING"
        print(f"  Config {cfg['num']} ({cfg['label']}): {status}")
        results.append(r)

    n_available = sum(1 for r in results if r is not None)
    print(f"\n{n_available}/6 configs available.")

    # ── Build report sections ──────────────────────────────────────────────
    ablation_table = build_ablation_table(results)
    deltas_table = build_deltas(results)
    qtype_table = build_question_type_table(results)
    mcnemar_section = build_mcnemar_section(results)

    # ── Write markdown ─────────────────────────────────────────────────────
    report_path = output_dir / "ablation_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 6D — Ablation Report\n\n")
        f.write(f"**Dataset:** VQA-RAD test (451 samples)  \n")
        f.write(f"**Configs available:** {n_available}/6\n\n")
        f.write("---\n\n")

        f.write("## 6-Configuration Ablation Table\n\n")
        f.write(ablation_table)
        f.write("\n\n")

        f.write("> ZS = zero-shot (no LoRA adapter). FT = fine-tuned (QLoRA adapter loaded).\n")
        f.write("> Original KG = SLAKE KG only (2,987 docs). Expanded KG = SLAKE KG + RadLex + QA pseudo-docs (13,435 docs).\n")
        f.write("> Acc (ans) = accuracy on non-abstained samples only.\n\n")

        f.write("---\n\n")
        f.write("## Component Contribution (Deltas)\n\n")
        f.write(deltas_table)
        f.write("\n\n")

        f.write("---\n\n")
        f.write("## Per Question-Type Breakdown\n\n")
        f.write(qtype_table)
        f.write("\n\n")

        f.write("---\n\n")
        f.write("## Statistical Significance (McNemar's Test)\n\n")
        f.write(mcnemar_section)
        f.write("\n\n")

        f.write("---\n\n")
        f.write("## Commands to Run Missing Configs\n\n")
        f.write("```bash\n")
        f.write("# Config 2 — baseline_agent (zero-shot VLM + Phase 5 agent)\n")
        f.write("AGREEMENT_METHOD=keyword \\\n")
        f.write("RETRIEVAL_METHOD=dense \\\n")
        f.write("INDEX_DIR=data/indices \\\n")
        f.write("CALIBRATION_METHOD=none \\\n")
        f.write("SUPERVISOR_HIGH_CONFIDENCE=0.85 \\\n")
        f.write("SUPERVISOR_LOW_CONFIDENCE=0.55 \\\n")
        f.write("VLM_ADAPTER_PATH=\"\" \\\n")
        f.write("  python scripts/run_evaluation.py \\\n")
        f.write("    --mode agent --dataset vqa_rad --split test --no-bertscore \\\n")
        f.write("    --output-dir data/evaluation_reports/config2_baseline_agent/\n\n")

        f.write("# Config 4 — finetuned_agent (FT VLM + Phase 5 agent)\n")
        f.write("AGREEMENT_METHOD=keyword \\\n")
        f.write("RETRIEVAL_METHOD=dense \\\n")
        f.write("INDEX_DIR=data/indices \\\n")
        f.write("CALIBRATION_METHOD=none \\\n")
        f.write("SUPERVISOR_HIGH_CONFIDENCE=0.85 \\\n")
        f.write("SUPERVISOR_LOW_CONFIDENCE=0.55 \\\n")
        f.write("VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \\\n")
        f.write("  python scripts/run_evaluation.py \\\n")
        f.write("    --mode agent --dataset vqa_rad --split test --no-bertscore \\\n")
        f.write("    --output-dir data/evaluation_reports/config4_finetuned_agent/\n")
        f.write("```\n\n")

    print(f"Report written: {report_path}")

    # ── Write JSON summary ─────────────────────────────────────────────────
    summary = {
        "configs_available": n_available,
        "configs": [],
    }
    for cfg, r in zip(CONFIGS, results):
        entry = {"num": cfg["num"], "label": cfg["label"], "available": r is not None}
        if r:
            entry["overall_accuracy"] = r.get("overall_accuracy")
            entry["accuracy_when_answered"] = r.get("accuracy_when_answered")
            entry["abstention_rate"] = r.get("abstention_rate", 0.0)
            entry["ece"] = r.get("ece")
            entry["confidence_auroc"] = r.get("confidence_auroc")
            entry["closed_accuracy"] = r.get("closed_accuracy")
            entry["open_accuracy"] = r.get("open_accuracy")
            entry["closed_f1"] = r.get("closed_f1")
        summary["configs"].append(entry)

    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
