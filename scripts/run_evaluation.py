"""Run the full evaluation pipeline.

Usage examples::

    # Quick test (20 samples, ~4 minutes on GPU)
    python scripts/run_evaluation.py --mode compare --max-samples 20

    # Full agent evaluation only
    python scripts/run_evaluation.py --mode agent

    # Full VLM-only baseline only
    python scripts/run_evaluation.py --mode vlm_only

    # Both + comparison + report (the full deliverable)
    python scripts/run_evaluation.py --mode compare

    # Resume from crash
    python scripts/run_evaluation.py --mode agent --resume

    # Skip BERTScore (faster, saves GPU memory)
    python scripts/run_evaluation.py --mode compare --no-bertscore

    # Custom dataset
    python scripts/run_evaluation.py --mode compare --dataset vqa_rad --split test
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the evaluation pipeline for the radiology VQA system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "vlm_only", "compare"],
        default="compare",
        help="Evaluation mode: 'agent' only, 'vlm_only' only, or 'compare' (both + comparison + report)",
    )
    parser.add_argument("--dataset", default="vqa_rad", help="Dataset name (default: vqa_rad)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        help="Limit number of samples (default: all)",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        dest="no_bertscore",
        help="Skip BERTScore computation (faster, less GPU memory)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from intermediate checkpoint (if evaluation was interrupted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Directory for output files (default: from config)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from radiology_vqa.config import settings
    from radiology_vqa.evaluation.comparator import BaselineComparator
    from radiology_vqa.evaluation.evaluator import AgentEvaluator
    from radiology_vqa.evaluation.report import generate_report

    output_dir = args.output_dir or settings.eval_output_dir
    compute_bertscore = not args.no_bertscore
    evaluator = AgentEvaluator(settings)

    agent_result = None
    baseline_result = None

    # ── Run agent evaluation ─────────────────────────────────────────────────
    if args.mode in ("agent", "compare"):
        resume_from = None
        if args.resume:
            resume_from = (
                output_dir / f"intermediate_agent_{args.dataset}_{args.split}.json"
            )

        agent_result = evaluator.evaluate(
            dataset=args.dataset,
            split=args.split,
            mode="agent",
            max_samples=args.max_samples,
            compute_bertscore=compute_bertscore,
            save_intermediate=True,
            resume_from=resume_from,
        )

        ts = agent_result.timestamp[:10]
        agent_path = output_dir / f"agent_{args.dataset}_{args.split}_{ts}.json"
        agent_result.save(agent_path)
        print(f"[agent] Results saved: {agent_path}")

        # Free the AgentRunner (and its embedded VLM) before loading the
        # standalone VLM for vlm_only mode.  Without this, compare mode would
        # hold two full VLM instances in GPU memory simultaneously (~9 GB on T4).
        if args.mode == "compare":
            del evaluator
            evaluator = AgentEvaluator(settings)

    # ── Run VLM-only baseline ────────────────────────────────────────────────
    if args.mode in ("vlm_only", "compare"):
        resume_from = None
        if args.resume:
            resume_from = (
                output_dir / f"intermediate_vlm_only_{args.dataset}_{args.split}.json"
            )

        baseline_result = evaluator.evaluate(
            dataset=args.dataset,
            split=args.split,
            mode="vlm_only",
            max_samples=args.max_samples,
            compute_bertscore=compute_bertscore,
            save_intermediate=True,
            resume_from=resume_from,
        )

        ts = baseline_result.timestamp[:10]
        vlm_path = output_dir / f"vlm_{args.dataset}_{args.split}_{ts}.json"
        baseline_result.save(vlm_path)
        print(f"[vlm_only] Results saved: {vlm_path}")

    # ── Compare and generate report ──────────────────────────────────────────
    comparison = None
    if args.mode == "compare" and agent_result is not None and baseline_result is not None:
        print("\n[compare] Generating comparison…")
        comparator = BaselineComparator()
        comparison = comparator.compare(agent_result, baseline_result)

        p_str = f"p={comparison.mcnemar_p_value:.4f}"
        sig_str = "significant at p<0.05" if comparison.is_significant else "not significant"
        print(f"[compare] McNemar's test: {p_str} ({sig_str})")
        print(f"[compare] Net grounding improvement: {comparison.net_improvement:+d} samples")

        ts = agent_result.timestamp[:10]
        cmp_path = output_dir / f"comparison_{args.dataset}_{args.split}_{ts}.json"
        comparison.save(cmp_path)
        print(f"[compare] Comparison saved: {cmp_path}")

    # ── Generate markdown report ─────────────────────────────────────────────
    if agent_result is not None:
        print("\nGenerating report…")
        result_dir = generate_report(
            agent_result=agent_result,
            baseline_result=baseline_result,
            comparison=comparison,
            output_dir=output_dir,
        )
        print(f"Report saved: {result_dir / 'report.md'}")


if __name__ == "__main__":
    main()
