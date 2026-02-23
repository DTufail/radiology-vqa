"""Generate report from previously saved evaluation results.

Usage::

    # From saved result files (agent + baseline)
    python scripts/generate_report.py \\
        --agent data/evaluation_reports/agent_vqa_rad_test.json \\
        --baseline data/evaluation_reports/vlm_vqa_rad_test.json

    # Agent-only report (no comparison section)
    python scripts/generate_report.py \\
        --agent data/evaluation_reports/agent_vqa_rad_test.json

    # With pre-computed comparison file
    python scripts/generate_report.py \\
        --agent data/evaluation_reports/agent_result.json \\
        --baseline data/evaluation_reports/baseline_result.json \\
        --comparison data/evaluation_reports/comparison.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation report from saved result files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--agent",
        type=Path,
        required=True,
        help="Path to agent EvaluationResult JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to VLM-only EvaluationResult JSON file (optional)",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
        help="Path to pre-computed ComparisonResult JSON file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Output directory (default: same directory as agent file)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from radiology_vqa.evaluation.comparator import BaselineComparator
    from radiology_vqa.evaluation.report import generate_report
    from radiology_vqa.evaluation.result import ComparisonResult, EvaluationResult

    print(f"Loading agent result: {args.agent}")
    agent_result = EvaluationResult.load(args.agent)

    baseline_result = None
    if args.baseline is not None:
        print(f"Loading baseline result: {args.baseline}")
        baseline_result = EvaluationResult.load(args.baseline)

    comparison = None
    if args.comparison is not None:
        print(f"Loading comparison: {args.comparison}")
        comparison = ComparisonResult.load(args.comparison)
    elif baseline_result is not None:
        print("Computing comparison…")
        comparator = BaselineComparator()
        comparison = comparator.compare(agent_result, baseline_result)

    output_dir = args.output_dir or args.agent.parent

    result_dir = generate_report(
        agent_result=agent_result,
        baseline_result=baseline_result,
        comparison=comparison,
        output_dir=output_dir,
    )
    print(f"Report saved: {result_dir / 'report.md'}")


if __name__ == "__main__":
    main()
