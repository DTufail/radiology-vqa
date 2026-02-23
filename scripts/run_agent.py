"""Run the multi-agent radiology VQA pipeline on individual samples or batches.

Usage:
    python scripts/run_agent.py --dataset vqa_rad --index 0
    python scripts/run_agent.py --dataset vqa_rad --index 0 --index 5 --index 10
    python scripts/run_agent.py --dataset vqa_rad --range 0 5
    python scripts/run_agent.py --image path/to/image.jpg --question "Is there a fracture?"
    python scripts/run_agent.py --dataset vqa_rad --range 0 20 --output results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SEP = "─" * 52


def _print_result(
    sample_id: str,
    question: str,
    ground_truth: str | None,
    result,
) -> None:
    print(_SEP)
    print(f"Sample:          {sample_id}")
    print(f"Question:        {question}")
    if ground_truth is not None:
        print(f"Ground truth:    {ground_truth}")
    print(_SEP)
    print(f"VLM answer:      {result.visual_answer}")
    print(f"Decision:        {result.decision}")
    print(f"Grounded answer: {result.answer}")
    print(f"Final confidence: {result.confidence:.3f}")
    if result.citations:
        print("Citations:")
        for i, c in enumerate(result.citations, 1):
            text_preview = c.get("text", "")[:80]
            score = c.get("score", 0.0)
            src = c.get("source_type", "")
            print(f"  [{i}] {text_preview} (score: {score:.2f}, {src})")
    print(f"Reasoning:       {result.reasoning}")
    print(f"Requires human review: {'Yes' if result.requires_human_review else 'No'}")
    print(_SEP)
    print()


def _load_samples(dataset: str, split: str) -> list:
    if dataset == "vqa_rad":
        from radiology_vqa.loader import load_vqa_rad
        return load_vqa_rad(split=split)
    elif dataset == "slake":
        from radiology_vqa.slake_loader import load_slake
        return load_slake(settings.slake_dir, split=split)
    else:
        from radiology_vqa.loader import load_pathvqa
        return load_pathvqa(split=split)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multi-agent radiology VQA pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        choices=["vqa_rad", "slake", "pathvqa"],
        help="Load samples from a HuggingFace dataset.",
    )
    source.add_argument(
        "--image",
        type=Path,
        help="Path to a local image file (use with --question).",
    )

    parser.add_argument("--question", default=None, help="Question (required with --image).")
    parser.add_argument("--answer-type", default="", choices=["open", "closed", ""],
                        help="Answer type (default: auto-infer).")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        dest="indices",
        metavar="N",
        help="Sample index to run (can be specified multiple times).",
    )
    parser.add_argument(
        "--range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Run samples in [START, END) range.",
    )
    parser.add_argument("--output", type=Path, help="Save results as JSON array to this file.")
    args = parser.parse_args()

    if not args.dataset and not args.image:
        # Default to vqa_rad index 0
        args.dataset = "vqa_rad"

    from radiology_vqa.graph.runner import create_runner

    print("Initialising agent pipeline…")
    runner = create_runner()
    print("Pipeline ready.\n")

    all_results: list[dict] = []

    if args.image:
        # ── Single-image mode ────────────────────────────────────────────────
        if not args.question:
            parser.error("--question is required when using --image")

        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        result = runner.run_query(image, args.question, answer_type=args.answer_type)
        _print_result(
            sample_id=args.image.stem,
            question=args.question,
            ground_truth=None,
            result=result,
        )
        all_results.append({
            "sample_id": args.image.stem,
            "question": args.question,
            "ground_truth": None,
            "result": result.model_dump(),
        })

    else:
        # ── Dataset mode ─────────────────────────────────────────────────────
        print(f"Loading {args.dataset}/{args.split}…")
        samples = _load_samples(args.dataset, args.split)
        print(f"Loaded {len(samples)} samples.\n")

        if args.range:
            start, end = args.range
            indices = list(range(start, min(end, len(samples))))
        elif args.indices:
            indices = args.indices
        else:
            indices = [0]

        for idx in indices:
            if idx >= len(samples):
                logger.warning(
                    "Index %d out of range (dataset has %d samples) — skipping.",
                    idx, len(samples),
                )
                continue

            sample = samples[idx]
            result = runner.run_query(
                sample.image,
                sample.question,
                answer_type=sample.answer_type,
            )
            sample_id = f"{args.dataset}_{args.split}_{idx}"
            _print_result(
                sample_id=sample_id,
                question=sample.question,
                ground_truth=sample.answer,
                result=result,
            )
            all_results.append({
                "sample_id": sample_id,
                "question": sample.question,
                "ground_truth": sample.answer,
                "answer_type": sample.answer_type,
                "result": result.model_dump(),
            })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
