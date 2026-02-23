"""Run VLM benchmark on medical VQA datasets.

Usage:
    python scripts/run_benchmark.py                                      # VQA-RAD test, default backend
    python scripts/run_benchmark.py --dataset vqa_rad --split test       # explicit dataset + split
    python scripts/run_benchmark.py --max-samples 50                     # quick sanity check
    python scripts/run_benchmark.py --backend blip2                      # use BLIP-2 backend
    python scripts/run_benchmark.py --compare                            # print comparison table
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import settings

logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def _load_dataset(dataset: str, split: str):
    """Load a dataset split using Phase 1 loaders."""
    if dataset == "vqa_rad":
        from radiology_vqa.loader import load_vqa_rad

        print(f"Loading VQA-RAD ({split}) ...")
        return load_vqa_rad(split=split)

    if dataset == "slake":
        from radiology_vqa.slake_loader import load_slake

        print(f"Loading SLAKE ({split}) ...")
        return load_slake(settings.slake_dir, split=split)

    if dataset == "pathvqa":
        from radiology_vqa.loader import load_pathvqa

        print(f"Loading PathVQA ({split}) ...")
        return load_pathvqa(split=split)

    raise ValueError(f"Unknown dataset: {dataset!r}. Choices: vqa_rad, slake, pathvqa")


def _load_all_datasets(split: str):
    samples = []
    for ds in ("vqa_rad", "slake", "pathvqa"):
        try:
            samples.extend(_load_dataset(ds, split))
        except Exception as e:
            logger.warning("Could not load dataset %s: %s", ds, e)
    return samples


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _print_comparison(output_dir: Path) -> None:
    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        print(f"No benchmark results found in {output_dir}")
        return

    rows = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics", {})
        rows.append(
            {
                "model": data.get("model_name", "?"),
                "dataset": data.get("dataset", "?"),
                "split": data.get("split", "?"),
                "n": data.get("total_samples", 0),
                "overall": m.get("overall_accuracy", 0.0),
                "closed": m.get("closed_accuracy", 0.0),
                "open": m.get("open_accuracy", 0.0),
                "ms/sample": round(
                    data.get("runtime", {}).get("mean_latency_seconds", 0.0) * 1000, 1
                ),
                "file": path.name,
            }
        )

    # Header
    print(
        f"\n{'Model':<40} {'Dataset':<10} {'Split':<8} {'N':>5} "
        f"{'Overall':>8} {'Closed':>8} {'Open':>8} {'ms/samp':>9}"
    )
    print("-" * 105)
    for r in rows:
        print(
            f"{r['model']:<40} {r['dataset']:<10} {r['split']:<8} {r['n']:>5} "
            f"{r['overall']:>7.1%} {r['closed']:>7.1%} {r['open']:>7.1%} {r['ms/sample']:>8.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VLM benchmark on medical VQA datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="vqa_rad",
        choices=["vqa_rad", "slake", "pathvqa", "all"],
        help="Dataset to benchmark.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (train/validate/test).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N samples (quick runs).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["llava", "llava_med", "blip2"],
        help="Override the VLM backend from config.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a comparison table of all saved benchmark results and exit.",
    )
    args = parser.parse_args()

    output_dir = settings.benchmark_output_dir

    if args.compare:
        _print_comparison(output_dir)
        return

    # Override backend if requested
    if args.backend:
        settings.vlm_backend = args.backend
        if (
            args.backend == "blip2"
            and settings.vlm_model_id == "microsoft/llava-med-v1.5-mistral-7b"
        ):
            settings.vlm_model_id = "Salesforce/blip2-opt-2.7b"

    # Create VLM backend
    from radiology_vqa.vlm.factory import create_vlm_backend

    print(f"\nInitializing VLM backend: {settings.vlm_backend} ...")
    vlm = create_vlm_backend(settings)
    print(f"Backend ready: {vlm.model_name}")

    # Load samples
    if args.dataset == "all":
        samples = _load_all_datasets(args.split)
    else:
        samples = _load_dataset(args.dataset, args.split)

    print(f"Loaded {len(samples)} samples from {args.dataset} ({args.split}).")

    # Run benchmark
    from radiology_vqa.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(vlm)
    result = runner.run(
        samples,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )

    # Save result
    path = runner.save_result(result, output_dir)

    # Print summary
    m = result.metrics
    r = result.runtime
    print(f"\n{'='*60}")
    print(f"  Model:    {result.model_name}")
    print(f"  Dataset:  {result.dataset} ({result.split})")
    print(f"  Samples:  {result.total_samples}")
    print(f"  Overall accuracy:  {m['overall_accuracy']:.1%}  ({m['correct_total']}/{m['total']})")
    print(
        f"  Closed accuracy:   {m['closed_accuracy']:.1%}"
        f"  ({m['correct_closed']}/{m['total_closed']})"
    )
    print(
        f"  Open accuracy:     {m['open_accuracy']:.1%}"
        f"  ({m['correct_open']}/{m['total_open']})"
    )
    print(f"  Mean latency:  {r['mean_latency_seconds']*1000:.1f} ms/sample")
    print(f"  Throughput:    {r['samples_per_second']:.2f} samples/sec")
    print(f"  Saved to:      {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
