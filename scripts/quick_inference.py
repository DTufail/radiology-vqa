"""Quick single-sample VLM inference for testing.

Usage:
    python scripts/quick_inference.py --image path/to/image.jpg --question "Is there a fracture?"
    python scripts/quick_inference.py --dataset vqa_rad --index 0
    python scripts/quick_inference.py --dataset slake --index 5 --backend blip2
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import settings

logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single VLM inference for quick testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to an image file.")
    source.add_argument(
        "--dataset",
        choices=["vqa_rad", "slake", "pathvqa"],
        help="Load a sample from a dataset.",
    )

    parser.add_argument("--question", default=None, help="Question to ask (with --image).")
    parser.add_argument(
        "--index", type=int, default=0, help="Sample index within the dataset (with --dataset)."
    )
    parser.add_argument("--split", default="test", help="Dataset split (with --dataset).")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["llava", "llava_med", "blip2"],
        help="Override the VLM backend.",
    )
    args = parser.parse_args()

    # Override backend if requested
    if args.backend:
        settings.vlm_backend = args.backend

    # Resolve image and question
    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        question = args.question or "What do you observe in this image?"
        ground_truth = None
    else:
        # Load from dataset
        if args.dataset == "vqa_rad":
            from radiology_vqa.loader import load_vqa_rad

            samples = load_vqa_rad(split=args.split)
        elif args.dataset == "slake":
            from radiology_vqa.slake_loader import load_slake

            samples = load_slake(settings.slake_dir, split=args.split)
        else:
            from radiology_vqa.loader import load_pathvqa

            samples = load_pathvqa(split=args.split)

        if args.index >= len(samples):
            print(
                f"Index {args.index} out of range — dataset has {len(samples)} samples.",
                file=sys.stderr,
            )
            sys.exit(1)

        sample = samples[args.index]
        image = sample.image
        question = sample.question
        ground_truth = sample.answer

    # Initialize VLM
    from radiology_vqa.vlm.factory import create_vlm_backend

    print(f"\nInitializing {settings.vlm_backend} ...")
    vlm = create_vlm_backend(settings)
    print(f"Model: {vlm.model_name}\n")

    # Run inference
    print(f"Question: {question}")
    prediction = vlm.predict(image, question)

    print(f"\nAnswer:     {prediction.answer}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Latency:    {prediction.latency_seconds*1000:.1f} ms")

    if ground_truth is not None:
        from radiology_vqa.benchmark.metrics import is_match

        match = is_match(prediction.answer, ground_truth, "closed")
        print(f"\nGround truth: {ground_truth}")
        print(f"Correct:      {'✓' if match else '✗'}")


if __name__ == "__main__":
    main()
