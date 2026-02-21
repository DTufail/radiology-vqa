"""Download VQA-RAD and PathVQA from HuggingFace. SLAKE is loaded locally — not downloaded here."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import settings
from radiology_vqa.loader import load_pathvqa, load_vqa_rad

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_vqa_rad() -> None:
    print("Downloading VQA-RAD...")
    for split in ["train", "test"]:
        samples = load_vqa_rad(split)
        print(f"  VQA-RAD {split}: {len(samples)} samples")
        if samples:
            s = samples[0]
            assert s.question and s.answer, f"Sanity check failed for VQA-RAD {split}"
            print(f"  Sanity check OK: sample_id={s.sample_id}")


def download_pathvqa() -> None:
    print("Downloading PathVQA...")
    for split in ["train", "validation", "test"]:
        samples = load_pathvqa(split)
        print(f"  PathVQA {split}: {len(samples)} samples")
        if samples:
            s = samples[0]
            assert s.question and s.answer, f"Sanity check failed for PathVQA {split}"
            print(f"  Sanity check OK: sample_id={s.sample_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VQA datasets from HuggingFace")
    parser.add_argument(
        "--dataset",
        choices=["vqa_rad", "pathvqa", "all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("vqa_rad", "all"):
        download_vqa_rad()

    if args.dataset in ("pathvqa", "all"):
        download_pathvqa()

    print(f"\nSLAKE must be downloaded manually.")
    print(f"Place Slake1.0/ directory at {settings.slake_dir}")


if __name__ == "__main__":
    main()
