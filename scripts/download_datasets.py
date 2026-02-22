"""Download VQA-RAD and PathVQA from HuggingFace. SLAKE is loaded locally — not downloaded here."""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from radiology_vqa.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _check_row(row: dict, dataset: str, split: str) -> None:
    """Raise if a sampled row is missing required fields."""
    if not row.get("question") or not row.get("answer"):
        raise RuntimeError(
            f"Sanity check failed for {dataset} {split}: "
            f"first row missing question or answer fields."
        )


def download_vqa_rad() -> None:
    """Trigger HuggingFace cache download for VQA-RAD without loading images into Python."""
    from datasets import load_dataset

    print("Downloading VQA-RAD...")
    for split in ["train", "test"]:
        # load_dataset caches to ~/.cache/huggingface; iterating is not required
        dataset = load_dataset(settings.vqa_rad_dataset, split=split)
        print(f"  VQA-RAD {split}: {len(dataset)} samples")
        _check_row(dataset[0], "VQA-RAD", split)
        print(f"  Sanity check OK: {len(dataset)} rows, first sample_id=vqa_rad_{split}_0")


def download_pathvqa() -> None:
    """Trigger HuggingFace cache download for PathVQA without loading images into Python."""
    from datasets import load_dataset

    print("Downloading PathVQA...")
    for split in ["train", "validation", "test"]:
        dataset = load_dataset(settings.pathvqa_dataset, split=split)
        print(f"  PathVQA {split}: {len(dataset)} samples")
        _check_row(dataset[0], "PathVQA", split)
        print(f"  Sanity check OK: {len(dataset)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VQA datasets from HuggingFace")
    parser.add_argument(
        "--dataset",
        choices=["vqa_rad", "pathvqa", "all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    args = parser.parse_args()

    tasks: dict = {}
    if args.dataset in ("vqa_rad", "all"):
        tasks["vqa_rad"] = download_vqa_rad
    if args.dataset in ("pathvqa", "all"):
        tasks["pathvqa"] = download_pathvqa

    if len(tasks) == 1:
        # Single dataset — run directly, no threading overhead
        list(tasks.values())[0]()
    else:
        # Multiple datasets — download in parallel (independent HF endpoints)
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"{name}: {e}")
                    logger.error("Download failed for %s: %s", name, e)

        if errors:
            raise RuntimeError(f"One or more downloads failed:\n" + "\n".join(errors))

    print(f"\nSLAKE must be downloaded manually.")
    print(f"Place Slake1.0/ directory at: {settings.slake_dir}")


if __name__ == "__main__":
    main()
