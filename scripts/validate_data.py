"""Validate all datasets are loadable and report statistics."""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import settings
from radiology_vqa.kg_loader import load_kg
from radiology_vqa.loader import load_pathvqa, load_vqa_rad
from radiology_vqa.schema import VQASample
from radiology_vqa.slake_loader import _SPLIT_FILE_MAP, load_slake

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_issues_found = False


def _flag_issue(msg: str) -> None:
    global _issues_found
    _issues_found = True
    logger.error(msg)


def _check_samples(samples: list[VQASample], dataset_name: str, split: str) -> None:
    empty_q = [s for s in samples if not s.question]
    empty_a = [s for s in samples if not s.answer]
    if empty_q:
        _flag_issue(f"{dataset_name} {split}: {len(empty_q)} samples with empty question")
    if empty_a:
        _flag_issue(f"{dataset_name} {split}: {len(empty_a)} samples with empty answer")


def validate_vqa_rad() -> None:
    print("\n=== VQA-RAD ===")
    total = 0
    for split in ["train", "test"]:
        samples = load_vqa_rad(split)
        print(f"  {split}: {len(samples)} samples")
        total += len(samples)
        _check_samples(samples, "VQA-RAD", split)
        if samples:
            at_dist = Counter(s.answer_type for s in samples)
            mod_dist = Counter(s.modality for s in samples)
            print(f"    answer_type: {dict(at_dist)}")
            print(f"    modality:    {dict(mod_dist)}")
    print(f"  Total: {total}")


def validate_pathvqa() -> None:
    print("\n=== PathVQA ===")
    total = 0
    for split in ["train", "validation", "test"]:
        samples = load_pathvqa(split)
        print(f"  {split}: {len(samples)} samples")
        total += len(samples)
        _check_samples(samples, "PathVQA", split)
        if samples:
            at_dist = Counter(s.answer_type for s in samples)
            mod_dist = Counter(s.modality for s in samples)
            print(f"    answer_type: {dict(at_dist)}")
            print(f"    modality:    {dict(mod_dist)}")
    print(f"  Total: {total}")


def validate_slake() -> None:
    print("\n=== SLAKE ===")
    if not settings.slake_dir.exists():
        print(f"  SLAKE directory not found: {settings.slake_dir}")
        print("  Skipping SLAKE validation.")
        return

    total = 0
    for split in ["train", "validation", "test"]:
        samples = load_slake(settings.slake_dir, split)
        print(f"  {split}: {len(samples)} samples")
        total += len(samples)
        _check_samples(samples, "SLAKE", split)
        if samples:
            at_dist = Counter(s.answer_type for s in samples)
            mod_dist = Counter(s.modality for s in samples)
            loc_dist = Counter(s.location for s in samples if s.location)
            ct_dist = Counter(s.content_type for s in samples if s.content_type)
            print(f"    answer_type:      {dict(at_dist)}")
            print(f"    modality:         {dict(mod_dist)}")
            print(f"    top locations:    {dict(loc_dist.most_common(5))}")
            print(f"    top content_type: {dict(ct_dist.most_common(5))}")
    print(f"  Total: {total}")

    # Verify all referenced images exist
    for split, filename in _SPLIT_FILE_MAP.items():
        json_path = settings.slake_dir / filename
        if not json_path.exists():
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                rows = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            _flag_issue(f"SLAKE {split}: failed to parse JSON: {e}")
            continue
        missing = [
            row.get("img_name", "")
            for row in rows
            if row.get("q_lang") == "en"
            and not (settings.slake_dir / "imgs" / row.get("img_name", "")).exists()
        ]
        if missing:
            _flag_issue(f"SLAKE {split}: {len(missing)} missing images: {missing[:5]}")


def validate_kg() -> None:
    print("\n=== SLAKE Knowledge Graph ===")
    if not settings.slake_dir.exists():
        print("  SLAKE directory not found, skipping KG validation.")
        return
    triples = load_kg(settings.slake_dir)
    cat_dist = Counter(t.category for t in triples)
    for cat, count in sorted(cat_dist.items()):
        print(f"  {cat}: {count} triples")
    print(f"  Total: {len(triples)} triples")


def main() -> None:
    validate_vqa_rad()
    validate_pathvqa()
    validate_slake()
    validate_kg()

    if _issues_found:
        print("\nCritical issues found. Exiting with code 1.")
        sys.exit(1)
    else:
        print("\nAll validations passed.")


if __name__ == "__main__":
    main()
