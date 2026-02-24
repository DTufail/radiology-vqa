import logging
from typing import Optional

from radiology_vqa.config import settings
from radiology_vqa.schema import VQASample

logger = logging.getLogger(__name__)


def _infer_answer_type(answer: str) -> str:
    if answer.strip().lower() in {"yes", "no"}:
        return "closed"
    return "open"


def load_vqa_rad(split: str = "train", max_samples: Optional[int] = None) -> list[VQASample]:
    try:
        from datasets import load_dataset

        dataset = load_dataset(settings.vqa_rad_dataset, split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        logger.info("VQA-RAD %s: %d samples", split, len(dataset))
        samples = []
        for index, row in enumerate(dataset):
            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            samples.append(VQASample(
                image=image,
                question=row["question"],
                answer=str(row["answer"]),
                answer_type=_infer_answer_type(str(row["answer"])),
                modality="unknown",
                source="vqa_rad",
                sample_id=f"vqa_rad_{split}_{index}",
            ))
        return samples

    except Exception as e:
        logger.error("Failed to load VQA-RAD %s: %s", split, e)
        raise


def load_pathvqa(
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> list[VQASample]:
    """
    Load PathVQA samples.

    Args:
        split:       HuggingFace split name ("train", "validation", "test").
        max_samples: Cap the number of samples loaded. Useful for audits/debugging.
                     Set to e.g. 500 to avoid loading all 20k+ train images into RAM.
        streaming:   If True, use HuggingFace streaming mode (no full download into RAM).
                     Slightly slower per-sample but much lower peak memory.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            settings.pathvqa_dataset,
            split=split,
            streaming=streaming,
        )

        if not streaming:
            # dataset is a Dataset object with known length
            total = len(dataset)
            cap = min(max_samples, total) if max_samples is not None else total
            if max_samples is not None and max_samples < total:
                logger.info(
                    "PathVQA %s: capping at %d / %d samples (audit mode)",
                    split, cap, total,
                )
                dataset = dataset.select(range(cap))
            else:
                logger.info("PathVQA %s: %d samples", split, total)
        else:
            logger.info("PathVQA %s: streaming mode (length unknown)", split)

        samples = []
        for index, row in enumerate(dataset):
            if max_samples is not None and index >= max_samples:
                break  # safety guard for streaming mode

            # Convert lazily — only pays the cost when we actually iterate
            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            samples.append(VQASample(
                image=image,
                question=row["question"],
                answer=str(row["answer"]),
                answer_type=_infer_answer_type(str(row["answer"])),
                modality="pathology",
                source="pathvqa",
                sample_id=f"pathvqa_{split}_{index}",
            ))

            if index % 1000 == 0 and index > 0:
                logger.debug("PathVQA %s: loaded %d samples so far...", split, index)

        return samples

    except Exception as e:
        logger.error("Failed to load PathVQA %s: %s", split, e)
        raise


def load_all(
    vqa_rad_splits: tuple[str, ...] = ("train", "test"),
    slake_splits: tuple[str, ...] = ("train", "validation", "test"),
    pathvqa_splits: tuple[str, ...] = ("train",),
    max_pathvqa: Optional[int] = None,
) -> list[VQASample]:
    """
    Unified loader for all three datasets.
    Use max_pathvqa to cap PathVQA during audits (e.g. max_pathvqa=500).

    WARNING: Default splits include test data.  For Phase 6 training,
    use ``load_training_data()`` instead — it loads only train splits
    and never touches VQA-RAD test.
    """
    from radiology_vqa.slake_loader import load_slake

    all_samples: list[VQASample] = []

    for split in vqa_rad_splits:
        logger.info("Loading VQA-RAD %s...", split)
        all_samples.extend(load_vqa_rad(split))

    for split in slake_splits:
        logger.info("Loading SLAKE %s...", split)
        all_samples.extend(load_slake(settings.slake_dir, split))

    for split in pathvqa_splits:
        logger.info("Loading PathVQA %s...", split)
        all_samples.extend(load_pathvqa(split, max_samples=max_pathvqa))

    logger.info("load_all(): total %d samples", len(all_samples))
    return all_samples


def load_training_data(
    *,
    include_pathvqa: bool = True,
    max_pathvqa: Optional[int] = None,
    deduplicate: bool = True,
) -> list[VQASample]:
    """Load ONLY training splits — safe for Phase 6 fine-tuning.

    Datasets included:
    - VQA-RAD  ``train`` (1,793 samples)
    - SLAKE    ``train`` (≈4,918 English, with empty-answer / dup filtering)
    - PathVQA  ``train`` (19,654 samples, optional)

    VQA-RAD ``test`` is NEVER loaded — it is the sacred evaluation set.

    Args:
        include_pathvqa: Whether to include PathVQA train (default True).
        max_pathvqa:     Cap PathVQA samples (useful for debugging).
        deduplicate:     Remove exact (question, answer) duplicate pairs
                         within each dataset (default True).

    Returns:
        Combined list of ``VQASample`` objects ready for training.
    """
    from radiology_vqa.slake_loader import load_slake

    all_samples: list[VQASample] = []

    # VQA-RAD train only
    vqa_train = load_vqa_rad("train")
    all_samples.extend(vqa_train)
    logger.info("Training data: VQA-RAD train = %d", len(vqa_train))

    # SLAKE train only (empty answers & dup triples already handled in loader)
    slake_train = load_slake(settings.slake_dir, "train")
    all_samples.extend(slake_train)
    logger.info("Training data: SLAKE train = %d", len(slake_train))

    # PathVQA train
    if include_pathvqa:
        pathvqa_train = load_pathvqa("train", max_samples=max_pathvqa)
        all_samples.extend(pathvqa_train)
        logger.info("Training data: PathVQA train = %d", len(pathvqa_train))

    if deduplicate:
        before = len(all_samples)
        seen: set[tuple[str, str, str]] = set()
        deduped: list[VQASample] = []
        for s in all_samples:
            key = (s.source, s.question.strip().lower(), s.answer.strip().lower())
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        all_samples = deduped
        if before != len(all_samples):
            logger.info(
                "Training data: deduplication removed %d samples (%d → %d)",
                before - len(all_samples), before, len(all_samples),
            )

    logger.info("Training data: total = %d samples", len(all_samples))
    return all_samples