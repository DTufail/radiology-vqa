"""Training dataset for QLoRA fine-tuning of LLaVA on medical VQA.

Combines VQA-RAD train, SLAKE train (English), and PathVQA train into a single
HuggingFace Dataset with LLaVA conversation format.

Usage:
    from radiology_vqa.training.dataset import build_training_dataset
    train_ds, val_ds = build_training_dataset()
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training data preparation."""

    include_vqa_rad: bool = True
    include_slake: bool = True
    include_pathvqa: bool = True
    seed: int = 42
    max_answer_length: int = 50  # truncate answers longer than this (words)
    normalize_answers: bool = True


def normalize_answer(answer: str) -> str:
    """Normalize answer for training: lowercase and strip whitespace only.

    Unlike evaluation normalization, we keep punctuation and articles
    because the model should learn natural answer patterns.
    """
    return answer.lower().strip()


def build_conversation(question: str, answer: str) -> list[dict]:
    """Format a QA pair as LLaVA conversation.

    Returns list of message dicts matching LLaVA chat template format.
    The <image> token tells the model where the image embedding goes.

    The instruction suffix matches the concise_mode inference prompt in
    llava.py LLaVABackend.predict() so training and inference formats align.
    """
    return [
        {
            "role": "user",
            "content": (
                f"<image>\n{question}\n"
                "Provide only the direct answer in 1-5 words. Do not explain."
            ),
        },
        {"role": "assistant", "content": answer},
    ]


def build_training_dataset(
    config: Optional[TrainingConfig] = None,
    slake_dir: Optional[Path] = None,
) -> tuple[Dataset, Dataset]:
    """Build training and validation HuggingFace Datasets.

    Training: VQA-RAD train + SLAKE train (EN) + PathVQA train
    Validation: SLAKE validation (EN) — ~1,053 samples

    VQA-RAD test (451 samples) is NEVER included in training or validation.

    Args:
        config:    TrainingConfig controlling which datasets to include.
        slake_dir: Path to Slake1.0 directory. Defaults to settings.slake_dir.

    Returns:
        (train_dataset, val_dataset) as HuggingFace Dataset objects.
        Each sample has keys: "image", "conversations", "source", "sample_id"

    The returned Datasets are ready for SFTTrainer with a custom data collator.
    """
    if config is None:
        config = TrainingConfig()

    if slake_dir is None:
        from radiology_vqa.config import Settings
        slake_dir = Settings().slake_dir

    train_samples: list[dict] = []
    val_samples: list[dict] = []

    # --- VQA-RAD train ---
    if config.include_vqa_rad:
        from radiology_vqa.loader import load_vqa_rad

        vqa_train = load_vqa_rad("train")
        before = len(train_samples)
        for s in vqa_train:
            answer = normalize_answer(s.answer) if config.normalize_answers else s.answer
            if not answer:
                continue
            if len(answer.split()) > config.max_answer_length:
                answer = " ".join(answer.split()[: config.max_answer_length])
            train_samples.append(
                {
                    "image": s.image,
                    "conversations": build_conversation(s.question, answer),
                    "source": s.source,
                    "sample_id": s.sample_id,
                }
            )
        logger.info(
            "VQA-RAD train: %d loaded, %d added to training set",
            len(vqa_train),
            len(train_samples) - before,
        )

    # --- SLAKE train + validation ---
    if config.include_slake:
        from radiology_vqa.slake_loader import load_slake

        slake_train = load_slake(slake_dir, "train")
        before = len(train_samples)
        for s in slake_train:
            answer = normalize_answer(s.answer) if config.normalize_answers else s.answer
            if not answer:
                continue
            if len(answer.split()) > config.max_answer_length:
                answer = " ".join(answer.split()[: config.max_answer_length])
            train_samples.append(
                {
                    "image": s.image,
                    "conversations": build_conversation(s.question, answer),
                    "source": s.source,
                    "sample_id": s.sample_id,
                }
            )
        logger.info(
            "SLAKE train: %d loaded, %d added to training set",
            len(slake_train),
            len(train_samples) - before,
        )

        # SLAKE validation = our validation set
        slake_val = load_slake(slake_dir, "validation")
        for s in slake_val:
            answer = normalize_answer(s.answer) if config.normalize_answers else s.answer
            if not answer:
                continue
            val_samples.append(
                {
                    "image": s.image,
                    "conversations": build_conversation(s.question, answer),
                    "source": s.source,
                    "sample_id": s.sample_id,
                }
            )
        logger.info(
            "SLAKE validation: %d loaded, %d added to validation set",
            len(slake_val),
            len(val_samples),
        )

    logger.info("Total non-PathVQA training samples: %d", len(train_samples))
    logger.info("Total validation samples: %d", len(val_samples))

    # Build the VQA-RAD + SLAKE portion as a Dataset (small — fits in RAM).
    vqa_slake_ds = Dataset.from_list(train_samples)
    val_ds = Dataset.from_list(val_samples)

    if not config.include_pathvqa:
        return vqa_slake_ds, val_ds

    # --- PathVQA train (generator-based to avoid 19k PIL images in heap) ---
    # Dataset.from_list() requires all samples in memory before Arrow serialisation.
    # For PathVQA (19,654 images × ~450 KB avg ≈ 8.8 GB) this would OOM on
    # ml.g4dn.2xlarge (32 GB RAM) once combined with model weights during training.
    # Dataset.from_generator() streams rows into Arrow one at a time.
    pathvqa_count = 0

    def _pathvqa_generator():
        nonlocal pathvqa_count
        try:
            from datasets import load_dataset as hf_load_dataset

            ds = hf_load_dataset("flaviagiammarino/path-vqa", split="train")
            for i, row in enumerate(ds):
                img = row["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                answer = (
                    normalize_answer(row["answer"])
                    if config.normalize_answers
                    else row["answer"]
                )
                if not answer:
                    continue
                if len(answer.split()) > config.max_answer_length:
                    answer = " ".join(answer.split()[: config.max_answer_length])
                pathvqa_count += 1
                if i % 5000 == 0 and i > 0:
                    logger.debug("PathVQA: processed %d samples so far", i)
                yield {
                    "image": img,
                    "conversations": build_conversation(row["question"], answer),
                    "source": "pathvqa",
                    "sample_id": f"pathvqa_train_{i}",
                }
        except Exception as e:
            logger.error("Failed to load PathVQA train: %s", e)
            raise

    try:
        from datasets import concatenate_datasets

        pathvqa_ds = Dataset.from_generator(_pathvqa_generator)
        logger.info("PathVQA train: %d added to training set", pathvqa_count)

        train_ds = concatenate_datasets([vqa_slake_ds, pathvqa_ds])
    except Exception as e:
        logger.error("PathVQA generator failed: %s", e)
        raise

    logger.info("Total training samples: %d", len(train_ds))

    return train_ds, val_ds
