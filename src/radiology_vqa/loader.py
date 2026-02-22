import logging

from radiology_vqa.config import settings
from radiology_vqa.schema import VQASample

logger = logging.getLogger(__name__)


def _infer_answer_type(answer: str) -> str:
    if answer.strip().lower() in {"yes", "no"}:
        return "closed"
    return "open"


def load_vqa_rad(split: str = "train") -> list[VQASample]:
    try:
        from datasets import load_dataset

        dataset = load_dataset(settings.vqa_rad_dataset, split=split)
        logger.info("VQA-RAD %s: %d samples", split, len(dataset))
        samples = []
        for index, row in enumerate(dataset):
            image = row["image"]
            # Guard: VQA-RAD can contain grayscale frames; models expect RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            sample = VQASample(
                image=image,
                question=row["question"],
                answer=str(row["answer"]),
                answer_type=_infer_answer_type(str(row["answer"])),
                modality="unknown",
                source="vqa_rad",
                sample_id=f"vqa_rad_{split}_{index}",
            )
            samples.append(sample)
        return samples
    except Exception as e:
        logger.error("Failed to load VQA-RAD %s: %s", split, e)
        raise


def load_pathvqa(split: str = "train") -> list[VQASample]:
    try:
        from datasets import load_dataset

        dataset = load_dataset(settings.pathvqa_dataset, split=split)
        logger.info("PathVQA %s: %d samples", split, len(dataset))
        samples = []
        for index, row in enumerate(dataset):
            image = row["image"].convert("RGB")
            sample = VQASample(
                image=image,
                question=row["question"],
                answer=str(row["answer"]),
                answer_type=_infer_answer_type(str(row["answer"])),
                modality="pathology",
                source="pathvqa",
                sample_id=f"pathvqa_{split}_{index}",
            )
            samples.append(sample)
        return samples
    except Exception as e:
        logger.error("Failed to load PathVQA %s: %s", split, e)
        raise
