"""Data collator for LLaVA multimodal training batches."""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class LlavaDataCollator:
    """Collate multimodal samples into batches for LLaVA training.

    Handles variable-length image tokens and text tokens.
    Pads to max length within batch. Masks padding in labels with -100.

    Args:
        processor:   AutoProcessor for LLaVA (handles both text and image).
        max_length:  Maximum token length per sample (default 256).
    """

    def __init__(self, processor: Any, max_length: int = 256) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts = []
        images = []

        for example in examples:
            text = self.processor.apply_chat_template(
                example["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            images.append(example["image"])

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
