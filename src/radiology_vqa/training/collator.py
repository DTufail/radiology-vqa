"""Data collator for LLaVA multimodal training batches."""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _to_multimodal_format(conversations: list[dict]) -> list[dict]:
    """Convert string-content conversation to LLaVA multimodal list-of-dicts format.

    Transformers >=4.45 requires content as typed blocks for LLaVA-Next:
        [{"type": "image"}, {"type": "text", "text": "..."}]

    build_conversation() stores content as a plain string (kept for test
    compatibility).  This function upgrades it at collation time without
    modifying the stored sample.
    """
    result = []
    for msg in conversations:
        content = msg["content"]
        if not isinstance(content, str):
            result.append(msg)  # already in list-of-dicts format
            continue
        if msg["role"] == "user":
            # Strip the <image> token; the image block is added explicitly.
            text = content.replace("<image>\n", "", 1).replace("<image>", "", 1)
            new_content: list[dict] = [
                {"type": "image"},
                {"type": "text", "text": text},
            ]
        else:
            new_content = [{"type": "text", "text": content}]
        result.append({"role": msg["role"], "content": new_content})
    return result


class LlavaDataCollator:
    """Collate multimodal samples into batches for LLaVA training.

    Handles variable-length image tokens and text tokens.
    Pads to max length within batch. Masks padding in labels with -100.

    Args:
        processor:   AutoProcessor for LLaVA (handles both text and image).
        max_length:  Maximum token length per sample (default 256).
    """

    def __init__(self, processor: Any, max_length: int = 2048) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts = []
        images = []

        for example in examples:
            text = self.processor.apply_chat_template(
                _to_multimodal_format(example["conversations"]),
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
