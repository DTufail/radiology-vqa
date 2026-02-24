"""Data collator for LLaVA multimodal training batches."""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Mistral [INST] / [/INST] boundary token used to locate where the assistant
# answer starts.  We tokenise this marker once at collator init to avoid
# repeated tokeniser calls.
_INST_END_MARKER = "[/INST]"


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


def _find_answer_start(
    input_ids: torch.Tensor,
    inst_end_token_ids: list[int],
) -> int:
    """Return the token index immediately after the last [/INST] marker.

    The Mistral chat template wraps user turns in [INST]...[/INST].
    The assistant answer tokens begin right after the closing [/INST].

    Args:
        input_ids:  1-D tensor of token IDs for one sample.
        inst_end_token_ids:  Token IDs of the [/INST] marker.

    Returns:
        Index of the first answer token. If marker is not found, returns 0
        (fall-back: train on full sequence rather than silently losing data).
    """
    ids = input_ids.tolist()
    marker_len = len(inst_end_token_ids)
    # Search from the END so that if there are multiple [/INST] blocks
    # (multi-turn), we find the LAST one (which precedes the final answer).
    for i in range(len(ids) - marker_len, -1, -1):
        if ids[i : i + marker_len] == inst_end_token_ids:
            return i + marker_len
    return 0  # marker not found — don't mask anything


class LlavaDataCollator:
    """Collate multimodal samples into batches for LLaVA training.

    Handles variable-length image tokens and text tokens.
    Pads to max length within batch.

    Label masking strategy:
        1. Padding tokens  → -100  (never contribute to loss)
        2. Prompt tokens   → -100  (system + image + question + instruction)
        3. Answer tokens   → kept  (model learns to predict the answer + EOS)

    This ensures the loss signal is focused on the 1-5 word medical VQA
    answer rather than diluted across ~3,000 image/prompt tokens.

    Args:
        processor:   AutoProcessor for LLaVA (handles both text and image).
        max_length:  Maximum token length per sample (default 2048).
    """

    def __init__(self, processor: Any, max_length: int = 2048) -> None:
        self.processor = processor
        self.max_length = max_length

        # Pre-tokenise the [/INST] marker so we can locate it in token IDs.
        # encode() may prepend BOS — strip it if present.
        raw_ids = self.processor.tokenizer.encode(
            _INST_END_MARKER, add_special_tokens=False
        )
        self._inst_end_ids: list[int] = raw_ids
        logger.debug("[/INST] marker token IDs: %s", self._inst_end_ids)

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

        # Build labels: mask padding and all prompt tokens, keep answer only.
        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id

        for i in range(labels.size(0)):
            # Mask padding
            labels[i][labels[i] == pad_token_id] = -100

            # Find where the assistant answer starts (after [/INST])
            answer_start = _find_answer_start(
                batch["input_ids"][i], self._inst_end_ids
            )
            # Mask everything before the answer (system + image + question)
            if answer_start > 0:
                labels[i, :answer_start] = -100

        batch["labels"] = labels

        return batch
