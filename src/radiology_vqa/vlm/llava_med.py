"""LLaVA-Med v1.5 Mistral-7B inference backend."""

import logging
import time

import torch
from PIL import Image

from radiology_vqa.vlm.interface import VLMPrediction

logger = logging.getLogger(__name__)


class LLaVAMedBackend:
    """LLaVA-Med v1.5 (Mistral-7B) inference backend.

    Uses LlavaForConditionalGeneration from transformers.
    Supports 4-bit / 8-bit quantization via bitsandbytes (CUDA only).
    Falls back to fp32 on CPU automatically.
    """

    def __init__(
        self,
        model_id: str = "microsoft/llava-med-v1.5-mistral-7b",
        quantize: str = "4bit",
        device: str = "auto",
        max_new_tokens: int = 128,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens

        cuda_available = torch.cuda.is_available()
        if not cuda_available and quantize != "none":
            logger.warning(
                "CUDA not available but quantize=%r requested. "
                "Falling back to fp32 on CPU — inference will be slow and may OOM.",
                quantize,
            )
            self._quantize = "none"
        else:
            self._quantize = quantize

        self._device = "cpu" if not cuda_available else device

        logger.info(
            "Loading LLaVA-Med: model=%s quantize=%s device=%s",
            model_id,
            self._quantize,
            self._device,
        )
        self._processor, self._model = self._load_model(model_id)
        logger.info("LLaVA-Med loaded successfully.")

    def _load_model(self, model_id: str):
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers components: {e}. "
                "Ensure transformers>=4.37 is installed."
            ) from e

        try:
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download/load processor for '{model_id}'. "
                f"Check your internet connection and model ID. Error: {e}"
            ) from e

        kwargs: dict = {}
        if self._quantize in ("4bit", "8bit") and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                if self._quantize == "4bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes not available; loading without quantization.")
                kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["torch_dtype"] = torch.float32

        if self._device != "cpu" and torch.cuda.is_available():
            kwargs["device_map"] = self._device

        try:
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
            if self._device == "cpu":
                model = model.to("cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_id}'. "
                "If this is an OOM error, try quantize='4bit' or quantize='8bit' on GPU. "
                f"Error: {e}"
            ) from e

        model.eval()

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info("GPU memory after model load: %.2f GB", vram_gb)

        return processor, model

    def predict(self, image: Image.Image, question: str) -> VLMPrediction:
        """Run inference on a single image-question pair."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = f"<image>\nUSER: {question}\nASSISTANT:"

        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        latency = time.perf_counter() - start

        # Decode only the generated tokens (strip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0, input_len:]
        raw_output = self._processor.decode(generated_ids, skip_special_tokens=True)
        answer = raw_output.strip() or "unknown"

        confidence = self._extract_confidence(output)

        return VLMPrediction(
            answer=answer,
            confidence=confidence,
            raw_output=raw_output,
            model_name=self.model_name,
            latency_seconds=latency,
        )

    def _extract_confidence(self, output) -> float:
        """Mean token probability from generation output scores."""
        try:
            scores = output.scores  # tuple of (1, vocab_size) tensors
            if not scores:
                return 0.5

            generated_ids = output.sequences[0, -len(scores):]
            probs = []
            for i, score in enumerate(scores):
                token_probs = torch.softmax(score, dim=-1)
                token_id = generated_ids[i]
                probs.append(token_probs[0, token_id].item())

            return float(sum(probs) / len(probs)) if probs else 0.5
        except Exception:
            return 0.5

    def predict_batch(
        self, samples: list[tuple[Image.Image, str]]
    ) -> list[VLMPrediction]:
        """Sequential batch inference (LLaVA-Med doesn't batch cleanly)."""
        return [self.predict(image, question) for image, question in samples]

    @property
    def model_name(self) -> str:
        return f"llava-med-v1.5-mistral-7b-{self._quantize}"
