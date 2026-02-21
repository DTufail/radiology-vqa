"""BLIP-2 OPT-2.7B inference backend."""

import logging
import time

import torch
from PIL import Image

from radiology_vqa.vlm.interface import VLMPrediction

logger = logging.getLogger(__name__)


class BLIP2Backend:
    """BLIP-2 (OPT-2.7B) inference backend.

    Uses Blip2ForConditionalGeneration from transformers.
    Supports 8-bit quantization via bitsandbytes (CUDA only).
    Falls back to fp32 on CPU automatically.
    General-purpose baseline for comparison against LLaVA-Med.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/blip2-opt-2.7b",
        quantize: str = "8bit",
        device: str = "auto",
        max_new_tokens: int = 64,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens

        cuda_available = torch.cuda.is_available()
        if not cuda_available and quantize != "none":
            logger.warning(
                "CUDA not available but quantize=%r requested. "
                "Falling back to fp32 on CPU.",
                quantize,
            )
            self._quantize = "none"
        else:
            self._quantize = quantize

        self._device = "cpu" if not cuda_available else device

        logger.info(
            "Loading BLIP-2: model=%s quantize=%s device=%s",
            model_id,
            self._quantize,
            self._device,
        )
        self._processor, self._model = self._load_model(model_id)
        logger.info("BLIP-2 loaded successfully.")

    def _load_model(self, model_id: str):
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers components: {e}. "
                "Ensure transformers>=4.37 is installed."
            ) from e

        try:
            processor = Blip2Processor.from_pretrained(model_id)
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
            model = Blip2ForConditionalGeneration.from_pretrained(model_id, **kwargs)
            if self._device == "cpu":
                model = model.to("cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_id}'. Error: {e}"
            ) from e

        model.eval()
        return processor, model

    def predict(self, image: Image.Image, question: str) -> VLMPrediction:
        """Run inference on a single image-question pair."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = f"Question: {question} Answer:"

        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        latency = time.perf_counter() - start

        # Strip the input tokens to get only generated text
        input_len = inputs.get("input_ids", torch.tensor([[]])).shape[1]
        generated_ids = output[0, input_len:]
        raw_output = self._processor.decode(generated_ids, skip_special_tokens=True).strip()
        answer = raw_output.lower().strip() or "unknown"

        return VLMPrediction(
            answer=answer,
            confidence=0.5,  # BLIP-2 generate() doesn't expose usable logprobs
            raw_output=raw_output,
            model_name=self.model_name,
            latency_seconds=latency,
        )

    def predict_batch(
        self, samples: list[tuple[Image.Image, str]]
    ) -> list[VLMPrediction]:
        """Sequential batch inference."""
        return [self.predict(image, question) for image, question in samples]

    @property
    def model_name(self) -> str:
        return f"blip2-opt-2.7b-{self._quantize}"
