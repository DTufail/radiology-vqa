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
        # Cache the resolved device once — avoids walking the parameter
        # iterator on every predict() call.
        self._inferred_device: torch.device = next(self._model.parameters()).device
        logger.info("BLIP-2 loaded on device=%s.", self._inferred_device)

    def _load_model(self, model_id: str):
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers components: {e}. "
                "Ensure transformers>=4.37 is installed."
            ) from e

        try:
            # use_fast=False avoids the "untagged enum ModelWrapper" error
            # caused by tokenizers library incompatibility with the fast
            # tokenizer file shipped with BLIP-2 OPT models.
            processor = Blip2Processor.from_pretrained(model_id, use_fast=False)
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
                # device_map={"": 0} places everything on GPU 0 directly without
                # going through accelerate's dispatch_model, which would call
                # model.to(device) and crash on already-quantized bitsandbytes models.
                kwargs["device_map"] = {"": 0}
                kwargs["attn_implementation"] = "sdpa"
            except ImportError:
                logger.warning("bitsandbytes not available; loading without quantization.")
                kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["torch_dtype"] = torch.float32

        if self._device != "cpu" and torch.cuda.is_available() and "device_map" not in kwargs:
            kwargs["device_map"] = self._device
            kwargs["attn_implementation"] = "sdpa"

        try:
            model = Blip2ForConditionalGeneration.from_pretrained(model_id, **kwargs)
            if self._device == "cpu":
                model = model.to("cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_id}'. Error: {e}"
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
        return self._infer_batch([image], [f"Question: {question} Answer:"])[0]

    def _infer_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[VLMPrediction]:
        """Core batched inference — shared by predict() and predict_batch().

        Args:
            images: RGB PIL images (already validated by callers).
            prompts: Formatted prompt strings, one per image.

        Returns:
            One :class:`VLMPrediction` per input pair, with latency split
            evenly across the batch.
        """
        inputs = self._processor(
            images=images, text=prompts, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._inferred_device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        # torch.inference_mode is strictly more efficient than no_grad for
        # pure inference: it additionally disables autograd version tracking.
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_latency = time.perf_counter() - start
        per_sample_latency = batch_latency / len(images)

        results: list[VLMPrediction] = []
        for idx in range(len(images)):
            generated_ids = output.sequences[idx, input_len:]
            raw_output = self._processor.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            answer = raw_output.strip() or "unknown"

            confidence = self._extract_confidence(output.scores, generated_ids, idx)

            results.append(
                VLMPrediction(
                    answer=answer,
                    confidence=confidence,
                    raw_output=raw_output,
                    model_name=self.model_name,
                    latency_seconds=per_sample_latency,
                )
            )

        return results

    def _extract_confidence(
        self, scores: tuple, generated_ids: torch.Tensor, sample_idx: int = 0
    ) -> float:
        """Compute mean token probability for a single sample in a batch.

        Uses a batched tensor operation:
          - stack scores:  (T, batch_size, vocab_size)
          - select sample: (T, vocab_size)
          - softmax once:  (T, vocab_size)
          - gather token probs: (T,)
          - mean → scalar
        """
        try:
            if not scores:
                return 0.5

            # scores: tuple of T tensors, each (batch_size, vocab_size)
            scores_tensor = torch.stack(scores)          # (T, B, vocab)
            sample_scores = scores_tensor[:, sample_idx, :]  # (T, vocab)
            probs = torch.softmax(sample_scores, dim=-1)     # (T, vocab)
            token_probs = probs[
                torch.arange(len(scores), device=probs.device),
                generated_ids[: len(scores)],
            ]  # (T,)
            return token_probs.mean().item()
        except Exception:
            return 0.5

    def predict_batch(
        self,
        samples: list[tuple[Image.Image, str]],
        batch_size: int = 8,
    ) -> list[VLMPrediction]:
        """True batched inference, processed in chunks of batch_size.

        Unlike LLaVA-Med, BLIP-2 handles fixed-size multimodal batches
        natively via processor padding. Larger batch_size improves GPU
        utilisation at the cost of peak VRAM.

        Args:
            samples: (image, question) pairs.
            batch_size: Number of samples per forward pass.
        """
        results: list[VLMPrediction] = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            images: list[Image.Image] = []
            prompts: list[str] = []
            for image, question in chunk:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)
                prompts.append(f"Question: {question} Answer:")
            results.extend(self._infer_batch(images, prompts))
        return results

    @property
    def model_name(self) -> str:
        return f"blip2-opt-2.7b-{self._quantize}"
