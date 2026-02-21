"""Factory for instantiating VLM backends from config."""

import logging

from radiology_vqa.config import Settings
from radiology_vqa.vlm.interface import VLMInterface

logger = logging.getLogger(__name__)


def create_vlm_backend(config: Settings) -> VLMInterface:
    """Instantiate the configured VLM backend.

    Reads ``config.vlm_backend`` to select the backend class.
    All other VLM settings (model_id, quantize, device, max_new_tokens)
    are forwarded to the backend constructor.

    Args:
        config: Application settings instance.

    Returns:
        An object satisfying :class:`VLMInterface`.

    Raises:
        ValueError: If ``config.vlm_backend`` is not a known backend name.
    """
    backend = config.vlm_backend
    logger.info("Creating VLM backend: %s", backend)

    if backend == "llava_med":
        from radiology_vqa.vlm.llava_med import LLaVAMedBackend

        return LLaVAMedBackend(
            model_id=config.vlm_model_id,
            quantize=config.vlm_quantize,
            device=config.vlm_device,
            max_new_tokens=config.vlm_max_new_tokens,
        )

    if backend == "blip2":
        from radiology_vqa.vlm.blip2 import BLIP2Backend

        # Use BLIP-2's own default model_id when the config still holds the
        # LLaVA-Med default (i.e. the user switched backend without updating model_id).
        model_id = config.vlm_model_id
        if model_id == "microsoft/llava-med-v1.5-mistral-7b":
            model_id = "Salesforce/blip2-opt-2.7b"
            logger.info(
                "vlm_backend=blip2 but vlm_model_id still points to llava-med; "
                "using BLIP-2 default model_id: %s",
                model_id,
            )

        return BLIP2Backend(
            model_id=model_id,
            quantize=config.vlm_quantize,
            device=config.vlm_device,
            max_new_tokens=config.vlm_max_new_tokens,
        )

    raise ValueError(
        f"Unknown VLM backend: {backend!r}. Supported values: 'llava_med', 'blip2'."
    )
