"""VLM interface contract and structured prediction output."""

import logging
from typing import Optional, Protocol, runtime_checkable

from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VLMPrediction(BaseModel):
    """Structured output from any VLM backend."""

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_output: str
    model_name: str
    latency_seconds: float
    # Phase 6C: raw (pre-calibration) confidence score.
    # None when calibration is disabled (backward compatible).
    raw_confidence: Optional[float] = None


@runtime_checkable
class VLMInterface(Protocol):
    """Contract that all VLM backends must satisfy.

    Phase 4 Visual Agent depends on this interface, not on concrete
    backend classes. Swapping models = changing one config value.
    """

    def predict(self, image: Image.Image, question: str) -> VLMPrediction:
        """Run inference on a single image-question pair."""
        ...

    def predict_batch(
        self, samples: list[tuple[Image.Image, str]]
    ) -> list[VLMPrediction]:
        """Run inference on a batch.

        Default implementation: sequential predict() calls.
        Backends may override for true batched inference.
        """
        ...

    @property
    def model_name(self) -> str:
        """Identifiable model name string."""
        ...
