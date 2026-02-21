import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class VQASample(BaseModel):
    """Single VQA data point normalized across all datasets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any  # PIL.Image.Image
    question: str
    answer: str
    answer_type: str  # "closed" or "open"
    modality: str  # "xray", "ct", "mri", "pathology", "unknown"
    source: str  # "vqa_rad", "slake", "pathvqa"
    sample_id: str  # unique ID: "{source}_{split}_{index}"


class SLAKESample(VQASample):
    """SLAKE-specific fields preserved as extension."""

    location: str = ""
    content_type: str = ""
    triple: list[str] = []
    img_name: str = ""


class KGTriple(BaseModel):
    """Single knowledge graph triple from SLAKE KG."""

    head: str
    relation: str
    tail: str
    category: str  # "disease" or "organ" or "organ_rel"
