import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    data_dir: Path = Path("./data")
    slake_dir: Path = Path("./data/raw/Slake1.0")

    # HuggingFace dataset IDs
    vqa_rad_dataset: str = "flaviagiammarino/vqa-rad"
    pathvqa_dataset: str = "flaviagiammarino/path-vqa"

    # RAG settings
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    index_dir: Path = Path("./data/indices")
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.3

    # VLM settings
    vlm_backend: str = "llava_med"  # "llava_med" or "blip2"
    vlm_model_id: str = "microsoft/llava-med-v1.5-mistral-7b"
    vlm_quantize: str = "4bit"  # "4bit", "8bit", "none"
    vlm_max_new_tokens: int = 128
    vlm_device: str = "auto"  # "auto", "cuda", "cpu"

    # Benchmark settings
    benchmark_output_dir: Path = Path("./data/benchmarks")

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
