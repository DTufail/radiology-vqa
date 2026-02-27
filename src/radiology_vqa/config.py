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

    # Hybrid retrieval settings (Phase 6B-1)
    # "dense" preserves Phase 5 behaviour.  "hybrid" enables BM25 + dense + RRF.
    retrieval_method: str = "dense"
    bm25_index_dir: Path = Path("data/bm25_index")
    bm25_top_k: int = 20   # BM25 candidates before fusion
    dense_top_k: int = 20  # Dense candidates before fusion
    rrf_k: int = 60         # RRF smoothing constant (standard value)

    # VLM settings
    vlm_backend: str = "llava"  # "llava", "llava_med" (alias), or "blip2"
    vlm_model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    vlm_quantize: str = "4bit"  # "4bit", "8bit", "none"
    vlm_max_new_tokens: int = 128
    vlm_device: str = "auto"  # "auto", "cuda", "cpu"
    vlm_concise_mode: bool = True  # True: 1-5 word answers + max_new_tokens=32; False: verbose
    vlm_adapter_path: str = ""  # Path to LoRA adapter dir; empty = zero-shot (no adapter)

    # Agent / Supervisor settings
    supervisor_high_confidence: float = 0.85   # Case A/B threshold (above = high)
    supervisor_low_confidence: float = 0.55    # Case C/D/E threshold (below = low)
    supervisor_evidence_threshold: float = 0.4  # min retrieval score to count as supporting
    supervisor_min_supporting_evidence: int = 1  # minimum supporting docs to count as grounded
    supervisor_max_retries: int = 1             # max re_query attempts before abstain
    supervisor_semantic_threshold: float = 0.87  # min cosine similarity (Phase 6B-3 embedding agreement)
    # Calibrated on S-PubMedBert-MS-MARCO: SAME/SYNONYM pairs score 0.88–0.93,
    # DIFF medical pairs (pneumonia/kidney, cardiomegaly/pleural effusion) score 0.82–0.86.
    # Natural gap: 0.857–0.882; 0.87 is the midpoint.

    # Agreement method for supervisor (Phase 6D config support)
    # "embedding" = Phase 6B-3 cosine similarity (default, current behaviour).
    # "keyword"   = Phase 5 keyword/token matching (used for ablation configs 2 and 4).
    agreement_method: str = "embedding"

    # Calibration settings (Phase 6C)
    # "none" = Phase 5/6A/6B behaviour — no calibration (backward compatible).
    # "platt" = Platt scaling (2-param logistic fit on validation set).
    # "isotonic" = Isotonic regression (non-parametric, requires more data).
    calibration_method: str = "none"
    calibration_model_path: str = ""   # path to saved calibrator JSON

    # Benchmark settings
    benchmark_output_dir: Path = Path("./data/benchmarks")

    # Evaluation settings (Phase 5B)
    eval_output_dir: Path = Path("./data/evaluation_reports")
    eval_bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    eval_calibration_bins: int = 10
    eval_intermediate_save_interval: int = 50

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
