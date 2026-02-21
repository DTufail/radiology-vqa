import logging
import time

import numpy as np

from radiology_vqa.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformers model for biomedical text embedding."""

    def __init__(self, model_name: str | None = None) -> None:
        """Load model. If model_name is None, use config default.
        Auto-detects device (cuda if available, else cpu)."""
        import torch
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name or settings.embedding_model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading embedding model %s on %s", self._model_name, device)
        self._model = SentenceTransformer(self._model_name, device=device)
        self._dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d", self._dimension)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed batch of texts. Returns (N, dim) float32 array, L2-normalized."""
        t0 = time.perf_counter()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Embedded %d texts in %.2fs", len(texts), elapsed)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query. Returns (1, dim) float32 array, L2-normalized."""
        t0 = time.perf_counter()
        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        logger.debug("Embedded query in %.3fs", elapsed)
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self._model_name
