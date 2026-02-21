import logging
import time
from pathlib import Path

import numpy as np

from radiology_vqa.rag.document import RetrievalResult
from radiology_vqa.rag.indexer import FAISSIndexer

logger = logging.getLogger(__name__)


class Retriever:
    """Stateless query interface over a persisted FAISS index."""

    def __init__(self, index_dir: Path, embedder=None) -> None:
        """Load index from disk. If embedder not provided, create one from config."""
        if embedder is None:
            from radiology_vqa.rag.embedder import Embedder

            embedder = Embedder()

        self._embedder = embedder
        self._index, self._documents, meta = FAISSIndexer.load(index_dir)
        logger.info(
            "Retriever loaded: %d docs, dim=%d, model=%s",
            meta.get("doc_count", len(self._documents)),
            meta.get("dimension", -1),
            meta.get("embedding_model", "unknown"),
        )

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[RetrievalResult]:
        """Embed query, search index, return ranked results."""
        t0 = time.perf_counter()

        if self._index.ntotal == 0:
            logger.info("Index is empty — returning no results.")
            return []

        query_vec = self._embedder.embed_query(query)
        k = min(top_k, self._index.ntotal)
        scores_arr, indices_arr = self._index.search(query_vec.astype(np.float32), k)

        elapsed = time.perf_counter() - t0

        valid_pairs = [
            (float(s), int(i))
            for s, i in zip(scores_arr[0], indices_arr[0])
            if i >= 0 and float(s) >= min_score
        ]

        results = [
            RetrievalResult(
                document=self._documents[idx],
                score=score,
                rank=rank,
            )
            for rank, (score, idx) in enumerate(valid_pairs, start=1)
        ]

        top_score = results[0].score if results else 0.0
        logger.info(
            "Query: %.80r → %d results (top_score=%.4f, latency=%.3fs)",
            query,
            len(results),
            top_score,
            elapsed,
        )
        return results

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with optional metadata post-filter by source_type."""
        fetch_k = top_k * 3 if source_type else top_k
        raw = self.retrieve(query, top_k=fetch_k)

        if source_type is not None:
            raw = [r for r in raw if r.document.meta.source_type == source_type]

        # Re-rank after filter
        results = []
        for rank, r in enumerate(raw[:top_k], start=1):
            results.append(
                RetrievalResult(document=r.document, score=r.score, rank=rank)
            )
        return results
