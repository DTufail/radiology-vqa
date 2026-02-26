import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from radiology_vqa.rag.document import Document, RetrievalResult
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


class HybridRetriever:
    """Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion (RRF).

    Wraps an existing Retriever (dense path) and adds a parallel BM25 path.
    Both paths retrieve a configurable number of candidates; results are merged
    via RRF, which operates on ranks rather than raw scores so that the
    incommensurable scales of BM25 (unbounded integers) and cosine similarity
    ([0, 1]) never need to be normalised against each other.

    The final scores returned to the caller are RRF scores divided by the
    maximum RRF score, giving values in [0, 1] with the same interface as
    Retriever.retrieve().

    Design rationale:
        Dense-only retrieval misses exact medical terminology (e.g. "consolidation")
        when the nearest semantic neighbours happen to be disease-entity documents
        that don't contain the term literally.  BM25 catches these via term
        frequency regardless of embedding proximity.  Dual membership in both
        ranked lists gets a double RRF contribution, so documents that are both
        semantically relevant AND contain the exact query terms rank highest.

    Args:
        dense_retriever: An existing Retriever instance.  Its internal
            _documents list is used to reconstruct RetrievalResult objects for
            any doc_id surfaced by BM25 that the dense retriever did not return.
        bm25_index_dir: Path to the directory produced by
            ``scripts/build_index.py --bm25``.  Must contain corpus.pkl and
            doc_ids.pkl (parallel lists).
        bm25_top_k: Number of BM25 candidates to retrieve before fusion.
        dense_top_k: Number of dense candidates to retrieve before fusion.
        rrf_k: RRF smoothing constant.  Standard value is 60 (Cormack et al.).
        min_score: Minimum normalised RRF score (exclusive) to include.
            Default 0.0 means all RRF-positive results are returned.
    """

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_index_dir: str,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        rrf_k: int = 60,
        min_score: float = 0.0,
    ) -> None:
        from rank_bm25 import BM25Okapi

        bm25_dir = Path(bm25_index_dir)
        with open(bm25_dir / "corpus.pkl", "rb") as f:
            corpus: list[list[str]] = pickle.load(f)
        with open(bm25_dir / "doc_ids.pkl", "rb") as f:
            self._bm25_doc_ids: list[str] = pickle.load(f)

        self._bm25 = BM25Okapi(corpus)
        self._dense_retriever = dense_retriever
        self._bm25_top_k = bm25_top_k
        self._dense_top_k = dense_top_k
        self._rrf_k = rrf_k
        self._min_score = min_score

        # Build doc_id → Document lookup from the dense retriever's document list.
        # Both BM25 and FAISS are built from the same documents list, so every
        # BM25 doc_id should resolve here.
        self._doc_store: dict[str, Document] = {
            doc.doc_id: doc for doc in dense_retriever._documents
        }

        logger.info(
            "HybridRetriever ready: %d BM25 docs, %d dense docs",
            len(corpus),
            len(self._doc_store),
        )

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float | None = None
    ) -> list[RetrievalResult]:
        """Retrieve top_k results using hybrid BM25 + dense retrieval with RRF.

        Returns the same type as Retriever.retrieve().  Scores are normalised
        to (0, 1] by dividing by the maximum RRF score in the candidate set.
        Results with normalised score <= min_score are excluded.

        Args:
            query: Free-text query string.
            top_k: Maximum number of results to return.
            min_score: Override the instance-level min_score for this call.
                Pass 0.0 to disable filtering (used by retrieval_agent_node).
        """
        t0 = time.perf_counter()
        effective_min_score = self._min_score if min_score is None else min_score

        # ── Step 1: Tokenise query ──────────────────────────────────────────
        query_tokens = query.lower().split()

        # ── Step 2: BM25 search ────────────────────────────────────────────
        bm25_scores = self._bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[-self._bm25_top_k:][::-1]
        bm25_hits: list[tuple[str, int]] = [
            (self._bm25_doc_ids[i], rank + 1)
            for rank, i in enumerate(top_bm25_indices)
            if bm25_scores[i] > 0  # discard zero-score docs (no query-term overlap)
        ]

        # ── Step 3: Dense search ───────────────────────────────────────────
        dense_results = self._dense_retriever.retrieve(query, top_k=self._dense_top_k)
        dense_hits: list[tuple[str, int]] = [
            (r.document.doc_id, rank + 1) for rank, r in enumerate(dense_results)
        ]

        # ── Step 4: RRF fusion ─────────────────────────────────────────────
        rrf_scores: dict[str, float] = defaultdict(float)
        for doc_id, rank in bm25_hits:
            rrf_scores[doc_id] += 1.0 / (rank + self._rrf_k)
        for doc_id, rank in dense_hits:
            rrf_scores[doc_id] += 1.0 / (rank + self._rrf_k)

        if not rrf_scores:
            logger.info("Hybrid retrieval: query=%.80r → 0 results", query)
            return []

        # ── Step 5: Sort descending, take top_k ───────────────────────────
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]

        # ── Step 6: Normalise to (0, 1] by dividing by maximum score ──────
        max_rrf = top_docs[0][1]

        # ── Step 7: Reconstruct RetrievalResult objects, apply min_score ──
        results: list[RetrievalResult] = []
        for rank, (doc_id, raw_rrf) in enumerate(top_docs, start=1):
            normalised = raw_rrf / max_rrf
            if normalised <= effective_min_score:
                continue
            if doc_id not in self._doc_store:
                logger.warning(
                    "doc_id %r in BM25 index but not in dense document store — skipping",
                    doc_id,
                )
                continue
            results.append(
                RetrievalResult(
                    document=self._doc_store[doc_id],
                    score=normalised,
                    rank=rank,
                )
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Hybrid retrieval: query=%.80r → %d results "
            "(bm25_hits=%d, dense_hits=%d, latency=%.3fs)",
            query,
            len(results),
            len(bm25_hits),
            len(dense_hits),
            elapsed,
        )
        return results
