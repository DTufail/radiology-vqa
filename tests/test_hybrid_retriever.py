"""Tests for HybridRetriever (BM25 + dense + RRF) and GraphBuilder factory dispatch.

All tests are fast (no GPU, no model downloads, no real FAISS index).
BM25 index files are written to tmp_path fixtures.
Dense retrieval is mocked via MockDenseRetriever.
"""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Allow importing build_bm25_index from scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from radiology_vqa.rag.document import Document, DocumentMeta, RetrievalResult
from radiology_vqa.rag.retriever import HybridRetriever


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_doc(text: str, doc_id: str) -> Document:
    return Document(
        text=text,
        meta=DocumentMeta(
            source_type="kg_disease",
            entity_name="test",
            attribute="symptom",
            source_file="test.csv",
        ),
        doc_id=doc_id,
    )


def _make_result(doc: Document, score: float, rank: int) -> RetrievalResult:
    return RetrievalResult(document=doc, score=score, rank=rank)


def _write_bm25_index(docs: list, bm25_dir: Path) -> str:
    """Write corpus.pkl and doc_ids.pkl directly (mirrors build_bm25_index logic)."""
    bm25_dir.mkdir(parents=True, exist_ok=True)
    corpus = [doc.text.lower().split() for doc in docs]
    doc_ids = [doc.doc_id for doc in docs]
    with open(bm25_dir / "corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    with open(bm25_dir / "doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)
    return str(bm25_dir)


class MockDenseRetriever:
    """Mimics Retriever._documents + retrieve() without loading a real FAISS index."""

    def __init__(self, docs: list, results: list | None = None):
        self._documents = docs
        self._results = results or []

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list:
        return self._results[:top_k]


# ── Test 1: build_bm25_index produces parallel files ──────────────────────────


class TestBuildBm25Index:
    def test_bm25_index_files_are_parallel(self, tmp_path):
        """corpus.pkl and doc_ids.pkl exist and are index-aligned (same length)."""
        from build_index import build_bm25_index

        docs = [_make_doc(f"text about finding {i}", f"doc_{i}") for i in range(7)]
        bm25_dir = tmp_path / "bm25"
        build_bm25_index(docs, str(bm25_dir))

        corpus_path = bm25_dir / "corpus.pkl"
        doc_ids_path = bm25_dir / "doc_ids.pkl"

        assert corpus_path.exists()
        assert doc_ids_path.exists()

        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
        with open(doc_ids_path, "rb") as f:
            doc_ids = pickle.load(f)

        assert len(corpus) == len(doc_ids) == 7


# ── Tests 2–4: return shape and uniqueness ─────────────────────────────────────


class TestReturnShape:
    def _build(self, docs, dense_results, tmp_path, **kwargs):
        bm25_dir = _write_bm25_index(docs, tmp_path / "bm25")
        mock_dense = MockDenseRetriever(docs=docs, results=dense_results)
        return HybridRetriever(
            dense_retriever=mock_dense,
            bm25_index_dir=bm25_dir,
            **kwargs,
        )

    def test_hybrid_retrieve_returns_top_k(self, tmp_path):
        """retrieve(query, top_k=5) returns exactly 5 results when ≥5 docs match."""
        docs = [_make_doc(f"lung finding {i}", f"doc_{i}") for i in range(20)]
        dense_results = [_make_result(d, 0.9 - i * 0.02, i + 1) for i, d in enumerate(docs[:5])]
        retriever = self._build(docs, dense_results, tmp_path, min_score=0.0)

        results = retriever.retrieve("lung finding", top_k=5)

        assert len(results) == 5

    def test_hybrid_retrieve_score_range(self, tmp_path):
        """All returned RetrievalResult.score values are in [0.0, 1.0]."""
        docs = [_make_doc(f"chest xray {i}", f"doc_{i}") for i in range(10)]
        dense_results = [_make_result(d, 0.9 - i * 0.05, i + 1) for i, d in enumerate(docs[:5])]
        retriever = self._build(docs, dense_results, tmp_path, min_score=0.0)

        results = retriever.retrieve("chest xray", top_k=5)

        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_hybrid_retrieve_doc_ids_unique(self, tmp_path):
        """No duplicate doc_id values in returned results."""
        docs = [_make_doc(f"pneumonia {i}", f"doc_{i}") for i in range(15)]
        dense_results = [_make_result(d, 0.9 - i * 0.05, i + 1) for i, d in enumerate(docs[:5])]
        retriever = self._build(docs, dense_results, tmp_path, min_score=0.0)

        results = retriever.retrieve("pneumonia", top_k=5)

        doc_ids = [r.document.doc_id for r in results]
        assert len(doc_ids) == len(set(doc_ids))


# ── Test 5: RRF dual-hit ranking ───────────────────────────────────────────────


class TestRRFFusion:
    def test_rrf_dual_hits_rank_higher(self, tmp_path):
        """A doc in BOTH BM25 and dense results ranks above one in only one list.

        doc_dual:        contains exact query term "consolidation" (BM25 hit rank 1)
                         AND is returned by mock dense at rank 1.
                         RRF score: 1/(1+60) + 1/(1+60) = 2/61

        doc_dense_only:  no query-term overlap in text (BM25 score = 0, excluded from BM25 hits)
                         but returned by mock dense at rank 2.
                         RRF score: 0 + 1/(2+60) = 1/62

        After normalisation: doc_dual → 1.0, doc_dense_only → 61/(62*2) ≈ 0.492
        """
        docs = [
            _make_doc("consolidation in the lung is a radiological finding", "doc_dual"),
            _make_doc("opacity shadows visible on xray scan", "doc_dense_only"),
            _make_doc("atelectasis partial collapse lung segment", "doc_bm25_only"),
            _make_doc("unrelated text about something else entirely", "doc_neither"),
        ]
        # Dense returns doc_dual (rank 1) and doc_dense_only (rank 2)
        dense_results = [
            _make_result(docs[0], 0.95, 1),
            _make_result(docs[1], 0.90, 2),
        ]
        bm25_dir = _write_bm25_index(docs, tmp_path / "bm25")
        mock_dense = MockDenseRetriever(docs=docs, results=dense_results)
        retriever = HybridRetriever(
            dense_retriever=mock_dense,
            bm25_index_dir=bm25_dir,
            bm25_top_k=10,
            dense_top_k=10,
            rrf_k=60,
            min_score=0.0,
        )

        results = retriever.retrieve("consolidation", top_k=4)
        result_ids = [r.document.doc_id for r in results]

        assert "doc_dual" in result_ids
        assert "doc_dense_only" in result_ids
        # doc_dual must rank above doc_dense_only
        assert result_ids.index("doc_dual") < result_ids.index("doc_dense_only")


# ── Tests 6–7: min_score filter ────────────────────────────────────────────────


class TestMinScoreFilter:
    def _make_retriever(self, tmp_path, min_score):
        docs = [_make_doc(f"finding {i}", f"doc_{i}") for i in range(5)]
        dense_results = [_make_result(d, 0.9 - i * 0.1, i + 1) for i, d in enumerate(docs)]
        bm25_dir = _write_bm25_index(docs, tmp_path / "bm25")
        mock_dense = MockDenseRetriever(docs=docs, results=dense_results)
        return HybridRetriever(
            dense_retriever=mock_dense,
            bm25_index_dir=bm25_dir,
            min_score=min_score,
        )

    def test_min_score_filter_zero(self, tmp_path):
        """With min_score=0.0, no results are excluded (all normalised scores > 0.0)."""
        retriever = self._make_retriever(tmp_path, min_score=0.0)
        results = retriever.retrieve("finding", top_k=5)
        assert len(results) == 5

    def test_min_score_filter_one(self, tmp_path):
        """With min_score=1.0, all results are excluded (none strictly above 1.0).

        After normalisation, the top result scores exactly 1.0.
        The filter condition is score > min_score (strictly greater), so
        score=1.0 with min_score=1.0 fails the filter.
        """
        retriever = self._make_retriever(tmp_path, min_score=1.0)
        results = retriever.retrieve("finding", top_k=5)
        assert results == []


# ── Tests 8–9: BM25-specific behaviour ────────────────────────────────────────


class TestBM25Behaviour:
    def test_bm25_exact_term_match(self, tmp_path):
        """A doc with exact query term is returned even when dense retriever returns nothing.

        This is the core value proposition of BM25: it catches exact medical
        terminology that may not be the nearest semantic neighbour in the embedding
        space.
        """
        docs = [
            _make_doc("consolidation is a sign of pneumonia in radiology", "doc_consolidation"),
            _make_doc("normal heart anatomy and function description", "doc_heart"),
            _make_doc("liver function and metabolic processes", "doc_liver"),
        ]
        # Dense returns nothing — only BM25 can contribute results
        mock_dense = MockDenseRetriever(docs=docs, results=[])
        bm25_dir = _write_bm25_index(docs, tmp_path / "bm25")
        retriever = HybridRetriever(
            dense_retriever=mock_dense,
            bm25_index_dir=bm25_dir,
            min_score=0.0,
        )

        results = retriever.retrieve("consolidation", top_k=5)
        result_ids = [r.document.doc_id for r in results]

        assert "doc_consolidation" in result_ids

    def test_bm25_zero_score_docs_excluded(self, tmp_path):
        """Documents with BM25 score=0 (no query-term overlap) are not in BM25 hits.

        When dense also returns nothing, only docs with positive BM25 scores appear.
        """
        docs = [
            _make_doc("consolidation in the lung radiological sign", "doc_match"),
            _make_doc("liver function metabolic process", "doc_nomatch"),
            _make_doc("heart anatomy circulatory system", "doc_nomatch2"),
        ]
        mock_dense = MockDenseRetriever(docs=docs, results=[])
        bm25_dir = _write_bm25_index(docs, tmp_path / "bm25")
        retriever = HybridRetriever(
            dense_retriever=mock_dense,
            bm25_index_dir=bm25_dir,
            min_score=0.0,
        )

        results = retriever.retrieve("consolidation", top_k=3)
        result_ids = [r.document.doc_id for r in results]

        assert "doc_match" in result_ids
        assert "doc_nomatch" not in result_ids
        assert "doc_nomatch2" not in result_ids


# ── Tests 10–11: GraphBuilder factory dispatch ─────────────────────────────────


class TestGraphBuilderFactory:
    """Verify GraphBuilder.build() dispatches on the retrieval_method config key.

    Both tests use SimpleNamespace as a lightweight config object (no pydantic
    overhead) and monkeypatch Retriever / HybridRetriever to record which class
    is instantiated without touching the filesystem or GPU.
    """

    def test_factory_dense_config(self, monkeypatch):
        """With retrieval_method='dense', Retriever is instantiated (not HybridRetriever)."""
        import radiology_vqa.rag.retriever as retriever_module
        from radiology_vqa.graph.builder import GraphBuilder

        instantiated = []

        class FakeDenseRetriever:
            def __init__(self, index_dir, **kw):
                instantiated.append("Retriever")

            def retrieve(self, q, **kw):
                return []

        class FakeHybridRetriever:
            def __init__(self, dense_retriever, bm25_index_dir, **kw):
                instantiated.append("HybridRetriever")

            def retrieve(self, q, **kw):
                return []

        monkeypatch.setattr(retriever_module, "Retriever", FakeDenseRetriever)
        monkeypatch.setattr(retriever_module, "HybridRetriever", FakeHybridRetriever)

        config = SimpleNamespace(
            retrieval_method="dense",
            index_dir=Path("data/indices"),
            retrieval_top_k=5,
            vlm_backend="llava",
        )
        builder = GraphBuilder(config)
        from unittest.mock import MagicMock

        mock_vlm = MagicMock()
        mock_vlm.model_name = "mock"
        try:
            builder.build(vlm=mock_vlm, retriever=None)
        except Exception:
            # LangGraph compilation may fail with fake components; that's fine —
            # we only care that the factory dispatch ran.
            pass

        assert "Retriever" in instantiated
        assert "HybridRetriever" not in instantiated

    def test_factory_hybrid_config(self, monkeypatch):
        """With retrieval_method='hybrid', HybridRetriever is instantiated."""
        import radiology_vqa.rag.retriever as retriever_module
        from radiology_vqa.graph.builder import GraphBuilder

        instantiated = []

        class FakeDenseRetriever:
            def __init__(self, index_dir, **kw):
                instantiated.append("Retriever")
                self._documents = []

            def retrieve(self, q, **kw):
                return []

        class FakeHybridRetriever:
            def __init__(self, dense_retriever, bm25_index_dir, **kw):
                instantiated.append("HybridRetriever")

            def retrieve(self, q, **kw):
                return []

        monkeypatch.setattr(retriever_module, "Retriever", FakeDenseRetriever)
        monkeypatch.setattr(retriever_module, "HybridRetriever", FakeHybridRetriever)

        config = SimpleNamespace(
            retrieval_method="hybrid",
            index_dir=Path("data/indices"),
            retrieval_top_k=5,
            bm25_index_dir="data/bm25_index",
            bm25_top_k=20,
            dense_top_k=20,
            rrf_k=60,
            retrieval_min_score=0.0,
            vlm_backend="llava",
        )
        builder = GraphBuilder(config)
        from unittest.mock import MagicMock

        mock_vlm = MagicMock()
        mock_vlm.model_name = "mock"
        try:
            builder.build(vlm=mock_vlm, retriever=None)
        except Exception:
            pass

        assert "HybridRetriever" in instantiated
