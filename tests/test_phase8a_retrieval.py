"""Tests for Phase 8A — citation_relevance metric and index versioning."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(question, citations=None, decision="answer"):
    """Build a minimal PerSampleResult-like object."""
    r = MagicMock()
    r.question = question
    r.decision = decision
    r.citations = citations or []
    return r


def _make_citation(text):
    """Build a citation object with a .text attribute."""
    c = MagicMock()
    c.text = text
    return c


# ── TestCitationRelevance ─────────────────────────────────────────────────────

class TestCitationRelevance:
    """Tests for the updated token-overlap citation_relevance function."""

    def _call(self, per_sample, min_token_overlap=1):
        from radiology_vqa.evaluation.agent_metrics import citation_relevance
        return citation_relevance(per_sample, min_token_overlap=min_token_overlap)

    def test_empty_list_returns_zero(self):
        """No samples → 0.0."""
        assert self._call([]) == 0.0

    def test_all_abstained_returns_zero(self):
        """All abstained → 0.0 (no answered samples)."""
        results = [_make_result("is there consolidation?", decision="abstain")]
        assert self._call(results) == 0.0

    def test_relevant_citation_as_string(self):
        """Citation is a plain string containing question term → hit."""
        results = [_make_result(
            "is there consolidation in the lung?",
            citations=["consolidation is a radiological finding in pneumonia"],
        )]
        assert self._call(results) == pytest.approx(1.0)

    def test_relevant_citation_as_object(self):
        """Citation is an object with .text attribute → hit."""
        cite = _make_citation("consolidation visible in left lower lobe")
        results = [_make_result(
            "is there consolidation?",
            citations=[cite],
        )]
        assert self._call(results) == pytest.approx(1.0)

    def test_irrelevant_citation_returns_zero(self):
        """Citation shares no tokens with question (after stop-word removal) → miss."""
        results = [_make_result(
            "is there airspace consolidation?",
            citations=["the liver is located in the right upper quadrant"],
        )]
        # "airspace" and "consolidation" not in citation
        assert self._call(results) == pytest.approx(0.0)

    def test_stop_words_not_counted(self):
        """Stop words (is, the, are) must not count as overlapping tokens."""
        results = [_make_result(
            "is there a fracture?",
            citations=["the patient is in stable condition"],
        )]
        # Only shared non-stop tokens: none (fracture not in citation, patient not in question)
        assert self._call(results) == pytest.approx(0.0)

    def test_partial_hit_rate(self):
        """2 answered, 1 relevant → 0.5."""
        results = [
            _make_result("is there consolidation?",
                         citations=["consolidation is present"]),
            _make_result("what is the modality?",
                         citations=["liver function test results"]),
        ]
        assert self._call(results) == pytest.approx(0.5)

    def test_abstained_excluded_from_denominator(self):
        """Abstained samples do not count in denominator."""
        results = [
            _make_result("is there consolidation?",
                         citations=["consolidation present"], decision="answer"),
            _make_result("what organ?",
                         citations=[], decision="abstain"),
        ]
        # 1 answered, 1 relevant → 1.0
        assert self._call(results) == pytest.approx(1.0)

    def test_only_one_relevant_citation_needed(self):
        """A sample counts as hit if ANY citation is relevant (not all)."""
        results = [_make_result(
            "is there consolidation?",
            citations=[
                "the liver is in the right upper quadrant",  # irrelevant
                "consolidation seen in right lower lobe",     # relevant
            ],
        )]
        assert self._call(results) == pytest.approx(1.0)

    def test_qa_vqarad_source_type_counts(self):
        """QA pseudo-doc citation format ('Question: X Answer: Y') is matched."""
        results = [_make_result(
            "is there cardiomegaly?",
            citations=["Question: is there cardiomegaly? Answer: yes"],
        )]
        assert self._call(results) == pytest.approx(1.0)

    def test_min_token_overlap_two(self):
        """With min_token_overlap=2, single shared token is not enough."""
        results = [_make_result(
            "is there consolidation in the lung?",
            citations=["consolidation is a finding"],   # only 1 shared: "consolidation"
        )]
        assert self._call(results, min_token_overlap=2) == pytest.approx(0.0)

    def test_min_token_overlap_two_passes_with_two_tokens(self):
        """With min_token_overlap=2, two shared tokens triggers a hit."""
        results = [_make_result(
            "is there consolidation in the lung?",
            citations=["consolidation visible in the lung"],  # 2 shared: consolidation, lung
        )]
        assert self._call(results, min_token_overlap=2) == pytest.approx(1.0)


# ── TestIndexVersioning ───────────────────────────────────────────────────────

class TestIndexVersioning:
    """Tests for the index_version and sources fields in index_meta.json."""

    def test_save_writes_sources_field(self, tmp_path):
        """FAISSIndexer.save() writes sources field derived from documents."""
        from radiology_vqa.rag.document import Document, DocumentMeta
        from radiology_vqa.rag.indexer import FAISSIndexer

        # Build a minimal index with two source types
        docs = [
            Document(
                text="pneumonia is an infection",
                meta=DocumentMeta(source_type="kg_disease", entity_name="pneumonia",
                                  attribute="summary", source_file="test.csv"),
                doc_id="kg_disease_pneumonia_summary_0",
            ),
            Document(
                text="Question: is there consolidation? Answer: yes",
                meta=DocumentMeta(source_type="qa_vqarad", entity_name="vqarad_0",
                                  attribute="qa_pair", source_file="vqa_rad"),
                doc_id="qa_vqarad_0",
            ),
        ]

        mock_embedder = MagicMock()
        mock_embedder.dimension = 8
        mock_embedder.model_name = "test-model"
        mock_embedder.embed_texts.return_value = __import__("numpy").random.rand(2, 8).astype("float32")

        indexer = FAISSIndexer(mock_embedder)
        indexer.build_index(docs)
        indexer.save(tmp_path, index_version="3.0.0")

        with open(tmp_path / "index_meta.json") as f:
            meta = json.load(f)

        assert "sources" in meta
        assert "qa_vqarad" in meta["sources"]
        assert "kg_disease" in meta["sources"]
        assert meta["sources"] == sorted(meta["sources"])  # must be sorted

    def test_save_writes_index_version(self, tmp_path):
        """FAISSIndexer.save() writes index_version field."""
        from radiology_vqa.rag.document import Document, DocumentMeta
        from radiology_vqa.rag.indexer import FAISSIndexer

        doc = Document(
            text="test document",
            meta=DocumentMeta(source_type="kg_disease", entity_name="test",
                              attribute="summary", source_file="test.csv"),
            doc_id="kg_disease_test_0",
        )
        mock_embedder = MagicMock()
        mock_embedder.dimension = 8
        mock_embedder.model_name = "test-model"
        mock_embedder.embed_texts.return_value = __import__("numpy").random.rand(1, 8).astype("float32")

        indexer = FAISSIndexer(mock_embedder)
        indexer.build_index([doc])
        indexer.save(tmp_path, index_version="3.0.0")

        with open(tmp_path / "index_meta.json") as f:
            meta = json.load(f)

        assert meta["index_version"] == "3.0.0"

    def test_save_default_version_is_1_0_0(self, tmp_path):
        """Default index_version is '1.0.0' for backward compatibility."""
        from radiology_vqa.rag.document import Document, DocumentMeta
        from radiology_vqa.rag.indexer import FAISSIndexer

        doc = Document(
            text="test",
            meta=DocumentMeta(source_type="kg_disease", entity_name="x",
                              attribute="s", source_file="f.csv"),
            doc_id="kg_disease_x_0",
        )
        mock_embedder = MagicMock()
        mock_embedder.dimension = 8
        mock_embedder.model_name = "test-model"
        mock_embedder.embed_texts.return_value = __import__("numpy").random.rand(1, 8).astype("float32")

        indexer = FAISSIndexer(mock_embedder)
        indexer.build_index([doc])
        indexer.save(tmp_path)   # no index_version arg

        with open(tmp_path / "index_meta.json") as f:
            meta = json.load(f)

        assert meta["index_version"] == "1.0.0"

    def test_load_reads_version_and_sources(self, tmp_path):
        """FAISSIndexer.load() returns meta dict containing version and sources."""
        from radiology_vqa.rag.document import Document, DocumentMeta
        from radiology_vqa.rag.indexer import FAISSIndexer

        doc = Document(
            text="test",
            meta=DocumentMeta(source_type="qa_slake", entity_name="s0",
                              attribute="qa_pair", source_file="train.json"),
            doc_id="qa_slake_0",
        )
        mock_embedder = MagicMock()
        mock_embedder.dimension = 8
        mock_embedder.model_name = "test-model"
        mock_embedder.embed_texts.return_value = __import__("numpy").random.rand(1, 8).astype("float32")

        indexer = FAISSIndexer(mock_embedder)
        indexer.build_index([doc])
        indexer.save(tmp_path, index_version="3.0.0")

        _, _, loaded_meta = FAISSIndexer.load(tmp_path)
        assert loaded_meta["index_version"] == "3.0.0"
        assert "qa_slake" in loaded_meta["sources"]

    def test_load_backward_compatible_no_version_field(self, tmp_path):
        """Loading an old index_meta.json without version/sources fields does not crash."""
        from radiology_vqa.rag.document import Document, DocumentMeta
        from radiology_vqa.rag.indexer import FAISSIndexer

        doc = Document(
            text="test",
            meta=DocumentMeta(source_type="kg_disease", entity_name="x",
                              attribute="s", source_file="f.csv"),
            doc_id="kg_disease_x_0",
        )
        mock_embedder = MagicMock()
        mock_embedder.dimension = 8
        mock_embedder.model_name = "test-model"
        mock_embedder.embed_texts.return_value = __import__("numpy").random.rand(1, 8).astype("float32")

        indexer = FAISSIndexer(mock_embedder)
        indexer.build_index([doc])
        indexer.save(tmp_path)

        # Manually strip the new fields to simulate an old index
        meta_path = tmp_path / "index_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta.pop("index_version", None)
        meta.pop("sources", None)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Must not raise
        _, _, loaded_meta = FAISSIndexer.load(tmp_path)
        assert loaded_meta.get("index_version", "unknown") == "unknown"
