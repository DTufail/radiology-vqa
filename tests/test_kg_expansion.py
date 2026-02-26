"""Tests for Phase 6B-2 knowledge base expansion.

Coverage:
  - RadLexProcessor (7 tests)  — Tier 1 filter, doc_id format, source_type,
                                  consolidation coverage, encoding, minimum count
  - QAPseudoProcessor (7 tests) — VQA-RAD content format, unique IDs,
                                   SLAKE English-only filter, metadata enrichment,
                                   global ID uniqueness
  - Build script integration (3 tests, @pytest.mark.slow) — --sources flag

All RadLex tests require data/Radlex.xls (local file, ~1.7s read time).
All SLAKE tests require data/raw/Slake1.0/train.json (local file).
VQA-RAD tests use mock data — no HuggingFace download needed.
"""

import sys
from pathlib import Path

import pytest

# Allow importing build scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from radiology_vqa.rag.document import Document

# ── Paths ──────────────────────────────────────────────────────────────────────

RADLEX_XLS = Path("data/Radlex.xls")
SLAKE_TRAIN = Path("data/raw/Slake1.0/train.json")

radlex_available = pytest.mark.skipif(
    not RADLEX_XLS.exists(), reason="data/Radlex.xls not present"
)
slake_available = pytest.mark.skipif(
    not SLAKE_TRAIN.exists(), reason="data/raw/Slake1.0/train.json not present"
)


# ── RadLex Processor Tests ─────────────────────────────────────────────────────


class TestRadLexProcessor:
    @pytest.fixture(scope="class")
    def radlex_docs(self):
        """Load RadLex documents once for all tests in this class."""
        if not RADLEX_XLS.exists():
            pytest.skip("data/Radlex.xls not present")
        from radiology_vqa.rag.radlex_processor import RadLexProcessor

        return RadLexProcessor(str(RADLEX_XLS)).process()

    def test_radlex_processor_returns_documents(self, radlex_docs):
        """RadLexProcessor.process() returns a non-empty list of Document objects."""
        assert len(radlex_docs) > 0
        assert all(isinstance(d, Document) for d in radlex_docs)

    def test_radlex_tier1_filter(self, radlex_docs):
        """All returned documents have non-empty text (definition present)."""
        for doc in radlex_docs:
            assert doc.text.strip(), f"Empty text in doc {doc.doc_id}"

    def test_radlex_doc_id_format(self, radlex_docs):
        """All doc_ids start with 'radlex_' and are globally unique."""
        for doc in radlex_docs:
            assert doc.doc_id.startswith("radlex_"), (
                f"doc_id {doc.doc_id!r} does not start with 'radlex_'"
            )
        doc_ids = [d.doc_id for d in radlex_docs]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids in RadLex output"

    def test_radlex_source_type(self, radlex_docs):
        """All documents have source_type == 'radlex'."""
        for doc in radlex_docs:
            assert doc.meta.source_type == "radlex", (
                f"Unexpected source_type {doc.meta.source_type!r} in {doc.doc_id}"
            )

    def test_radlex_consolidation_present(self, radlex_docs):
        """At least one document mentions 'consolidation' — key coverage gap filled."""
        texts = [d.text.lower() for d in radlex_docs]
        assert any("consolidation" in t for t in texts), (
            "No RadLex document contains 'consolidation'. Coverage gap not filled."
        )

    def test_radlex_encoding_clean(self, radlex_docs):
        """No document content contains the Unicode replacement character (U+FFFD)."""
        for doc in radlex_docs:
            assert "\ufffd" not in doc.text, (
                f"Replacement character found in {doc.doc_id}: {doc.text[:80]!r}"
            )

    def test_radlex_minimum_count(self, radlex_docs):
        """At least 3,000 documents returned — Tier 1 filter is not over-pruning."""
        assert len(radlex_docs) >= 3000, (
            f"Only {len(radlex_docs)} RadLex docs returned; expected >= 3000"
        )


# ── QA Pseudo-Document Processor Tests ────────────────────────────────────────


class TestQAPseudoProcessor:
    """All VQA-RAD tests use mock data — no HuggingFace download required."""

    MOCK_VQARAD = [
        {"question": "Is there consolidation in the left lung?", "answer": "yes"},
        {"question": "What modality is this image?", "answer": "CT"},
        {"question": "Is the cardiac silhouette enlarged?", "answer": "no"},
    ]

    @pytest.fixture
    def proc(self):
        if not SLAKE_TRAIN.exists():
            pytest.skip("data/raw/Slake1.0/train.json not present")
        from radiology_vqa.rag.qa_pseudo_processor import QAPseudoProcessor

        return QAPseudoProcessor(str(SLAKE_TRAIN))

    def test_vqarad_pseudo_returns_documents(self, proc):
        """process_vqarad() returns a non-empty list with source_type='qa_vqarad'."""
        docs = proc.process_vqarad(dataset=self.MOCK_VQARAD)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.meta.source_type == "qa_vqarad"

    def test_vqarad_content_format(self, proc):
        """Each VQA-RAD document starts with 'Question:' and contains 'Answer:'."""
        docs = proc.process_vqarad(dataset=self.MOCK_VQARAD)
        for doc in docs:
            assert doc.text.startswith("Question:"), (
                f"Content does not start with 'Question:': {doc.text[:60]!r}"
            )
            assert "Answer:" in doc.text, (
                f"Content does not contain 'Answer:': {doc.text[:60]!r}"
            )

    def test_vqarad_doc_id_unique(self, proc):
        """All doc_ids from process_vqarad() are unique."""
        docs = proc.process_vqarad(dataset=self.MOCK_VQARAD)
        doc_ids = [d.doc_id for d in docs]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids in VQA-RAD output"

    def test_slake_pseudo_english_only(self, proc):
        """process_slake() returns only English entries — no Chinese characters."""
        docs = proc.process_slake()
        assert len(docs) > 0
        for doc in docs:
            # Chinese Unicode range U+4E00–U+9FFF
            assert not any("\u4e00" <= ch <= "\u9fff" for ch in doc.text), (
                f"Chinese character found in SLAKE doc {doc.doc_id}: {doc.text[:80]!r}"
            )

    def test_slake_pseudo_returns_documents(self, proc):
        """process_slake() returns a non-empty list with source_type='qa_slake'."""
        docs = proc.process_slake()
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.meta.source_type == "qa_slake"

    def test_slake_metadata_appended(self, proc):
        """At least some SLAKE documents have 'Body region:' or 'Modality:' appended."""
        docs = proc.process_slake()
        enriched = [
            d for d in docs
            if "Body region:" in d.text or "Modality:" in d.text
        ]
        assert len(enriched) > 0, (
            "No SLAKE documents contain metadata enrichment (Body region / Modality)"
        )

    def test_qa_doc_ids_globally_unique(self, proc):
        """No doc_id collision between process_vqarad() and process_slake() outputs."""
        vqarad_docs = proc.process_vqarad(dataset=self.MOCK_VQARAD)
        slake_docs = proc.process_slake()
        all_ids = [d.doc_id for d in vqarad_docs] + [d.doc_id for d in slake_docs]
        assert len(all_ids) == len(set(all_ids)), (
            "doc_id collision between VQA-RAD and SLAKE pseudo-document outputs"
        )


# ── Build Script Integration Tests (slow) ─────────────────────────────────────


@pytest.mark.slow
class TestBuildIndexSources:
    """Verify --sources flag produces correct document counts.

    These tests call build_from_kg() / RadLexProcessor / QAPseudoProcessor
    directly (without the subprocess overhead of calling the CLI).
    Marked @pytest.mark.slow — excluded from 'make test -m not slow'.
    """

    def test_build_index_sources_kg_only(self):
        """KG-only build produces the same 2,987 documents as before (no regression)."""
        from build_index import build_from_kg

        docs = build_from_kg()
        assert len(docs) == 2987, (
            f"KG-only build changed: expected 2987 docs, got {len(docs)}"
        )

    @pytest.mark.skipif(not RADLEX_XLS.exists(), reason="data/Radlex.xls not present")
    def test_build_index_sources_radlex(self):
        """KG + RadLex build produces > 6,000 documents."""
        from build_index import build_from_kg
        from radiology_vqa.rag.radlex_processor import RadLexProcessor

        docs = build_from_kg()
        docs += RadLexProcessor(str(RADLEX_XLS)).process()
        assert len(docs) > 6000, (
            f"KG + RadLex build too small: {len(docs)} docs"
        )

    @pytest.mark.skipif(
        not RADLEX_XLS.exists() or not SLAKE_TRAIN.exists(),
        reason="data/Radlex.xls or SLAKE train.json not present",
    )
    def test_build_index_sources_all(self):
        """Full build (KG + RadLex + QA) produces > 13,000 documents."""
        from build_index import build_from_kg
        from radiology_vqa.rag.qa_pseudo_processor import QAPseudoProcessor
        from radiology_vqa.rag.radlex_processor import RadLexProcessor

        docs = build_from_kg()
        docs += RadLexProcessor(str(RADLEX_XLS)).process()

        proc = QAPseudoProcessor(str(SLAKE_TRAIN))
        docs += proc.process_vqarad()   # loads from HuggingFace
        docs += proc.process_slake()

        assert len(docs) > 13000, (
            f"Full expanded build too small: {len(docs)} docs"
        )
