"""Build the FAISS knowledge base index from SLAKE KG data.

Usage:
    python scripts/build_index.py                          # FAISS index only (default)
    python scripts/build_index.py --bm25                   # FAISS + BM25 index
    python scripts/build_index.py --bm25 --bm25-dir data/bm25_index
"""

import argparse
import logging
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import settings
from radiology_vqa.kg_loader import load_kg
from radiology_vqa.rag.embedder import Embedder
from radiology_vqa.rag.indexer import FAISSIndexer
from radiology_vqa.rag.kg_processor import KGProcessor

logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_from_kg() -> list:
    print(f"Loading KG triples from {settings.slake_dir} ...")
    triples = load_kg(settings.slake_dir)
    if not triples:
        print("ERROR: No KG triples loaded. Check that SLAKE data is at the configured path.")
        sys.exit(1)
    print(f"  Loaded {len(triples)} triples.")

    print("Processing triples into documents ...")
    processor = KGProcessor()
    documents = processor.process_all(triples)
    print(f"  Generated {len(documents)} documents.")
    return documents


def build_bm25_index(documents: list, output_dir: str) -> None:
    """Build a BM25 index from documents and persist to disk.

    Creates two parallel pickle files in output_dir:
      - corpus.pkl:   List[List[str]] — tokenised document texts
      - doc_ids.pkl:  List[str]       — document IDs, index-aligned with corpus

    corpus[i] and doc_ids[i] always refer to the same document.

    Tokenisation: lowercase + whitespace split.  No NLTK/spaCy dependency.
    This matches the tokenisation applied to queries at retrieval time in
    HybridRetriever.retrieve(), ensuring term-frequency counts are consistent.

    Args:
        documents: List of Document objects (same list used to build the FAISS index).
        output_dir: Directory where corpus.pkl and doc_ids.pkl will be written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus = [doc.text.lower().split() for doc in documents]
    doc_ids = [doc.doc_id for doc in documents]

    vocab_size = len({token for doc_tokens in corpus for token in doc_tokens})
    logger.info("Building BM25 index from %d documents", len(documents))
    logger.info("Vocabulary size: %d unique tokens", vocab_size)

    with open(output_path / "corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    with open(output_path / "doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)

    logger.info("BM25 index saved to %s", output_dir)
    print(f"  BM25 index: {len(documents)} docs, {vocab_size} vocab tokens → {output_dir}")


def print_summary(documents: list, index_dir: Path, build_time: float) -> None:
    counts = Counter(doc.meta.source_type for doc in documents)
    print("\n--- Index Summary ---")
    for source_type, count in sorted(counts.items()):
        print(f"  {source_type:<20} {count:>5} docs")
    print(f"  {'TOTAL':<20} {len(documents):>5} docs")

    # Index size on disk
    faiss_path = index_dir / "index.faiss"
    if faiss_path.exists():
        size_mb = faiss_path.stat().st_size / (1024 * 1024)
        print(f"  index.faiss size:    {size_mb:.2f} MB")

    print(f"  Build time:          {build_time:.1f}s")
    print(f"  Index saved to:      {index_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base index.")
    parser.add_argument(
        "--source",
        choices=["kg", "all"],
        default="all",
        help="Legacy single-source flag (default: all). Use --sources for multi-source builds.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=["kg", "radlex", "qa"],
        help=(
            "Document sources to include in the index. "
            "Choices: kg (SLAKE KG), radlex (RadLex ontology), qa (VQA-RAD + SLAKE QA pairs). "
            "Default when omitted: ['kg'] (backward-compatible Phase 5 behaviour)."
        ),
    )
    parser.add_argument(
        "--bm25",
        action="store_true",
        help="Also build BM25 index alongside the FAISS index (Phase 6B hybrid retrieval)",
    )
    parser.add_argument(
        "--bm25-dir",
        default="data/bm25_index",
        help="Output directory for BM25 index files (default: data/bm25_index)",
    )
    parser.add_argument(
        "--radlex-xls",
        default="data/raw/radlex/Radlex.xls",
        help="Path to Radlex.xls ontology file (default: data/raw/radlex/Radlex.xls)",
    )
    parser.add_argument(
        "--slake-train",
        default="data/raw/Slake1.0/train.json",
        help="Path to SLAKE train.json for QA pseudo-docs (default: data/raw/Slake1.0/train.json)",
    )
    parser.add_argument(
        "--output-index-dir",
        default=None,
        help=(
            "Output directory for FAISS index. "
            "Defaults to the index_dir from settings (data/indices). "
            "Use data/indices_v2 for the expanded Phase 6B-2 build."
        ),
    )
    args = parser.parse_args()

    # Resolve which sources to build
    # --sources takes priority over legacy --source flag
    if args.sources is not None:
        active_sources = args.sources
    else:
        # Backward-compatible default: KG only (same as Phase 5 behaviour)
        active_sources = ["kg"]

    # Resolve output directory
    output_index_dir = Path(args.output_index_dir) if args.output_index_dir else settings.index_dir

    t_start = time.perf_counter()
    documents: list = []

    # ── KG source ──────────────────────────────────────────────────────────────
    if "kg" in active_sources:
        kg_docs = build_from_kg()
        documents.extend(kg_docs)

    # ── RadLex source ──────────────────────────────────────────────────────────
    if "radlex" in active_sources:
        from radiology_vqa.rag.radlex_processor import RadLexProcessor

        print(f"\nLoading RadLex ontology from {args.radlex_xls} ...")
        radlex_docs = RadLexProcessor(args.radlex_xls).process()
        print(f"  RadLex Tier 1 documents: {len(radlex_docs)}")
        documents.extend(radlex_docs)

    # ── QA pseudo-documents source ─────────────────────────────────────────────
    if "qa" in active_sources:
        from radiology_vqa.rag.qa_pseudo_processor import QAPseudoProcessor

        proc = QAPseudoProcessor(args.slake_train)

        print("\nLoading VQA-RAD QA pseudo-documents ...")
        vqarad_docs = proc.process_vqarad()
        print(f"  VQA-RAD pseudo-documents: {len(vqarad_docs)}")
        documents.extend(vqarad_docs)

        print(f"\nLoading SLAKE QA pseudo-documents from {args.slake_train} ...")
        slake_qa_docs = proc.process_slake()
        print(f"  SLAKE QA pseudo-documents (English only): {len(slake_qa_docs)}")
        documents.extend(slake_qa_docs)

    print(f"\nInitializing embedding model ({settings.embedding_model}) ...")
    embedder = Embedder()

    print("Building FAISS index ...")
    indexer = FAISSIndexer(embedder)
    indexer.build_index(documents)

    print(f"Saving index to {output_index_dir} ...")
    indexer.save(output_index_dir)

    if args.bm25:
        print(f"\nBuilding BM25 index in {args.bm25_dir} ...")
        build_bm25_index(documents, args.bm25_dir)

    build_time = time.perf_counter() - t_start
    print_summary(documents, output_index_dir, build_time)


if __name__ == "__main__":
    main()
