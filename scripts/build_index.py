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
        help="Which knowledge sources to index (default: all)",
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
    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.source in ("kg", "all"):
        documents = build_from_kg()

    if args.source == "all":
        print("Note: PubMed and SLAKE-QA sources are deferred. Indexing KG only.")

    print(f"\nInitializing embedding model ({settings.embedding_model}) ...")
    embedder = Embedder()

    print("Building FAISS index ...")
    indexer = FAISSIndexer(embedder)
    indexer.build_index(documents)

    print(f"Saving index to {settings.index_dir} ...")
    indexer.save(settings.index_dir)

    if args.bm25:
        print(f"\nBuilding BM25 index in {args.bm25_dir} ...")
        build_bm25_index(documents, args.bm25_dir)

    build_time = time.perf_counter() - t_start
    print_summary(documents, settings.index_dir, build_time)


if __name__ == "__main__":
    main()
