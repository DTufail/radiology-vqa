"""Interactive retrieval quality testing.

Usage:
    python scripts/test_retrieval.py                                  # run validation queries
    python scripts/test_retrieval.py --query "symptoms of pneumonia"  # single query
    python scripts/test_retrieval.py --interactive                    # interactive loop
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import settings
from radiology_vqa.rag.retriever import Retriever

logging.basicConfig(level="WARNING", format="%(levelname)s %(name)s: %(message)s")

VALIDATION_QUERIES = [
    "What are the symptoms of pneumonia?",
    "What is the function of the liver?",
    "What causes lung cancer?",
    "Where is the heart located?",
    "What is the treatment for asthma?",
    "Which organs belong to the digestive system?",
    "What does consolidation in the lung indicate?",
    "What are the symptoms of a brain tumor?",
    "What is the function of the pancreas?",
    "How is tuberculosis transmitted?",
]

_TRUNC = 120


def _trunc(text: str, n: int = _TRUNC) -> str:
    return text[:n] + "..." if len(text) > n else text


def run_query(retriever: Retriever, query: str, top_k: int = 5) -> None:
    print(f"\nQuery: {query}")
    print("-" * 80)
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        print("  (no results)")
        return
    for r in results:
        m = r.document.meta
        print(
            f"  [{r.rank}] score={r.score:.4f}  [{m.source_type}] {m.entity_name} / {m.attribute}"
        )
        print(f"       {_trunc(r.document.text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test retrieval quality.")
    parser.add_argument("--query", type=str, default=None, help="Single query to test")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.index_dir,
        help="Path to the index directory",
    )
    args = parser.parse_args()

    if not args.index_dir.exists():
        print(f"ERROR: Index not found at {args.index_dir}. Run `make build-index` first.")
        sys.exit(1)

    print(f"Loading index from {args.index_dir} ...")
    retriever = Retriever(args.index_dir)
    print("Index loaded.\n")

    if args.query:
        run_query(retriever, args.query)
    elif args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                run_query(retriever, query)
    else:
        print(f"Running {len(VALIDATION_QUERIES)} validation queries ...\n")
        for query in VALIDATION_QUERIES:
            run_query(retriever, query)


if __name__ == "__main__":
    main()
