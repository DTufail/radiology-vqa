#!/usr/bin/env python3
"""Phase 8A — Retrieval diagnostic: verify citation hit rate improvement.

Loads the hybrid index and runs retrieval on a sample of VQA-RAD test
questions WITHOUT loading the VLM. Reports:
  1. Citation hit rate (token-overlap definition, consistent with Phase 8A metric)
  2. Per-source-type hit counts (shows whether qa_vqarad/qa_slake are contributing)
  3. Top-5 retrieved docs for up to 10 sample questions (qualitative check)
  4. BM25 vs dense contribution breakdown

This is a diagnostic tool only — it does not run the full agent pipeline.
Run before and after rebuilding the index to confirm improvement.

Usage
-----
    # Diagnose current index (v2):
    python scripts/diagnose_retrieval.py \
        --index-dir data/indices_v2 \
        --bm25-dir data/bm25_index \
        --n-samples 50

    # Diagnose new index (v3) after Phase 8A build:
    python scripts/diagnose_retrieval.py \
        --index-dir data/indices_v3 \
        --bm25-dir data/bm25_index_v3 \
        --n-samples 50

    # Compare both indexes side by side:
    python scripts/diagnose_retrieval.py \
        --index-dir data/indices_v3 \
        --bm25-dir data/bm25_index_v3 \
        --compare-index-dir data/indices_v2 \
        --compare-bm25-dir data/bm25_index \
        --n-samples 50
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("diagnose_retrieval")

# Add src/ to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Stop words (same as citation_relevance in agent_metrics.py) ───────────────
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "this", "that",
    "there", "in", "on", "at", "of", "to", "for", "and", "or",
    "not", "what", "which", "where", "how", "any", "with",
})
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def _tokens(text: str) -> frozenset[str]:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return frozenset(t for t in cleaned.split() if t not in _STOP and len(t) > 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 8A retrieval diagnostic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--index-dir", default="data/indices_v3",
                   help="FAISS index directory (default: data/indices_v3)")
    p.add_argument("--bm25-dir", default="data/bm25_index_v3",
                   help="BM25 index directory (default: data/bm25_index_v3)")
    p.add_argument("--compare-index-dir", default=None,
                   help="Optional second index to compare against")
    p.add_argument("--compare-bm25-dir", default=None,
                   help="Optional second BM25 index to compare against")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of VQA-RAD test questions to diagnose (default: 50)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of documents to retrieve per question (default: 5)")
    p.add_argument("--show-examples", type=int, default=5,
                   help="Number of questions to print full retrieval results for (default: 5)")
    p.add_argument("--output", default=None,
                   help="Optional JSON output file for full results")
    return p.parse_args()


def _load_retriever(index_dir: str, bm25_dir: str):
    """Load HybridRetriever from disk. Returns retriever and index metadata."""
    from radiology_vqa.rag.retriever import HybridRetriever, Retriever

    logger.info("Loading dense index from %s ...", index_dir)
    dense = Retriever(Path(index_dir))

    # Load index metadata for version/sources info
    meta_path = Path(index_dir) / "index_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    logger.info("Loading BM25 index from %s ...", bm25_dir)
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_index_dir=bm25_dir,
        bm25_top_k=20,
        dense_top_k=20,
        rrf_k=60,
    )
    return hybrid, meta


def _load_vqarad_test_questions(n_samples: int) -> list[dict]:
    """Load VQA-RAD test questions from HuggingFace. Returns list of dicts."""
    logger.info("Loading VQA-RAD test split (%d samples) ...", n_samples)
    from datasets import load_dataset
    dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
    samples = []
    for i, row in enumerate(dataset):
        if i >= n_samples:
            break
        samples.append({
            "question": str(row.get("question", "")).strip(),
            "answer": str(row.get("answer", "")).strip(),
            "answer_type": str(row.get("answer_type", "open")).strip().lower(),
        })
    logger.info("Loaded %d VQA-RAD test samples", len(samples))
    return samples


def _diagnose(
    retriever,
    questions: list[dict],
    top_k: int,
    label: str,
) -> dict:
    """Run retrieval on all questions and collect statistics.

    Returns a results dict with:
      - citation_hit_rate: float
      - source_type_counts: dict[str, int]  (how many hits came from each source)
      - per_question: list of per-question result dicts
    """
    logger.info("\n=== Diagnosing retrieval: %s ===", label)

    hit_count = 0
    source_type_hits: Counter = Counter()
    per_question_results = []

    for i, sample in enumerate(questions):
        question = sample["question"]
        q_tokens = _tokens(question)

        results = retriever.retrieve(question, top_k=top_k)

        # Check citation relevance (token overlap with question)
        hit = False
        sources_hit = set()
        retrieved_texts = []
        for r in results:
            cite_tokens = _tokens(r.document.text)
            overlap = len(q_tokens & cite_tokens)
            is_relevant = overlap >= 1
            if is_relevant:
                hit = True
                sources_hit.add(r.document.meta.source_type)
            retrieved_texts.append({
                "rank": r.rank,
                "score": round(r.score, 4),
                "source_type": r.document.meta.source_type,
                "text_preview": r.document.text[:100],
                "token_overlap": overlap,
                "relevant": is_relevant,
            })

        if hit:
            hit_count += 1
            for st in sources_hit:
                source_type_hits[st] += 1

        per_question_results.append({
            "question": question,
            "answer": sample["answer"],
            "answer_type": sample["answer_type"],
            "hit": hit,
            "retrieved": retrieved_texts,
        })

    citation_hit_rate = hit_count / len(questions) if questions else 0.0

    logger.info("Citation hit rate [%s]: %.1f%% (%d/%d)",
                label, citation_hit_rate * 100, hit_count, len(questions))
    logger.info("Hit counts by source_type:")
    for st, count in source_type_hits.most_common():
        logger.info("  %-20s %d", st, count)

    return {
        "label": label,
        "citation_hit_rate": citation_hit_rate,
        "n_questions": len(questions),
        "n_hits": hit_count,
        "source_type_hits": dict(source_type_hits),
        "per_question": per_question_results,
    }


def _print_examples(results: dict, n: int) -> None:
    """Print full retrieval details for the first n questions."""
    logger.info("\n=== Example Retrievals: %s ===", results["label"])
    for sample in results["per_question"][:n]:
        hit_str = "✓ HIT" if sample["hit"] else "✗ MISS"
        logger.info("\n  Q: %s", sample["question"])
        logger.info("  GT: %s [%s] %s", sample["answer"], sample["answer_type"], hit_str)
        for r in sample["retrieved"]:
            rel = "✓" if r["relevant"] else " "
            logger.info(
                "    [%d] %s score=%.3f type=%-12s overlap=%d | %s",
                r["rank"], rel, r["score"], r["source_type"],
                r["token_overlap"], r["text_preview"],
            )


def _print_comparison(r1: dict, r2: dict) -> None:
    """Print a side-by-side comparison of two diagnostic runs."""
    logger.info("\n=== Comparison: %s vs %s ===", r1["label"], r2["label"])
    logger.info("%-30s %8s %8s %8s",
                "Metric", r1["label"][:8], r2["label"][:8], "Delta")
    logger.info("-" * 60)

    chr1 = r1["citation_hit_rate"]
    chr2 = r2["citation_hit_rate"]
    logger.info("%-30s %8.1f%% %8.1f%% %+8.1fpp",
                "Citation hit rate",
                chr1 * 100, chr2 * 100, (chr1 - chr2) * 100)

    # Per source-type breakdown
    all_types = sorted(set(r1["source_type_hits"]) | set(r2["source_type_hits"]))
    for st in all_types:
        c1 = r1["source_type_hits"].get(st, 0)
        c2 = r2["source_type_hits"].get(st, 0)
        logger.info("  hits %-24s %8d %8d %+8d", st, c1, c2, c1 - c2)


def main() -> None:
    args = _parse_args()

    # ── Load primary retriever ────────────────────────────────────────────────
    retriever, meta = _load_retriever(args.index_dir, args.bm25_dir)

    logger.info("Index metadata: version=%s sources=%s doc_count=%s",
                meta.get("index_version", "unknown"),
                meta.get("sources", "unknown"),
                meta.get("doc_count", "unknown"))

    # ── Load questions ────────────────────────────────────────────────────────
    questions = _load_vqarad_test_questions(args.n_samples)

    # ── Run primary diagnostic ────────────────────────────────────────────────
    label = Path(args.index_dir).name
    results = _diagnose(retriever, questions, args.top_k, label)
    _print_examples(results, args.show_examples)

    # ── Optional comparison ───────────────────────────────────────────────────
    if args.compare_index_dir:
        compare_label = Path(args.compare_index_dir).name
        logger.info("\nLoading comparison index: %s", args.compare_index_dir)
        compare_retriever, compare_meta = _load_retriever(
            args.compare_index_dir, args.compare_bm25_dir or args.bm25_dir
        )
        compare_results = _diagnose(
            compare_retriever, questions, args.top_k, compare_label
        )
        _print_comparison(results, compare_results)

    # ── Save output JSON ──────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "primary": {k: v for k, v in results.items() if k != "per_question"},
            "primary_examples": results["per_question"][:args.show_examples],
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Diagnostic results saved to %s", output_path)

    # ── Final verdict ─────────────────────────────────────────────────────────
    chr_pct = results["citation_hit_rate"] * 100
    logger.info("\n=== Phase 8A Diagnostic Verdict ===")
    if chr_pct >= 40.0:
        logger.info("✓ PASS: Citation hit rate %.1f%% >= 40%% target", chr_pct)
    elif chr_pct >= 25.0:
        logger.info("~ PARTIAL: Citation hit rate %.1f%% (target: 40%%)", chr_pct)
        logger.info("  Consider adding more QA pseudo-documents or expanding top_k.")
    else:
        logger.info("✗ FAIL: Citation hit rate %.1f%% < 25%% — index may not have QA docs", chr_pct)
        logger.info("  Verify --sources qa was used when building the index.")


if __name__ == "__main__":
    main()
