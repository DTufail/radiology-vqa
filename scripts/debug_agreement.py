"""Debug script: diagnose agreement scoring on a real consolidation sample.

Usage:
    python scripts/debug_agreement.py

What it does:
    1. Loads the FAISS retriever from data/indices/
    2. Issues query "is there consolidation in the lungs? Yes" → top_k=5
    3. Prints every evidence doc (full text, score, entity_name)
    4. Runs _compute_agreement() manually and shows every step
    5. Runs supervisor_node() on the full simulated state and prints the decision
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import re

# ── 1. Import internals ────────────────────────────────────────────────────────

from radiology_vqa.agents.supervisor import (
    EVIDENCE_SUPPORT_THRESHOLD,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    _STOP_WORDS,
    _compute_agreement,
    _keyword_in_text,
    supervisor_node,
)
from radiology_vqa.config import settings

print("=" * 70)
print("SUPERVISOR MODULE IDENTITY CHECK")
print("=" * 70)
import radiology_vqa.agents.supervisor as _sup_mod
print(f"  Module file : {_sup_mod.__file__}")
print(f"  HIGH_CONF   : {_sup_mod.HIGH_CONFIDENCE}")
print(f"  LOW_CONF    : {_sup_mod.LOW_CONFIDENCE}")
print(f"  'image' in _STOP_WORDS  : {'image' in _STOP_WORDS}")
print(f"  'visible' in _STOP_WORDS: {'visible' in _STOP_WORDS}")
print(f"  'evidence' in _STOP_WORDS: {'evidence' in _STOP_WORDS}")
print(f"  _STOP_WORDS size        : {len(_STOP_WORDS)}")
has_dual = "va_norm" in open(_sup_mod.__file__).read()
print(f"  Dual-signal code present: {has_dual}")

# ── 2. Load retriever ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("LOADING RETRIEVER")
print("=" * 70)

index_dir = settings.index_dir
print(f"  index_dir: {index_dir}")

index_faiss = index_dir / "index.faiss"
index_docs  = index_dir / "documents.jsonl"
if not index_faiss.exists():
    print(f"  ERROR: {index_faiss} not found — build the index first.")
    sys.exit(1)

from radiology_vqa.rag.embedder import Embedder
from radiology_vqa.rag.retriever import Retriever

print("  Loading embedder…")
embedder = Embedder()
print("  Loading retriever…")
retriever = Retriever(index_dir=index_dir, embedder=embedder)
print("  Done.")

# ── 3. Issue the two queries ──────────────────────────────────────────────────

QUESTION     = "is there consolidation in the lungs?"
VISUAL_ANSWER = "Yes"
VISUAL_CONF   = 0.892
ANSWER_TYPE   = "closed"

query_first   = f"{QUESTION} {VISUAL_ANSWER}"
query_requery = QUESTION

for label, query in [("FIRST QUERY  (retry=0)", query_first),
                     ("RE-QUERY     (retry=1)", query_requery)]:
    print("\n" + "=" * 70)
    print(f"{label}")
    print(f"  query={query!r}")
    print("=" * 70)

    results = retriever.retrieve(query, top_k=5, min_score=0.0)

    if not results:
        print("  No results returned.")
        continue

    evidence_dicts = []
    for r in results:
        d = {
            "text":        r.document.text,
            "score":       r.score,
            "source_type": r.document.meta.source_type,
            "entity_name": r.document.meta.entity_name,
            "attribute":   r.document.meta.attribute,
            "rank":        r.rank,
        }
        evidence_dicts.append(d)
        above = "✓" if r.score >= EVIDENCE_SUPPORT_THRESHOLD else "✗ (below threshold)"
        print(f"\n  [{r.rank}] score={r.score:.4f} {above}")
        print(f"       entity : {r.document.meta.entity_name!r}")
        print(f"       attr   : {r.document.meta.attribute!r}")
        print(f"       text   : {r.document.text!r}")

    # ── 4. Manual keyword extraction trace ────────────────────────────────────

    print(f"\n  KEYWORD EXTRACTION TRACE (question={QUESTION!r}, answer_type={ANSWER_TYPE!r})")

    va_norm = VISUAL_ANSWER.strip().lower()
    use_q   = (ANSWER_TYPE == "closed") or (va_norm in ("yes", "no"))
    print(f"    va_norm={va_norm!r}  use_question_keywords={use_q}")

    if use_q:
        words    = re.findall(r"[a-z]+", QUESTION.lower())
        keywords = {w for w in words if len(w) > 2 and w not in _STOP_WORDS}
        print(f"    raw words from question  : {words}")
        print(f"    after len>2 + stop filter: {sorted(keywords)}")
        if not keywords:
            words2   = re.findall(r"[a-z]+", VISUAL_ANSWER.lower())
            keywords = {w for w in words2 if len(w) > 2}
            print(f"    fallback (visual_answer) : {sorted(keywords)}")
    else:
        words    = re.findall(r"[a-z]+", VISUAL_ANSWER.lower())
        keywords = {w for w in words if len(w) > 2}
        print(f"    raw words from visual_answer: {words}")
        print(f"    keywords                    : {sorted(keywords)}")

    print(f"    FINAL KEYWORDS: {sorted(keywords)}")

    # ── 5. Per-evidence matching trace ────────────────────────────────────────

    print(f"\n  EVIDENCE MATCHING (support_threshold={EVIDENCE_SUPPORT_THRESHOLD})")
    supporting = []
    for i, item in enumerate(evidence_dicts):
        score_ok = item["score"] >= EVIDENCE_SUPPORT_THRESHOLD
        text_lower   = item["text"].lower()
        entity_lower = item["entity_name"].lower()
        matched_kw = [kw for kw in keywords
                      if _keyword_in_text(kw, text_lower) or _keyword_in_text(kw, entity_lower)]
        match = bool(matched_kw) and score_ok
        print(f"    [{i+1}] score={item['score']:.4f} above_thresh={score_ok} "
              f"matched_kw={matched_kw} → {'SUPPORT' if match else 'skip'}")
        if match:
            supporting.append(item)

    agreement = len(supporting) / len(evidence_dicts) if evidence_dicts else 0.0
    print(f"\n    supporting={len(supporting)}/{len(evidence_dicts)}  agreement={agreement:.4f}")

    # ── 6. supervisor_node() on this evidence ─────────────────────────────────

    state = {
        "question":          QUESTION,
        "answer_type":       ANSWER_TYPE,
        "visual_answer":     VISUAL_ANSWER,
        "visual_confidence": VISUAL_CONF,
        "visual_raw_output": VISUAL_ANSWER,
        "visual_model":      "debug",
        "visual_error":      "",
        "retrieved_evidence": evidence_dicts,
        "retrieval_query":   query,
        "retrieval_error":   "",
        "retry_count":       0 if "FIRST" in label else 1,
    }
    out = supervisor_node(state)
    print(f"\n  SUPERVISOR RESULT:")
    print(f"    decision           = {out['decision']!r}")
    print(f"    agreement_score    = {out['agreement_score']:.4f}")
    print(f"    grounded_confidence= {out['grounded_confidence']:.4f}")
    print(f"    reasoning          = {out['decision_reasoning']!r}")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
