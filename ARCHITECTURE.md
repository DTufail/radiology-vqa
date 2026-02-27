# System Architecture

Grounded Multi-Agent Radiology VQA — a retrieval-augmented visual question answering system for radiology. Answers are grounded in retrieved evidence from a structured biomedical knowledge base; the system knows when to abstain.

---

## Pipeline Overview

```
┌─────────────────────────── Offline (Build Time) ──────────────────────────────┐
│                                                                                │
│  SLAKE KG CSVs ─┐                                                             │
│  RadLex.xls    ─┼─► Processors ──► Documents ──► Embedder ──► FAISSIndexer   │
│  QA pseudo-docs─┘   (KG, RadLex,   (~13K docs)   (S-PubMed  data/indices_v2  │
│                       QA)                          Bert)                      │
│                                                    BM25Index ──► data/bm25_index│
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                               loaded at startup
                                      │
                                      ▼
┌────────────────────────── Online (Per Query) ──────────────────────────────────┐
│                                                                                │
│  Image + Question                                                              │
│       │                                                                        │
│       ▼                                                                        │
│  Entry Node — validates input, infers answer_type (open/closed)               │
│       │                                                                        │
│       ▼                                                                        │
│  Visual Agent — LLaVA-Next 7B + QLoRA adapter + isotonic calibration         │
│       │  visual_answer, visual_confidence (calibrated)                        │
│       ▼                                                                        │
│  Retrieval Agent — HybridRetriever (BM25 + FAISS dense + RRF fusion)         │
│       │  retrieved_evidence [top-5 docs with scores + citations]              │
│       ▼                                                                        │
│  Supervisor — deterministic rule-based fusion                                 │
│       │  embedding agreement (PubMedBERT cosine ≥ 0.87)                      │
│       │  threshold routing: HIGH_CONF=0.60 / LOW_CONF=0.35                   │
│       ├── answer    → Output Formatter → grounded_answer + citations          │
│       ├── re_query  → Retrieval Agent (re-query with question only)           │
│       └── abstain   → Output Formatter → empty answer + reasoning             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Map

```
src/radiology_vqa/
│
├── config.py                  Settings (pydantic-settings); all config fields,
│                              env-var overridable, .env file support
├── schema.py                  Core data models: VQASample, SLAKESample, KGTriple
├── loader.py                  VQA-RAD and PathVQA HuggingFace loaders
├── slake_loader.py            SLAKE JSON loader with English-only filter
├── kg_loader.py               SLAKE KG CSV loader (#-delimited)
│
├── rag/
│   ├── document.py            DocumentMeta, Document, RetrievalResult (Pydantic)
│   ├── embedder.py            Embedder wrapping S-PubMedBert-MS-MARCO
│   ├── chunker.py             Word-based text chunker with configurable overlap
│   ├── kg_processor.py        KGTriple → NL Documents (templates + summaries)
│   ├── radlex_processor.py    RadLex.xls → Documents (Tier 1 filter, 3,737 docs)
│   ├── qa_pseudo_processor.py VQA-RAD / SLAKE QA pairs → pseudo-documents
│   ├── indexer.py             FAISSIndexer (IndexFlatIP, offline build)
│   └── retriever.py           Retriever (dense) + HybridRetriever (BM25+dense+RRF)
│
├── vlm/
│   ├── interface.py           VLMPrediction (Pydantic), VLMInterface (Protocol)
│   ├── llava.py               LLaVABackend: LLaVA-Next 7B + LoRA + calibration
│   ├── blip2.py               BLIP2Backend (alternative backend)
│   └── factory.py             create_vlm_backend(config) → VLMInterface
│
├── agents/
│   ├── state.py               AgentState (TypedDict), SystemOutput (Pydantic)
│   ├── visual_agent.py        visual_agent_node: calls VLM, populates visual_*
│   ├── retrieval_agent.py     retrieval_agent_node: calls Retriever, populates evidence
│   ├── supervisor.py          supervisor_node: threshold routing + agreement scoring
│   └── output_formatter.py   output_formatter_node: builds final SystemOutput
│
├── graph/
│   ├── entry.py               entry_node: validates image/question, infers answer_type
│   ├── routing.py             route_after_supervisor: answer/re_query/abstain → edges
│   ├── builder.py             GraphBuilder.build() (real), build_lightweight() (test)
│   └── runner.py              AgentRunner.run_query() / run_batch(), create_runner()
│
├── calibration/
│   ├── platt.py               Platt scaling (sigmoid fit, 2-param)
│   └── isotonic.py            Isotonic regression (non-parametric, JSON-serialised)
│
├── training/
│   ├── dataset.py             build_training_dataset(): VQA-RAD + SLAKE + PathVQA
│   └── collator.py            LlavaDataCollator: padding + label masking
│
└── evaluation/
    ├── result.py              PerSampleResult, EvaluationResult, ComparisonResult
    ├── evaluator.py           AgentEvaluator: lazy loading, crash recovery, resume
    ├── comparator.py          BaselineComparator: McNemar's test, grounding breakdown
    ├── report.py              generate_report(): markdown + JSON
    ├── metrics.py             exact_match, closed_precision_recall_f1, token_f1, BLEU
    ├── agent_metrics.py       abstention_rate, accuracy_when_answered, re_query_rate
    └── calibration.py         ECE, AUROC, calibration_bins, threshold_analysis
```

---

## Data Flow (Single Query)

1. **Entry node** receives `{image, question}`. Infers `answer_type` ("open"/"closed") from question-word heuristics. Validates both fields are non-empty.

2. **Visual agent** calls `LLaVABackend.predict(image, question)`. The backend generates a text answer and extracts confidence from next-token log-probabilities. The isotonic calibrator transforms the raw confidence score (was typically 0.93) into a calibrated probability. Returns `visual_answer` and `visual_confidence`.

3. **Retrieval agent** queries the `HybridRetriever`: BM25 retrieves 20 candidates, dense FAISS retrieves 20 candidates, RRF fusion (k=60) merges and re-ranks them, top-5 are returned. On re-query pass (retry_count > 0), the query is the question only (visual_answer suffix dropped). Returns `retrieved_evidence` with text, citation, score, entity_name.

4. **Supervisor** applies the 5-case routing logic:
   - Computes `agreement_score` via PubMedBERT cosine similarity between the query embedding and evidence embeddings (threshold 0.87).
   - `visual_confidence ≥ 0.60 + agreement > 0` → **answer** (Case A, grounded_confidence = conf × 0.75 + 0.25 × agreement)
   - `visual_confidence ≥ 0.60 + no agreement` → **re_query** if retry < 1, else **abstain** (Case B)
   - `0.35 ≤ visual_confidence < 0.60 + agreement > 0` → **answer** (Case C, grounded_confidence = conf × 0.3 + 0.7 × agreement)
   - `0.35 ≤ visual_confidence < 0.60 + no agreement` → **re_query** / **abstain** (Case D)
   - `visual_confidence < 0.35` → **abstain** regardless of evidence (Case E)

5. **Output formatter** serialises the final decision into `SystemOutput`. Abstentions have an empty `final_answer` with `decision="abstain"` and `reasoning` explaining why.

---

## Key Design Decisions

**1. Deterministic supervisor over LLM-based arbitration.**
The supervisor is pure Python with no LLM calls. Every routing decision is reproducible from the same inputs. This is essential for clinical auditability: a doctor reviewing a system output can trace exactly why an answer was given or withheld.

**2. Selective prediction (abstention) as a safety primitive.**
The system is designed to answer fewer questions rather than answer wrong ones confidently. Abstention is a first-class output, not a fallback. The AUROC of 0.868 means confidence scores are a reliable signal of correctness — the system knows what it doesn't know.

**3. QLoRA over full fine-tuning.**
Training a 7B parameter model to convergence requires ~22 GB VRAM. QLoRA (4-bit NF4 quantization + rank-16 LoRA on q/v projection layers) reduces this to ~14 GB while achieving comparable accuracy gains. The adapter adds 13M trainable parameters over the frozen 7B base.

**4. Protocol-based VLM abstraction.**
`VLMInterface` is a `@runtime_checkable` Protocol. Backends (LLaVA, BLIP2) are interchangeable without modifying the rest of the pipeline. Tests inject `MockVLMBackend` without touching agent or graph code.

**5. Configuration-driven A/B testing.**
Every Phase 6 improvement has a config flag with the previous phase's behavior as its default. `retrieval_method="dense"` preserves Phase 5 behavior; `"hybrid"` enables Phase 6B. `agreement_method="keyword"` restores Phase 5 supervisor; `"embedding"` is Phase 6B-3. `calibration_method="none"` is uncalibrated; `"isotonic"` is Phase 6C. This means any Phase 6 config can be reproduced by environment variables with no code changes.

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| VLM | `transformers` (LlavaForConditionalGeneration) | ≥4.37 |
| LoRA adapter | `peft` | ≥0.9 |
| Quantization | `bitsandbytes` | ≥0.41 (CUDA only) |
| Embeddings | `sentence-transformers` (S-PubMedBert-MS-MARCO) | ≥2.2 |
| Vector index | `faiss-cpu` (IndexFlatIP) | ≥1.7 |
| BM25 | `rank_bm25` | ≥0.2 |
| Agent graph | `langgraph` | ≥0.2 |
| Config | `pydantic-settings` | ≥2.0 |
| Calibration | `scikit-learn` (IsotonicRegression) | ≥1.3 |
| Training | `trl` (SFTTrainer) | ≥0.8 |
| Tests | `pytest` + `pytest-mock` | ≥7.0 |
| Hardware | NVIDIA A10G (SageMaker ml.g5.2xlarge) | 22.1 GB VRAM |
