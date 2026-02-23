# Phase 4 — Multi-Agent Pipeline

Phase 4 is split into two sub-phases:
- **Phase 4A** — pure-function agent nodes (no LangGraph dependency)
- **Phase 4B** — LangGraph graph wiring, conditional routing, and re-query loop

---

## Goal

Replace the raw VLM benchmark approach with a grounded, supervised pipeline. Instead of passing the VLM answer directly, a supervisor fuses the VLM prediction with evidence retrieved from the SLAKE knowledge graph. If agreement is low the pipeline retries retrieval once and, if still unsupported, abstains and flags the case for human review.

---

## Phase 4A — Agent Nodes

All nodes are **pure functions** — they take an `AgentState` dict and return an updated `AgentState` dict. No LangGraph imports.

### Shared State (`src/radiology_vqa/agents/state.py`)

`AgentState` is a `TypedDict` with `total=False` (all keys optional). It is progressively populated as the pipeline runs:

| Stage | Keys added |
|-------|-----------|
| Input | `image`, `question`, `answer_type` |
| Visual agent | `visual_answer`, `visual_confidence`, `visual_raw_output`, `visual_model`, `visual_error` |
| Retrieval agent | `retrieval_query`, `retrieved_evidence`, `retrieval_error` |
| Supervisor | `decision`, `decision_reasoning`, `agreement_score`, `grounded_answer`, `grounded_confidence`, `retry_count` |
| Output formatter | `final_answer`, `final_confidence`, `citations`, `requires_human_review`, `output_reasoning` |

`SystemOutput` is a Pydantic model — the public-facing result returned by `AgentRunner.run_query()`.

### Visual Agent (`src/radiology_vqa/agents/visual_agent.py`)

Calls `vlm.predict(image, question)` and writes `visual_answer`, `visual_confidence`, `visual_raw_output`, `visual_model` into state. Any exception is caught and written to `visual_error` — the node never raises.

### Retrieval Agent (`src/radiology_vqa/agents/retrieval_agent.py`)

Constructs a retrieval query as `"{question} {visual_answer}"` (or question-only when the VLM errored). Calls `retriever.retrieve(query, top_k)` and serialises `RetrievalResult` objects to plain dicts (JSON-safe for LangGraph). Errors written to `retrieval_error`.

### Supervisor (`src/radiology_vqa/agents/supervisor.py`)

Deterministic fusion with no LLM calls. Thresholds calibrated to VQA-RAD:

```
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE  = 0.55
```

Five decision cases:

| Case | Condition | Decision |
|------|-----------|----------|
| A | conf ≥ HIGH and agreement ≥ 0.5 | `answer` |
| B | conf ≥ HIGH and agreement < 0.5 and retries left | `re_query` |
| C | conf < LOW or visual_error | `abstain` |
| D | LOW ≤ conf < HIGH and agreement < 0.5 and retries left | `re_query` |
| E | LOW ≤ conf < HIGH and agreement ≥ 0.5 | `answer` |
| — | retries exhausted (B or D, 2nd pass) | `abstain` |

Agreement scoring: keyword overlap between the retrieved evidence texts and the VLM answer + question tokens. Phase 6 can upgrade this to semantic similarity.

`retry_count` is incremented by the supervisor whenever it emits `re_query`.

### Output Formatter (`src/radiology_vqa/agents/output_formatter.py`)

- `decision = "answer"`: writes `grounded_answer` to `final_answer`, top-3 citations by score
- `decision = "abstain"` (or unexpected): writes `"ABSTAIN: Unable to provide a reliable answer"`, `final_confidence=0.0`, `requires_human_review=True`, empty citations

`format_system_output(state)` converts the final `AgentState` → `SystemOutput` (Pydantic).

---

## Phase 4B — LangGraph Wiring

### Entry Node (`src/radiology_vqa/graph/entry.py`)

First node in the graph. Responsibilities:

- Validates image: converts to RGB, resizes if any dimension > 4096 px
- Validates question: non-empty after strip
- Infers `answer_type` from first word (`is/are/does/do/was/were/has/have/can/will/should` → `"closed"`, else `"open"`)
- Initialises `retry_count = 0` and clears error fields
- Never raises — all failures go to `visual_error` so the supervisor can route to abstain

### Routing (`src/radiology_vqa/graph/routing.py`)

```
route_after_supervisor(state) → node_name

  "answer"    → "output_formatter"
  "re_query"  → "retrieval_agent"   (if retry_count ≤ max_retries)
               → "output_formatter"  (safety bound — prevents infinite loop)
  "abstain"   → "output_formatter"
  unknown     → "output_formatter"
```

`max_retries` comes from `settings.supervisor_max_retries` (default 1).

### Graph (`src/radiology_vqa/graph/builder.py`)

`GraphBuilder.build()` wires the real pipeline (loads VLM + Retriever):

```
START
  └─► entry
        └─► visual_agent
              └─► retrieval_agent
                    └─► supervisor
                          ├─(answer/abstain)─► output_formatter ─► END
                          └─(re_query)────────► retrieval_agent  (loop, max 1 retry)
```

`build_lightweight()` substitutes passthrough nodes that forward pre-populated state values. Used for fast tests without a GPU or FAISS index.

### Runner (`src/radiology_vqa/graph/runner.py`)

```python
runner = create_runner()                           # loads VLM + Retriever once
result = runner.run_query(image, question)         # → SystemOutput
results = runner.run_batch(samples)                # → list[SystemOutput]
```

`run_query` never raises — a catastrophic pipeline failure returns an abstain `SystemOutput` with the error message.

### CLI (`scripts/run_agent.py`)

```bash
# Single sample from dataset
python scripts/run_agent.py --dataset vqa_rad --index 0

# Multiple samples
python scripts/run_agent.py --dataset vqa_rad --index 0 --index 5 --index 10

# Range
python scripts/run_agent.py --dataset vqa_rad --range 0 20 --output results.json

# Local image
python scripts/run_agent.py --image path/to/xray.jpg --question "Is there pneumonia?"
```

---

## Architecture Diagram

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    AgentRunner.run_query()                    │
  │                                                              │
  │  entry ──► visual_agent ──► retrieval_agent ──► supervisor   │
  │                                   ▲                  │       │
  │                                   │    re_query       │       │
  │                                   └──────────────────┘       │
  │                                           │ answer/abstain   │
  │                                           ▼                  │
  │                                   output_formatter           │
  │                                           │                  │
  │                                           ▼                  │
  │                                     SystemOutput             │
  └──────────────────────────────────────────────────────────────┘
```

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| `tests/test_agent_state.py` | AgentState / SystemOutput validation | Pass |
| `tests/test_visual_agent.py` | MockVLMBackend, error handling | Pass |
| `tests/test_retrieval_agent.py` | MockRetriever, query construction, serialisation | Pass |
| `tests/test_supervisor.py` | All 5 cases (A–E), re-query count, abstain paths | Pass |
| `tests/test_output_formatter.py` | Answer path, abstain path, citations top-3 | Pass |
| `tests/test_entry_node.py` | Image validation, RGB conversion, resize, answer_type inference | Pass |
| `tests/test_routing.py` | All routing branches, safety bound | Pass |
| `tests/test_graph_builder.py` | Lightweight graph structure + execution, `TestReQueryLoop` | Pass |
| `tests/test_graph_integration.py` | Real VLM + FAISS (slow, `@pytest.mark.slow`) | Skip in CI |
| **Total Phase 4A fast tests** | **166 / 166** | **All pass** |
| **Total Phase 4B fast tests** | **55 / 55** | **All pass** |
| **Combined Phase 4 fast tests** | **221 / 221** | **All pass** |

Including the re-query loop tests added in the post-audit fix (C12):

| Suite | Tests | Status |
|-------|-------|--------|
| `TestReQueryLoop` in `test_graph_builder.py` | 9 tests — loop terminates, retry_count, abstain path | Pass |
| **Grand total (all phases)** | **230 / 230** | **All pass** |

---

## Pipeline Evaluation — Phase 4 Results

Evaluated on a **20-sample pilot** from VQA-RAD test split using the full multi-agent pipeline (LLaVA v1.6 4-bit + FAISS KG index, SageMaker ml.g4dn.xlarge).

Result file: `phase4_results.json` (20 samples, vqa_rad test split)

### Decision Breakdown

| Decision | Count | % |
|----------|-------|---|
| Answer | 11 | 55% |
| Abstain | 9 | 45% |

### Accuracy on Answered Cases

| Metric | Value |
|--------|-------|
| Correct / answered | 4 / 11 = **36.4%** |
| Wrong / answered | 7 / 11 |
| Overall accuracy (answered only, over 20 total) | 4 / 20 = **20.0%** |

Broken down by answer type:

| Type | Answered | Correct | Accuracy |
|------|----------|---------|----------|
| Closed | 8 | 4 | **50.0%** |
| Open | 3 | 0 | **0.0%** |

### Abstain Analysis

Of 9 abstained cases:
- **5 protected correctly** — VLM answer was wrong; abstaining was the right call
- **3 over-abstained** — VLM was actually right but the KG had no matching evidence
- **1 borderline** — low VLM confidence (0.507 < 0.55 threshold), correct to abstain

The 3 over-abstained cases are a KG coverage gap: questions about imaging modality ("X-ray"), spatial orientation ("colon prominence"), and AP/PA comparison don't map to disease/organ entities in the SLAKE KG.

### Qualitative Observations

1. **Closed yes/no questions** — pipeline performs best when the question entity appears in the KG (e.g., "temporal bone fracture", "brain atrophy", "liver"). Agreement scoring reliably finds supporting evidence.

2. **Entity-keyword agreement limitation** — for sample 0 ("is there evidence of an aortic aneurysm?"), the supervisor returned `decision=answer` with agreement=1.0 even though the VLM said "No" and ground truth was "yes". The KG matched "Aortic Aneurysm" based on question keywords, not the binary VLM answer. This is the known limitation of keyword-based agreement.

3. **Open-ended anatomical questions** — "which side of the heart border is obscured?" and similar questions require spatial reasoning that the KG doesn't encode. The KG returns entity descriptions (organ anatomy, disease definitions) rather than image-specific spatial facts.

4. **Pipeline stability** — no crashes, no infinite loops, all 20 runs terminated with well-formed `SystemOutput` objects.

### Comparison to Standalone VLM

| Approach | Answered correctly | Wrong answers returned | Abstains |
|----------|--------------------|----------------------|---------|
| Raw LLaVA (Phase 3) | 41.2% on 451 | 58.8% returned | None |
| Agent pipeline (Phase 4, 20 samples) | 20.0% overall | 35% returned wrong | 45% abstained |

The pipeline trades raw accuracy for reliability: it returns fewer wrong answers by abstaining on uncertain cases. The 45% abstain rate reflects the current KG coverage — expanding the knowledge base is the primary lever for improving recall in later phases.

---

## Known Limitations

| Issue | Impact | Plan |
|-------|--------|------|
| Keyword-based agreement can't detect wrong binary answers | High-confidence wrong answers can slip through (cases 0, 18, 19) | Phase 6: semantic agreement using embeddings |
| SLAKE KG doesn't cover imaging modality or spatial relations | Abstains on "what type of image?" and "which side?" questions | Phase 5: expand index with radiology-specific text |
| Open-ended accuracy low even when answered | VLM generates short labels that may not match ground truth phrasing | Phase 6: normalised answer comparison |
| `requires_human_review=False` on wrong confident answers | False sense of reliability | Phase 6: per-class confidence calibration |
