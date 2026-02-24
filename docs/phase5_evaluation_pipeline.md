# Phase 5 — Evaluation Pipeline

Phase 5 is split into two sub-phases:
- **Phase 5A** — pure-function metrics library (no model loading, no file I/O)
- **Phase 5B** — evaluation orchestrator, comparator, and report generator

---

## Goal

Measure the value of the multi-agent pipeline quantitatively. Phase 5 produces three artefacts:

1. **Agent evaluation result** — runs the full Phase 4 graph on a dataset split and records every metric.
2. **VLM-only baseline** — runs the VLM directly (no RAG, no supervisor) on the same split for comparison.
3. **Comparison report** — statistical analysis (McNemar's test) of whether RAG grounding improves accuracy, plus a markdown report suitable for a paper appendix.

---

## Phase 5A — Metrics Library

All Phase 5A functions are **pure**: no side-effects, no model loading, no file I/O. They operate on batched Python lists and can be called from tests without any GPU.

### Standard VQA Metrics (`src/radiology_vqa/evaluation/metrics.py`)

#### `normalize_answer(answer: str) → str`

Normalises an answer string for comparison. Steps (in order):
1. Lowercase + strip
2. Remove punctuation except hyphens within words
3. Remove articles: `"a"`, `"an"`, `"the"`
4. Collapse whitespace

Examples: `"The Left Lung"` → `"left lung"`, `"Yes."` → `"yes"`, `"X-ray"` → `"x-ray"`.

This matches the normalization used in VQA-RAD, SLAKE, and PathVQA papers.

#### `exact_match_accuracy(predictions, ground_truths) → float`

Primary metric. Fraction of predictions that match ground truth after normalization.

#### `closed_precision_recall_f1(predictions, ground_truths) → dict`

Binary P/R/F1 for yes/no questions. Treats `"yes"` as positive class. Exposes the yes-bias present in many VLMs: LLaVA v1.6 predicts `"yes"` 165/251 times on VQA-RAD closed questions. A yes-biased model achieves 60% accuracy if the ground truth is 60% `"yes"` but has low precision on `"no"`. F1 exposes this.

#### `token_f1(prediction, ground_truth) → float` / `batch_token_f1(...) → float`

Standard SQuAD token-level F1. Normalises both strings, tokenises by whitespace, computes precision/recall on word sets. Good for open-ended answers where partial credit is meaningful (e.g., `"left lung"` vs `"left lower lung"`).

#### `bleu_1(prediction, ground_truth) → float` / `batch_bleu_1(...) → float`

Unigram BLEU with nltk smoothing. Medical VQA answers are 1–3 words so higher-order BLEU is meaningless. BLEU-1 measures whether the model produced the correct medical term.

#### `bert_score_f1(predictions, ground_truths, ...) → dict`

BERTScore using contextual embeddings (`microsoft/deberta-xlarge-mnli` by default). Falls back to `bert-base-uncased` on OOM. Returns `{"precision", "recall", "f1"}`.

#### `compute_all_metrics(predictions, ground_truths, answer_types, ...) → dict`

One-call convenience: splits samples by answer type (`"closed"` / `"open"`), computes all metrics, returns flat dict:

| Key | Description |
|-----|-------------|
| `overall_accuracy` | Exact match over all samples |
| `closed_accuracy` | Exact match on closed (yes/no) questions |
| `closed_precision` / `_recall` / `_f1` | Binary P/R/F1 |
| `closed_confusion` | `{"tp", "tn", "fp", "fn"}` |
| `closed_count` | Number of closed samples |
| `open_accuracy` | Exact match on open questions |
| `open_token_f1` | Mean token F1 on open questions |
| `open_bleu_1` | Mean BLEU-1 on open questions |
| `open_bertscore_f1` / `_precision` / `_recall` | BERTScore (−1.0 if skipped) |
| `open_count` | Number of open samples |
| `total_count` | Total samples |

---

### Agent-Specific Metrics (`src/radiology_vqa/evaluation/agent_metrics.py`)

These metrics evaluate **decision quality** — when to answer, when to abstain, when to re-query — not just answer quality. Standard VQA papers don't report these.

#### `abstention_rate(decisions) → float`

Fraction of samples where `decision == "abstain"`. Neither high nor low is inherently good — interpret alongside `accuracy_when_answered`.

#### `accuracy_when_answered(predictions, ground_truths, decisions) → float`

Exact match accuracy computed **only on non-abstain samples**. This is the selective prediction metric:

- If `accuracy_when_answered > VLM-only accuracy` → the agent abstains on the right cases (uncertain ones), raising precision.
- If `accuracy_when_answered < VLM-only accuracy` → the supervisor's routing is broken: the agent is answering the *wrong* questions confidently.

#### `correct_abstention_rate(vlm_only_predictions, ground_truths, decisions) → float`

Of samples the agent abstained on, what fraction would the VLM have gotten wrong? High value = the system correctly identifies cases where the VLM would fail. Low value = over-abstention on cases the VLM handles fine.

#### `re_query_rate_from_counts(retry_counts) → float`

Fraction of samples that triggered at least one re-query (`retry_count > 0`).

#### `grounding_improvement(agent_preds, vlm_preds, ground_truths, decisions) → dict`

Per-sample categorization into:

| Category | Meaning |
|----------|---------|
| `improved` | Agent correct AND VLM wrong — RAG helped |
| `degraded` | Agent wrong AND VLM correct — RAG hurt |
| `both_correct` | Both right — RAG didn't change the outcome |
| `both_wrong` | Both wrong — neither approach works |
| `agent_abstained` | Agent abstained (further split below) |
| `abstain_vlm_correct` | Abstained but VLM would have been right (over-abstention) |
| `abstain_vlm_wrong` | Abstained and VLM also wrong (justified abstention) |
| `net_improvement` | `improved − degraded` (the headline number) |

#### `citation_relevance(citations_per_sample, ground_truths) → dict`

Keyword-based check: for each cited sample, does any citation text contain a ground-truth token (length > 2)? Returns `citation_hit_rate` and `mean_relevant_citations`.

---

### Confidence Calibration (`src/radiology_vqa/evaluation/calibration.py`)

Calibration measures whether confidence scores are meaningful. A model saying "0.9 confidence" should be correct ~90% of the time. If poorly calibrated, the supervisor's thresholds (0.85, 0.55) are arbitrary rather than meaningful decision boundaries.

#### `expected_calibration_error(confidences, correct, n_bins=10) → float`

ECE partitions predictions into equal-width confidence bins and measures `sum(bin_weight × |bin_accuracy − bin_confidence|)`. ECE = 0 is perfect calibration; ECE = 1 is maximally miscalibrated. LLaVA v1.6 on VQA-RAD typically achieves ECE ≈ 0.1–0.2.

#### `calibration_bins(confidences, correct, n_bins=10) → list[dict]`

Per-bin data for reliability diagrams: `bin_start`, `bin_end`, `count`, `mean_confidence`, `accuracy`, `gap` (accuracy − confidence).

#### `confidence_discrimination(confidences, correct) → dict`

Measures whether high-confidence predictions are actually more often correct:
- `mean_correct_confidence` — mean confidence on correct predictions
- `mean_wrong_confidence` — mean confidence on wrong predictions
- `auroc` — AUC of confidence as a binary classifier (correct vs wrong). AUROC = 0.5 means confidence is random; AUROC > 0.7 means it has real signal.

#### `threshold_analysis(confidences, correct, thresholds) → list[dict]`

For each confidence threshold, compute coverage (fraction of samples above threshold), accuracy on those samples, and count. Used to recommend an optimal operating threshold for the supervisor.

---

## Phase 5B — Evaluation Orchestrator

### Data Models (`src/radiology_vqa/evaluation/result.py`)

All models are Pydantic with `.save(path)` / `.load(path)` for JSON roundtrip.

#### `PerSampleResult`

One row per evaluated sample:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | str | Dataset sample identifier |
| `question` | str | Input question |
| `ground_truth` | str | Correct answer |
| `prediction` | str | System prediction (empty string on error) |
| `correct` | bool | `normalize(prediction) == normalize(ground_truth)` |
| `answer_type` | str | `"closed"` or `"open"` |
| `confidence` | float | VLM or agent confidence [0, 1] |
| `latency_seconds` | float | Wall-clock time for this sample |
| `decision` | str | `"answer"` / `"abstain"` / `""` (vlm_only) |
| `citations` | list[dict] | Top-3 retrieved documents (agent mode only) |
| `reasoning` | str | Supervisor reasoning |
| `retrieval_query` | str | Query sent to FAISS |
| `visual_answer` | str | Raw VLM answer before grounding |
| `retry_count` | int | Number of re-query loops (default 0) |

#### `EvaluationResult`

Full run result. Key fields beyond all metrics from `compute_all_metrics`:

| Field | Description |
|-------|-------------|
| `evaluation_mode` | `"agent"` or `"vlm_only"` |
| `config_snapshot` | Dict of supervisor thresholds, VLM model ID, etc. |
| `abstention_rate` | Fraction abstained (agent mode only) |
| `accuracy_when_answered` | Accuracy on non-abstain samples |
| `re_query_rate` | Fraction that triggered a retry |
| `citation_relevance_hit_rate` | Fraction of cited samples with a relevant citation |
| `ece` | Expected Calibration Error |
| `confidence_auroc` | AUC of confidence as correct/wrong discriminator |
| `calibration_bins` | List of per-bin dicts (for reliability diagrams) |
| `threshold_analysis` | Coverage/accuracy at each threshold |
| `total_seconds` | Total wall-clock time |
| `mean_latency_seconds` | Mean per-sample latency |
| `median_latency_seconds` | Median per-sample latency |
| `per_sample` | List of all `PerSampleResult` objects |

#### `ComparisonResult`

Output of `BaselineComparator.compare()`. Contains metric deltas (agent − baseline), the full grounding breakdown (`improved`, `degraded`, `both_correct`, `both_wrong`, `agent_abstained`, `net_improvement`), McNemar test results (`mcnemar_statistic`, `mcnemar_p_value`, `is_significant`), and two markdown tables (`comparison_table_md`, `grounding_table_md`).

---

### `AgentEvaluator` (`src/radiology_vqa/evaluation/evaluator.py`)

```python
evaluator = AgentEvaluator(settings)

agent_result   = evaluator.evaluate(dataset="vqa_rad", split="test", mode="agent")
baseline_result = evaluator.evaluate(dataset="vqa_rad", split="test", mode="vlm_only")
```

**Lazy loading**: `__init__` does NOT load any model. `_agent_runner` and `_vlm` are `None` until the first `evaluate()` call. This means creating an `AgentEvaluator` to load saved results has no cost.

**`evaluate()` steps:**

1. Load dataset samples via `_load_dataset()` (supports `"vqa_rad"` and `"pathvqa"`).
2. If `resume_from` is provided and the file exists, load completed samples and skip their `sample_id`s.
3. For each sample: call `_run_agent()` or `_run_vlm_only()`. Per-sample exceptions are caught — prediction recorded as `""`, correct=False, evaluation continues.
4. Every 25 samples: log progress (accuracy so far, abstention count, mean latency, ETA).
5. Every 50 samples: save intermediate results to `{eval_output_dir}/intermediate_{mode}_{dataset}_{split}.json` for crash recovery.
6. After all samples: compute aggregate metrics using all Phase 5A pure functions.
7. Return `EvaluationResult`.

**Memory management in compare mode**: After the agent run completes, the calling code should `del evaluator; evaluator = AgentEvaluator(settings)` before the vlm_only run. This frees the `AgentRunner`'s embedded VLM (~4.5 GB on T4) before loading the standalone VLM, avoiding two VLMs in GPU memory simultaneously (~9 GB).

---

### `BaselineComparator` (`src/radiology_vqa/evaluation/comparator.py`)

```python
comparator = BaselineComparator()
comparison = comparator.compare(agent_result, baseline_result)
comparison.save(Path("data/evaluation_reports/comparison.json"))
```

**`compare()` steps:**

1. **Align by `sample_id`** — builds `{sample_id: result}` dicts for both runs and takes the intersection. Logs a warning if counts differ.
2. **Metric deltas** — `agent_accuracy − baseline_accuracy` for overall, closed, open, token F1, BERTScore.
3. **Grounding improvement** — calls `grounding_improvement()` from `agent_metrics.py` to get the `improved / degraded / both_correct / both_wrong / agent_abstained` breakdown.
4. **Correct abstention rate** — calls `correct_abstention_rate()` using baseline VLM predictions as the reference.
5. **McNemar's test** — tests statistical significance of the difference:
   - `b+c < 5` discordant pairs → `(stat=0.0, p=1.0)` (insufficient data)
   - `b+c < 25` → exact binomial test (`scipy.stats.binomtest`)
   - `b+c ≥ 25` → chi-squared with continuity correction
6. **Format markdown tables** — `comparison_table_md` (metric/VLM/agent/delta rows) and `grounding_table_md` (category/count/% rows).

---

### `generate_report()` (`src/radiology_vqa/evaluation/report.py`)

```python
result_dir = generate_report(
    agent_result=agent_result,
    baseline_result=baseline_result,   # optional
    comparison=comparison,             # optional
    output_dir=Path("data/evaluation_reports"),
)
```

Creates the following files in `output_dir`:

| File | Contents |
|------|----------|
| `report.md` | Full markdown report |
| `report.json` | Metadata summary (accuracy, ECE, AUROC, abstention rate, McNemar p-value) |
| `agent_result.json` | Full `EvaluationResult` (agent) |
| `baseline_result.json` | Full `EvaluationResult` (vlm_only) — if provided |
| `comparison.json` | Full `ComparisonResult` — if provided |

**`report.md` sections:**

- **Executive Summary** — key numbers in prose (accuracy, abstention rate, net grounding improvement, significance)
- **Results** — metric comparison table (or standalone agent table if no baseline)
- **Closed-Ended Analysis** — confusion matrix, P/R/F1, yes-bias check
- **Open-Ended Analysis** — token F1, BLEU-1, BERTScore
- **Agent Behavior** — abstention rate, accuracy-when-answered, re-query rate, citation hit rate, grounding breakdown
- **Confidence Calibration** — ECE, AUROC, calibration bin table, threshold table with recommendation
- **Error Analysis** — wrong-and-confident samples, over-abstentions, improved vs degraded cases
- **Recommendations** — threshold recommendation from `threshold_analysis`, actionable next steps

---

## CLI Scripts

### `scripts/run_evaluation.py`

The primary entry point for running evaluations end-to-end.

```bash
# Quick test (20 samples, ~4 minutes on T4)
python scripts/run_evaluation.py --mode compare --max-samples 20 --no-bertscore

# Agent evaluation only
python scripts/run_evaluation.py --mode agent

# VLM-only baseline only
python scripts/run_evaluation.py --mode vlm_only

# Full: both modes + comparison + report
python scripts/run_evaluation.py --mode compare

# Resume after crash
python scripts/run_evaluation.py --mode agent --resume

# Custom dataset/split
python scripts/run_evaluation.py --mode compare --dataset pathvqa --split test
```

Results are saved to `{eval_output_dir}/` (default: `data/evaluation_reports/`):
- `agent_vqa_rad_test_{date}.json`
- `vlm_vqa_rad_test_{date}.json`
- `comparison_vqa_rad_test_{date}.json`
- `report.md` + `report.json`

### `scripts/generate_report.py`

Generate a report from previously saved JSON files (no GPU required):

```bash
python scripts/generate_report.py \
    --agent data/evaluation_reports/agent_vqa_rad_test_2026-02-23.json \
    --baseline data/evaluation_reports/vlm_vqa_rad_test_2026-02-23.json \
    --output-dir data/evaluation_reports
```

---

## Architecture Diagram

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        run_evaluation.py                             │
  │                                                                      │
  │  AgentEvaluator.evaluate(mode="agent")                               │
  │  ┌───────────────────────────────────────────────────────────────┐   │
  │  │  for each sample:                                             │   │
  │  │    AgentRunner.run_query() ──► SystemOutput                   │   │
  │  │         ┌────────────────────────────────┐                    │   │
  │  │         │  entry → visual → retrieval    │                    │   │
  │  │         │  → supervisor ──────────────►  │                    │   │
  │  │         │  (re_query loop, max 1 retry)  │                    │   │
  │  │         │  → output_formatter            │                    │   │
  │  │         └────────────────────────────────┘                    │   │
  │  │    PerSampleResult (+ intermediate save every 50)             │   │
  │  └───────────────────────────────────────────────────────────────┘   │
  │           ▼                                                          │
  │  compute_all_metrics()  ──► EvaluationResult  ──► agent_result.json  │
  │                                                                      │
  │  [del evaluator]  ← free VLM before loading second VLM              │
  │                                                                      │
  │  AgentEvaluator.evaluate(mode="vlm_only")                            │
  │  ┌───────────────────────────────────────────────────────────────┐   │
  │  │  for each sample:                                             │   │
  │  │    VLMBackend.predict() ──► VLMPrediction                     │   │
  │  └───────────────────────────────────────────────────────────────┘   │
  │           ▼                                                          │
  │  compute_all_metrics()  ──► EvaluationResult  ──► vlm_result.json    │
  │                                                                      │
  │  BaselineComparator.compare(agent, vlm)                              │
  │    align → deltas → grounding_improvement → McNemar → tables         │
  │           ▼                                                          │
  │  ComparisonResult  ──► comparison.json                               │
  │                                                                      │
  │  generate_report()  ──► report.md  +  report.json                    │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## Test Coverage

| Suite | Tests | Notes |
|-------|-------|-------|
| `tests/test_metrics.py` | Core VQA metrics | normalize_answer, exact_match, F1/BLEU/BERTScore |
| `tests/test_agent_metrics.py` | Agent behavior metrics | abstention, grounding, citation, correct_abstention |
| `tests/test_calibration.py` | Calibration functions | ECE, bins, discrimination, threshold_analysis |
| `tests/test_result.py` | Pydantic models | Save/load roundtrip, field validation, large per_sample |
| `tests/test_evaluator.py` | AgentEvaluator | Lazy init, crash recovery, resume, error handling |
| `tests/test_comparator.py` | BaselineComparator | Deltas, grounding sum invariant, McNemar edge cases |
| `tests/test_report.py` | generate_report | File creation, sections, threshold recommendation |
| **Phase 5A total** | **114 fast tests** | All pass |
| **Phase 5B total** | **50 fast tests** | All pass |
| **Phase 5 critical fixes** | **14 fast tests** | All pass |
| **Cumulative (all phases)** | **399 fast tests** | All pass |

All Phase 5 tests are fast (no GPU, no model downloads). BERTScore is mocked in tests that compute it.

---

## 20-Sample Pilot Results (Post-Fix)

Evaluated on 20 samples from the VQA-RAD test split after applying the Phase 5 critical fixes (agreement scoring + re-query).

### Decision Breakdown

| Decision | Before Fix | After Fix |
|----------|-----------|-----------|
| Answer | 11 (55%) | 13 (65%) |
| Abstain | 9 (45%) | 7 (35%) |

The plural normalisation fix (`"lungs"→"lung"`, `"kidneys"→"kidney"`) unblocked agreement scoring for 2 samples that were previously routing to Case B → abstain.

### Accuracy

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Overall accuracy (of 20) | 4/20 = 20.0% | 4/20 = 20.0% |
| Accuracy-when-answered | 4/11 = 36.4% | 4/13 = 30.8% |

Overall accuracy did not improve. Both recovered samples (test_5, test_15) were cases where the VLM answered "Yes" confidently but the ground truth was "No". The agreement function confirmed the evidence was topically relevant (it is about the right anatomy), but it cannot distinguish yes from no.

### Remaining 7 Abstentions (KG Coverage Gaps)

| Sample | Question | VLM Answer | Why Agreement = 0 |
|--------|----------|------------|-------------------|
| test_1 | Airspace consolidation on left side? | Yes | "airspace" and "consolidation" not in KG |
| test_4 | Where are the kidney? | In body | "body" not useful; KG has no location-answer entity |
| test_6 | Colon most prominent right or left? | Left | Open-ended: "Left" not in KG text |
| test_7 | Colon most prominent from view? | Top | Low VLM confidence (0.507 < 0.55) → Case E, correct abstain |
| test_8 | Heart size smaller or larger? | Smaller | Comparative question; "smaller" not in KG |
| test_11 | What structures are visible? | Teeth | Dental anatomy not in SLAKE KG |
| test_16 | What type of image is this? | X-ray | Imaging modality not in KG |

These are genuine KG coverage gaps and are not fixable by improving agreement scoring logic alone.

### The Fundamental Limitation Revealed

The pilot exposed a deeper architectural constraint: **agreement scoring checks topical relevance, not binary answer correctness.**

When the VLM is confidently wrong on a yes/no question — for example, predicting "Yes" to "is there aortic aneurysm?" when the ground truth is "No" — the supervisor routes to Case A (answer) if the KG returns documents about "aortic aneurysm". The documents are topically relevant, so agreement > 0. But the VLM's binary answer is wrong, and the agreement function has no way to detect this because it only checks whether the question's medical terms appear in evidence, not whether the evidence supports "yes" vs "no".

This affects 6 of the 13 answered samples in the pilot: the VLM was high-confidence and wrong, evidence was topically relevant, so the agent returned a confident wrong answer.

The only remedies are:
1. **Semantic agreement** (Phase 6): replace keyword matching with embedding similarity between the VLM answer and each evidence passage, which may catch cases where evidence contradicts the VLM
2. **VLM calibration**: a better-calibrated VLM would have lower confidence on cases it gets wrong, pushing them to Case D/E
3. **KG expansion**: adding image-specific facts (not just entity descriptions) so the retriever finds evidence that directly addresses the binary question

---

## Running on SageMaker

```bash
# 1. Pull latest code and install new dependency (scipy)
cd ~/radiology-vqa && git pull
pip install -e ".[dev]"

# 2. Verify fast tests (should show 399 passed, ~15 sec)
pytest tests/ -m "not slow" -q

# 3. Quick sanity check — 20 samples, ~4 minutes on T4
python scripts/run_evaluation.py --mode compare --max-samples 20 --no-bertscore

# 4. Full evaluation — agent + VLM-only + comparison + report (~3 hours on T4)
python scripts/run_evaluation.py --mode compare

# 5. Resume if interrupted
python scripts/run_evaluation.py --mode compare --resume

# 6. View report
cat data/evaluation_reports/report.md
```

Progress is printed to stdout every 25 samples (`{done}/{n} | {correct} correct | ~{lat}s/sample | ETA: {eta}min`). Intermediate results are saved every 50 samples so a crash loses at most 50 samples of work.

---

## Known Limitations

| Issue | Impact | Plan |
|-------|--------|------|
| Agreement checks topic, not binary answer | High-confidence wrong yes/no answers pass agreement (evidence is topically relevant even when VLM is wrong) — 6/13 answered samples in pilot were confidently wrong | Phase 6: semantic agreement using evidence-vs-answer embeddings |
| KG coverage gaps: modality, anatomy, spatial | 6 of 7 remaining abstentions are questions the KG cannot answer (imaging type, dental anatomy, comparative size, positional/spatial) | KG expansion with radiology-specific text corpus |
| Keyword matching misses synonyms | "opacity" won't match "consolidation"; "neoplasm" won't match "tumor" | Phase 6: embedding cosine similarity |
| McNemar's test requires discordant pairs | Needs ≥ 5 discordant pairs for meaningful p-value; small pilots return p=1.0 | Run ≥ 100 samples for significance testing |
| BERTScore loads a separate transformer model | ~1.5 GB additional GPU memory; OOM on T4 if VLM still loaded | Use `--no-bertscore` or free VLM before scoring |
| `retry_count` not surfaced in `SystemOutput` | `re_query_rate` always 0 in current `_run_agent()` | Expose `retry_count` from `AgentState` in a follow-up |
| `generate_report.py` computes comparison from file | Requires both agent + baseline JSON; comparison is optional | Pass `--agent` only to get agent-only report |
