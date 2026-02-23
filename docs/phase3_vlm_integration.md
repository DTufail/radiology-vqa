# Phase 3 — VLM Integration

## Goal

Connect vision-language models (VLMs) to the existing RAG pipeline as the primary visual inference engine. The goal was to build a backend-agnostic VLM layer, run the first end-to-end inference on VQA-RAD, and establish a benchmarking baseline before the multi-agent pipeline was introduced in Phase 4.

---

## What We Built

### 1. `VLMInterface` Protocol (`src/radiology_vqa/vlm/interface.py`)

A `@runtime_checkable` Protocol that all backends must implement:

```python
class VLMInterface(Protocol):
    @property
    def model_name(self) -> str: ...
    def predict(self, image: PIL.Image.Image, question: str) -> VLMPrediction: ...
    def predict_batch(self, samples: list[tuple], batch_size: int = 1) -> list[VLMPrediction]: ...
```

`VLMPrediction` is a Pydantic model with fields: `answer` (min_length=1), `confidence` (0.0–1.0), `raw_output`, `model_name`, `latency_seconds`.

### 2. LLaVA v1.6 Mistral-7B Backend (`src/radiology_vqa/vlm/llava_med.py`)

- Model: `llava-hf/llava-v1.6-mistral-7b-hf`
- Automatic weight key remapping to handle `transformers>=4.45` naming changes (flat format → nested `text_config.*` format, 686/689 parameters remapped)
- Three-tier processor loading strategy: AutoProcessor → LlavaProcessor → manual construction fallback
- 4-bit / 8-bit quantization via `bitsandbytes` (CUDA only; CPU falls back to fp32 automatically)
- Shard-by-shard streaming weight loading for T4 VRAM efficiency (~4.35 GB GPU footprint at 4-bit)
- Confidence computed from output token logits (softmax + gather mean over generated tokens)
- **Bug fixed during development**: `AlignDevicesHook.weights_map` was becoming stale after manual `set_module_tensor_to_device`, causing garbage outputs with confidence≈0.0. Fixed by reversing load order: `init_empty_weights → infer_auto_device_map → materialize weights → dispatch_model` (so hooks attach after weights are loaded).

### 3. BLIP-2 OPT-2.7B Backend (`src/radiology_vqa/vlm/blip2.py`)

- Model: `Salesforce/blip2-opt-2.7b`
- True batched inference with configurable `batch_size`
- 8-bit quantization (CUDA only)
- Default confidence = 0.5 (OPT decoder architecture; logits extraction not straightforward for batched sampling)
- Lighter and faster than LLaVA; used as a comparison baseline

### 4. Factory (`src/radiology_vqa/vlm/factory.py`)

```python
create_vlm_backend(config: Settings) -> VLMInterface
```

Routes `config.vlm_backend` (`"llava"` / `"llava_med"` / `"blip2"`) to the appropriate class. Auto-redirects legacy LLaVA-Med v1.5 checkpoint IDs to the v1.6 model.

### 5. Benchmark Suite (`src/radiology_vqa/benchmark/`)

- **`metrics.py`**: `normalize_answer()` (strip, lowercase, remove punctuation), `is_match()` (exact match post-normalisation), `compute_metrics()` (overall / closed / open accuracy + counts)
- **`runner.py`**: `BenchmarkRunner` is backend-agnostic — depends only on `VLMInterface`. Saves `BenchmarkResult` as JSON to `data/benchmarks/`. Reports mean/median latency and samples/sec.

### 6. CLI Scripts

- `scripts/run_benchmark.py` — flags: `--dataset`, `--split`, `--max-samples`, `--backend`, `--compare`
- `scripts/quick_inference.py` — single image or dataset sample, flags: `--dataset`, `--index`, `--image`, `--question`, `--backend`

---

## Architecture

```
PIL.Image + question
        │
        ▼
  VLMInterface.predict()
        │
        ▼
  VLMPrediction
   ├── answer
   ├── confidence  (from logits for LLaVA, default 0.5 for BLIP-2)
   ├── raw_output
   ├── model_name
   └── latency_seconds
        │
        ▼
  BenchmarkRunner
   ├── accumulate per-sample results
   ├── compute_metrics()  →  overall / closed / open accuracy
   └── save BenchmarkResult JSON
```

---

## Test Coverage

| Suite | Fast tests | Status |
|-------|-----------|--------|
| VLMInterface / VLMPrediction validation | included | Pass |
| LLaVA backend (MockVLMBackend, quantize, batch) | included | Pass |
| BLIP-2 backend (batch, CPU fallback) | included | Pass |
| Benchmark metrics (normalize, is_match, compute_metrics) | included | Pass |
| Benchmark runner (flow, JSON output) | included | Pass |
| **Total Phase 3 fast tests** | **81 / 81** | **All pass** |

Slow tests (`@pytest.mark.slow`) require HuggingFace model downloads and are skipped in CI.

---

## Benchmark Results

All runs on **SageMaker `ml.g4dn.xlarge`** — T4 GPU (15 GB VRAM), evaluated on VQA-RAD test split (451 samples).

### Run 1 — BLIP-2 OPT-2.7B (4-bit)

- **Date**: 2026-02-21
- **Result file**: `blip2-opt-2.7b-4bit_vqa_rad_test_2026-02-21T16-51-27.json`

| Metric | Value |
|--------|-------|
| Overall accuracy | **25.06%** (113 / 451) |
| Closed-ended accuracy | **41.83%** (105 / 251) |
| Open-ended accuracy | **4.00%** (8 / 200) |
| Mean latency | 0.772 s / sample |
| Median latency | 0.614 s / sample |
| Throughput | ~1.3 samples / sec |
| Total runtime | 348 s (~5.8 min) |

BLIP-2 answers closed yes/no questions at above-chance accuracy but almost completely fails on open-ended questions that require descriptive answers. Short generic outputs ("yes"/"no") match ground truth by chance; descriptive outputs do not.

---

### Run 2 — LLaVA v1.6 Mistral-7B (4-bit) — broken run

- **Date**: 2026-02-23 03:52
- **Result file**: `llava-v1.6-mistral-7b-4bit_vqa_rad_test_2026-02-23T03-52-33.json`

| Metric | Value |
|--------|-------|
| Overall accuracy | **0%** (0 / 451) |
| Closed-ended accuracy | 0% |
| Open-ended accuracy | 0% |
| Mean latency | 21.0 s / sample |
| Total runtime | 9,490 s (~2.6 hours) |

The model loaded without error (686/689 parameters applied, 4.35 GB GPU) but all outputs were garbage. Root cause: `AlignDevicesHook.weights_map` was initialized from random/meta tensors before the checkpoint was loaded, so every forward call overwrote the correct weights with random values. Confidence was always ≈0.0. See `docs/llava_med_loading_fixes.md` for the full fix.

---

### Run 3 — LLaVA v1.6 Mistral-7B (4-bit) — after fix

- **Date**: 2026-02-23 05:23
- **Result file**: `llava-v1.6-mistral-7b-4bit_vqa_rad_test_2026-02-23T05-23-13.json`

| Metric | Value |
|--------|-------|
| Overall accuracy | **41.24%** (186 / 451) |
| Closed-ended accuracy | **61.35%** (154 / 251) |
| Open-ended accuracy | **16.00%** (32 / 200) |
| Mean latency | 10.38 s / sample |
| Median latency | 10.49 s / sample |
| Throughput | ~0.1 samples / sec |
| Total runtime | 4,682 s (~1.3 hours) |

After the weight-loading fix LLaVA substantially outperforms BLIP-2: +16 pp overall, +20 pp on closed, +12 pp on open. Open-ended accuracy is still low because LLaVA generates verbose clinical descriptions that don't match the short ground-truth tokens expected by exact-match evaluation.

---

## Model Comparison Summary

| Model | Overall | Closed | Open | Latency | Notes |
|-------|---------|--------|------|---------|-------|
| BLIP-2 OPT-2.7B (4-bit) | 25.1% | 41.8% | 4.0% | 0.77 s | Fast; weak on open QA |
| LLaVA v1.6 Mistral-7B (4-bit) — broken | 0.0% | 0.0% | 0.0% | 21 s | AlignDevicesHook bug |
| **LLaVA v1.6 Mistral-7B (4-bit) — fixed** | **41.2%** | **61.4%** | **16.0%** | 10.4 s | Best standalone VLM |

LLaVA v1.6 is the primary backend from Phase 4 onward. Its lower open-ended accuracy is mitigated by the Phase 4 supervisor, which can abstain rather than emit a wrong answer.

---

## Known Limitations (addressed in Phase 4)

| Limitation | Resolution |
|------------|-----------|
| LLaVA verbose outputs don't match short ground-truth tokens | Phase 4 supervisor selects first token / short phrase |
| No grounding of VLM answer in medical knowledge | Phase 4 RAG retrieval + agreement scoring |
| High-confidence wrong answers slip through (e.g., brain atrophy) | Phase 4 abstain mechanism |
| Quantization only on CUDA; CPU inference is slow | Documented; SageMaker used for production runs |
