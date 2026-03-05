# Grounded Multi-Agent Radiology VQA

A retrieval-augmented visual question answering system for radiology. The system combines a QLoRA fine-tuned LLaVA-Next 7B model with a hybrid knowledge retrieval pipeline and a deterministic supervisor that decides when to answer and when to abstain. **Phase 8A result: 50.3% overall accuracy, 71.7% closed accuracy, 44.9% citation hit rate, 7.1% abstention on VQA-RAD test (451 samples).** ECE 0.091 — well-calibrated confidence.

---

## Quick Start

```bash
# Single image inference
python scripts/quick_inference.py \
  --image path/to/xray.png \
  --question "Is there consolidation in the left lung?"

# Phase 8A defaults — no env vars needed (baked into config.py)
# adapter, isotonic calibration, hybrid retrieval, and indices_v3 are all defaults
python scripts/quick_inference.py \
  --image path/to/xray.png \
  --question "Is there consolidation in the left lung?"
```

---

## Architecture

```
Image + Question → Entry Node
                       │
                       ▼
              Visual Agent (LLaVA-Next 7B + QLoRA)
                       │  visual_answer + calibrated_confidence
                       ▼
           Retrieval Agent (Hybrid BM25 + FAISS, 13K docs)
                       │  top-5 evidence docs + citations
                       ▼
              Supervisor (deterministic, rule-based)
              PubMedBERT embedding agreement ≥ 0.87
              HIGH_CONF=0.50 / LOW_CONF=0.15  (closed)
              HIGH_CONF=0.50 / LOW_CONF=0.25  (open)
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
           answer   re_query  abstain
              │        │        │
              └────────┴────────┘
                       │
              Output Formatter
              grounded_answer + citations + reasoning
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map, data flow, and design decisions.

---

## Results

Evaluated on VQA-RAD test (451 samples). Full ablation in [EVALUATION.md](EVALUATION.md).

| Config | Overall Acc | Closed Acc | Open Acc | Abstain | Citation | ECE | AUROC |
|--------|------------|-----------|---------|---------|---------|-----|-------|
| 1. Zero-shot VLM | 41.5% | 61.4% | 16.5% | 0% | — | 0.434 | 0.769 |
| 3. Fine-tuned VLM | 50.8% | 71.7% | 24.5% | 0% | — | 0.351 | 0.751 |
| 5. FT + full agent (Phase 6) | 42.1% | 58.6% | 21.5% | 19.5% | 15.3% | 0.214 | 0.761 |
| 6. FT + agent + calibration (Phase 6C) | 35.5% | 50.6% | 16.5% | 41.0% | 15.3% | 0.075 | 0.868 |
| 7. Phase 7A — mixed calibrator | 42.1% | 58.6% | 21.5% | 21.5% | 15.3% | 0.081 | 0.793 |
| **8. Phase 8A — QA index v3 (FINAL)** | **50.3%** | **71.7%** | **23.5%** | **7.1%** | **44.9%** | **0.091** | **0.755** |

The Phase 8A system answers 92.9% of questions and is correct 50.3% overall (54.2% when it chooses to answer). At threshold 0.85, it answers 19.5% of questions with 88.6% accuracy — suitable for high-precision clinical use. AUROC 0.755 means confidence scores are a useful signal of correctness. See [EVALUATION.md](EVALUATION.md) for the full ablation and calibration analysis.

---

## Project Structure

```
radiology-vqa/
├── src/radiology_vqa/
│   ├── config.py              # All settings, env-var overridable
│   ├── schema.py              # Core data models
│   ├── loader.py              # VQA-RAD + PathVQA loaders
│   ├── slake_loader.py        # SLAKE loader
│   ├── rag/                   # Embedder, FAISS index, BM25, hybrid retriever
│   ├── vlm/                   # LLaVA backend + calibration integration
│   ├── agents/                # Visual, retrieval, supervisor, formatter nodes
│   ├── graph/                 # LangGraph wiring (builder, runner, routing)
│   ├── calibration/           # Platt scaling, isotonic regression
│   ├── training/              # Dataset builder, data collator
│   └── evaluation/            # Metrics, evaluator, comparator, report
├── scripts/
│   ├── quick_inference.py         # Single image/question inference
│   ├── run_evaluation.py          # Full evaluation pipeline
│   ├── generate_report.py         # Report from saved JSON results
│   ├── generate_ablation_report.py # 6-config ablation table
│   ├── fit_calibration.py         # Fit Platt/isotonic calibrators
│   ├── finetune_qlora.py          # QLoRA fine-tuning
│   └── build_index.py             # Build FAISS + BM25 index
├── configs/
│   ├── base.yaml                  # Phase 5 defaults
│   ├── phase6.yaml                # Phase 6B (hybrid retrieval, expanded KG)
│   ├── phase6_calibrated.yaml     # Phase 6C (isotonic calibration)
│   ├── config2_baseline_agent.yaml # Ablation: zero-shot + Phase 5 agent
│   └── config4_finetuned_agent.yaml # Ablation: FT + Phase 5 agent
├── tests/                         # 455 fast tests, no GPU required
├── docs/
│   ├── phase6_finetuning.md       # Complete Phase 6 documentation
│   ├── phase6c_calibration.md     # Phase 6C decisions and results
│   └── ...
├── ARCHITECTURE.md
├── EVALUATION.md
└── Makefile
```

---

## Setup

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Tested on 3.12.7 |
| torch | ≥2.0 | 2.2.2 on Mac (CPU); 2.7.1 on Colab; 2.10 on SageMaker |
| CUDA | ≥11.8 | Required for 4-bit quantization; CPU falls back to fp32 |
| sentence-transformers | ≥2.2 | S-PubMedBert-MS-MARCO (~440 MB on first run) |
| transformers | ≥4.37 | Key remapping in llava.py handles ≥4.45 naming changes |

### Install

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

### Datasets

VQA-RAD and PathVQA download automatically:
```bash
make download-data
```

SLAKE requires manual download from [med-vqa.com/slake](https://www.med-vqa.com/slake/). Place at:
```
data/raw/Slake1.0/
├── imgs/
├── train.json
├── validate.json
├── test.json
└── KG/
    ├── en_disease.csv
    ├── en_organ.csv
    └── en_organ_rel.csv
```

### Build Index

```bash
# Phase 5 index (SLAKE KG only, 2,987 docs) — used for ablation configs 2 and 4
make build-index-kg

# Phase 6B expanded index (SLAKE KG + RadLex + QA pseudo-docs, 13,435 docs)
python scripts/build_index.py \
  --sources kg radlex qa \
  --radlex-xls data/raw/radlex/Radlex.xls \
  --output-index-dir data/indices_v2
```

---

## Training

QLoRA fine-tuning on SageMaker ml.g5.2xlarge (A10G, 22 GB VRAM):

```bash
# Requires SLAKE, VQA-RAD, PathVQA datasets to be present
python scripts/finetune_qlora.py --config configs/training/qlora.yaml
```

Key hyperparameters (from `configs/training/qlora.yaml`):
- Base model: `llava-hf/llava-v1.6-mistral-7b-hf`
- Quantization: 4-bit NF4 with double quantization
- LoRA rank: 16, alpha: 32, dropout: 0.05
- Target modules: q_proj, v_proj
- Learning rate: 2e-4 (paged_adamw_8bit)
- Epochs: 3, batch size: 4 with gradient accumulation × 8
- Training time: ~23h on A10G

The adapter is saved to `checkpoints/llava-med-qlora/best/` with the processor.

### Calibration

```bash
# Fit calibrators on SLAKE validation (held-out during training)
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \
  python scripts/fit_calibration.py \
    --val-dataset slake \
    --val-split validation \
    --test-dataset vqa_rad \
    --test-split test \
    --output data/calibration/
```

---

## Evaluation

### Reproduce all 6 ablation configs

```bash
# Config 1 — zero-shot VLM only
python scripts/run_evaluation.py --mode vlm_only --dataset vqa_rad --split test --no-bertscore

# Config 2 — zero-shot + Phase 5 agent (keyword agreement, original index)
AGREEMENT_METHOD=keyword RETRIEVAL_METHOD=dense INDEX_DIR=data/indices \
CALIBRATION_METHOD=none SUPERVISOR_HIGH_CONFIDENCE=0.85 SUPERVISOR_LOW_CONFIDENCE=0.55 \
VLM_ADAPTER_PATH="" \
  python scripts/run_evaluation.py --mode agent --dataset vqa_rad --split test --no-bertscore \
    --output-dir data/evaluation_reports/config2_baseline_agent/

# Config 3 — fine-tuned VLM only
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \
  python scripts/run_evaluation.py --mode vlm_only --dataset vqa_rad --split test --no-bertscore

# Config 4 — fine-tuned + Phase 5 agent
AGREEMENT_METHOD=keyword RETRIEVAL_METHOD=dense INDEX_DIR=data/indices \
CALIBRATION_METHOD=none SUPERVISOR_HIGH_CONFIDENCE=0.85 SUPERVISOR_LOW_CONFIDENCE=0.55 \
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \
  python scripts/run_evaluation.py --mode agent --dataset vqa_rad --split test --no-bertscore \
    --output-dir data/evaluation_reports/config4_finetuned_agent/

# Config 5 — full pipeline (no calibration)
RETRIEVAL_METHOD=hybrid INDEX_DIR=data/indices_v2 \
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \
  python scripts/run_evaluation.py --mode agent --dataset vqa_rad --split test --no-bertscore

# Config 6 — full pipeline + isotonic calibration
RETRIEVAL_METHOD=hybrid INDEX_DIR=data/indices_v2 \
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \
CALIBRATION_METHOD=isotonic CALIBRATION_MODEL_PATH=data/calibration/isotonic_scaler.json \
SUPERVISOR_HIGH_CONFIDENCE=0.60 SUPERVISOR_LOW_CONFIDENCE=0.35 \
  python scripts/run_evaluation.py --mode agent --dataset vqa_rad --split test --no-bertscore
```

### Generate ablation report

```bash
python scripts/generate_ablation_report.py \
  --config1 data/evaluation_reports/vlm_vqa_rad_test_2026-02-27.json \
  --config3 data/evaluation_reports/vlm_vqa_rad_test_2026-02-27-2.json \
  --config5 data/evaluation_reports/agent_result-3.json \
  --config6 data/evaluation_reports/agent_result-4.json \
  --output data/evaluation_reports/ablation/
```

---

## Development

```bash
make test          # 455 fast tests, no GPU, runs in <10s
make test-slow     # Full suite including embedding model and real retrieval
make lint          # ruff check src/ tests/
make format        # ruff format src/ tests/
```

All test configuration flags default to Phase 5 behavior. Adding a new Phase 6 feature means: add the config field with a safe default, implement it, write fast tests with mocks, verify `make test` still passes.

All settings are in `src/radiology_vqa/config.py` and can be overridden via environment variables (uppercase field name) or a `.env` file:

```bash
# Key environment variables (Phase 8A production defaults — already set in config.py)
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best
RETRIEVAL_METHOD=hybrid           # "dense" (Phase 5) or "hybrid" (Phase 6B+)
INDEX_DIR=data/indices_v3         # "data/indices" (v1), "data/indices_v2" (v2), "data/indices_v3" (Phase 8A)
AGREEMENT_METHOD=embedding        # "keyword" (Phase 5) or "embedding" (Phase 6B-3)
CALIBRATION_METHOD=isotonic       # "none", "platt", or "isotonic"
CALIBRATION_MODEL_PATH=data/calibration/mixed/isotonic_scaler.json  # Phase 8A mixed calibrator
SUPERVISOR_HIGH_CONFIDENCE=0.50   # Phase 8A (was 0.85 in Phase 5, 0.60 in Phase 6C)
SUPERVISOR_LOW_CONFIDENCE=0.15    # Phase 8A (was 0.55 in Phase 5, 0.35 in Phase 6C)
```
