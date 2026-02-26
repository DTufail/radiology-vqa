# Phase 6: Comprehensive Implementation Roadmap (v2)

## Grounded Multi-Agent Radiology VQA System

**Date:** February 24, 2026
**Version:** 2.0 — Updated with PathVQA, system architecture, and professional practices
**Hardware:** AWS SageMaker ml.g4dn.xlarge, NVIDIA Tesla T4 (15GB VRAM), 100GB storage
**Current Baseline:** VLM-only 41.2%, Agent 32.8% (47.9% when answered, 73.2% correct abstention)
**Target:** 60-75% overall accuracy, <15% abstention rate, ECE < 0.10

---

## Table of Contents

1. System Architecture Overview
2. Pre-Flight: Storage Cleanup and Environment Setup
3. Phase 6A: QLoRA Fine-Tuning
4. Phase 6B: Knowledge Graph Expansion + Embedding Agreement
5. Phase 6C: Confidence Recalibration
6. Phase 6D: Final Evaluation and Portfolio Packaging
7. Software Engineering Practices
8. Timeline, Dependencies, and Risk Mitigation
9. References

---

## 1. System Architecture Overview

### 1.1 Current Architecture (Post Phase 5)

```
USER INPUT: Image + Question
         |
         v
+--------------------------------------------------+
|       LLaVA v1.6 Mistral 7B (4-bit)             |
|       [Zero-Shot - No Medical Training]           |
|                                                   |
|  Prompt: <image>\n{question}\nAnswer in 1-3 words |
|  Output: visual_answer + confidence_score         |
+---------------------+----------------------------+
                      |
         +------------+------------+
         v                         v
+----------------+    +-----------------------------------+
| VLM-Only Path  |    |        RAG Pipeline               |
| (Baseline)     |    |                                   |
|                |    | Query: question + visual_answer    |
| Return answer  |    |        |                          |
| directly       |    |        v                          |
|                |    | PubMedBert -> FAISS Index          |
|                |    | (2,987 SLAKE KG docs)              |
|                |    |        |                          |
|                |    |        v                          |
|                |    | Top-K Evidence Documents           |
|                |    +----------------+------------------+
|                |                     |
|                |                     v
|                |    +-----------------------------------+
|                |    |      SUPERVISOR NODE               |
|                |    |                                   |
|                |    | Agreement: keyword overlap         |
|                |    | (Strategy A: open questions)       |
|                |    | (Strategy B: closed questions)     |
|                |    |                                   |
|                |    | Case A: High agreement -> Answer   |
|                |    | Case B: Low agreement -> Re-query  |
|                |    | Case C: Still low -> ABSTAIN       |
|                |    +----------------+------------------+
|                |                     |
+--------+-------+                     |
         v                             v
+--------------------------------------------------+
|                    OUTPUT                         |
| Answer + Confidence + Citations (or ABSTAIN)      |
+--------------------------------------------------+
```

### 1.2 Target Architecture (Post Phase 6)

Phase 6 modifies three specific points in the pipeline. Module boundaries and
data contracts remain stable:

```
USER INPUT: Image + Question
         |
         v
+--------------------------------------------------+
|   LLaVA v1.6 Mistral 7B (4-bit + QLoRA)  <-[6A] |
|   [Fine-tuned on VQA-RAD + SLAKE + PathVQA]      |
|                                                   |
|   LoRA Adapter: rank 16, alpha=32                 |
|   Training: ~27K QA pairs, 3 epochs               |
|   Expected: 41% -> 60-75% accuracy                |
+---------------------+----------------------------+
                      |
         +------------+------------+
         v                         v
+----------------+    +-----------------------------------+
| VLM-Only Path  |    |   Expanded RAG Pipeline    <-[6B] |
| (Fine-tuned    |    |                                   |
|  Baseline)     |    | PubMedBert -> FAISS Index          |
|                |    | (~35K docs: SLAKE KG + RadLex      |
|                |    |  + QA pseudo-docs)                  |
|                |    +----------------+------------------+
|                |                     |
|                |                     v
|                |    +-----------------------------------+
|                |    |  UPGRADED SUPERVISOR         <-[6B]|
|                |    |                                   |
|                |    | Agreement: EMBEDDING COSINE SIM    |
|                |    | (PubMedBert sentence embeddings)   |
|                |    |                                   |
|                |    | >= 0.7: Strong -> Case A (high)    |
|                |    | 0.5-0.7: Moderate -> Case A (mod)  |
|                |    | < 0.5: Weak -> Case B/Abstain      |
|                |    +----------------+------------------+
|                |                     |
+--------+-------+                     |
         v                             v
+--------------------------------------------------+
|        TEMPERATURE SCALING               <-[6C]  |
|                                                   |
| Learned parameter T on validation set             |
| calibrated_conf = softmax(logits / T)             |
| Expected: ECE 0.198 -> < 0.10                     |
+---------------------+----------------------------+
                      |
                      v
+--------------------------------------------------+
|                    OUTPUT                         |
| Answer + Calibrated Confidence + Citations        |
| OR: ABSTAIN + Reason + Suggested Review           |
+--------------------------------------------------+
```

### 1.3 Key Architectural Principles

**Principle 1 - Interface Stability:** Phase 6 changes internal implementations but
NOT the interfaces between modules. VLMPrediction, AgentState, SupervisorDecision
schemas stay unchanged. The evaluation framework (337 tests) continues working.

**Principle 2 - Configuration-Driven Behavior:** Every Phase 6 change is controlled
by a config flag that defaults to Phase 5 behavior:

```yaml
# config/base.yaml - Phase 5 (backward compatible)
vlm:
  model_id: "llava-hf/llava-v1.6-mistral-7b-hf"
  adapter_path: null
  temperature: 1.0
knowledge:
  index_path: "data/faiss_index"
  agreement_method: "keyword"

# config/phase6.yaml - Full Phase 6
vlm:
  model_id: "llava-hf/llava-v1.6-mistral-7b-hf"
  adapter_path: "checkpoints/llava-med-qlora/best"
  temperature: 1.42
knowledge:
  index_path: "data/faiss_index_v2"
  agreement_method: "embedding"
  agreement_threshold: 0.5
```

**Principle 3 - Reproducibility:** Every experiment is deterministic through config
files, random seeds, and logged hyperparameters.

### 1.4 Module Map

```
src/radiology_vqa/
|-- data/                      # Phase 1 - Dataset loaders
|   |-- vqa_rad.py
|   |-- slake.py
|   +-- pathvqa.py             # [EXISTS]
|
|-- knowledge/                 # Phase 2 - RAG pipeline
|   |-- encoder.py             # PubMedBert embedding
|   |-- faiss_index.py         # FAISS vector store
|   |-- knowledge_graph.py     # KG document store
|   +-- radlex.py              # [NEW 6B] RadLex ontology parser
|
|-- vlm/                       # Phase 3 - VLM backend
|   |-- interface.py           # VLMInterface ABC, VLMPrediction schema
|   +-- llava.py               # LlavaBackend (modified in 6A)
|
|-- agents/                    # Phase 4 - Multi-agent nodes
|   |-- state.py               # AgentState schema
|   |-- visual_qa.py
|   |-- retrieval_agent.py
|   |-- supervisor.py          # (modified in 6B)
|   +-- graph.py               # LangGraph wiring
|
|-- training/                  # [NEW 6A]
|   |-- dataset.py             # MedVQADataset (3 datasets unified)
|   |-- collator.py            # LlavaDataCollator
|   +-- callbacks.py           # Training callbacks
|
|-- calibration/               # [NEW 6C]
|   +-- temperature.py         # TemperatureScaling
|
+-- evaluation/                # Phase 5 - Evaluation framework
    |-- metrics.py
    |-- evaluator.py
    |-- comparator.py
    +-- report.py
```

### 1.5 Data Flow Contracts (Unchanged by Phase 6)

```
VLMInterface.predict(image, question) -> VLMPrediction
    Fields: answer: str, confidence: float, raw_output: str

KnowledgeBase.retrieve(query, top_k) -> List[Document]
    Fields: doc_id: str, content: str, score: float

Supervisor.route(visual_answer, evidence, confidence, answer_type) -> SupervisorDecision
    Fields: action: str, answer: str, confidence: float, citations: List[str], reason: str
```

---

## 2. Pre-Flight: Storage Cleanup and Environment Setup

### 2.1 Storage Cleanup (~38GB recoverable)

```bash
# Remove unused model caches
rm -rf ~/.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b        # ~14GB
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b  # ~14GB
rm -rf ~/.cache/huggingface/hub/models--microsoft--deberta-xlarge-mnli    # ~3GB
pip cache purge                                                           # ~7GB

# KEEP: models--llava-hf--llava-v1.6-mistral-7b-hf (~14GB)
# KEEP: datasets/flaviagiammarino___path-vqa (~4GB)
```

### 2.2 Install Dependencies

```bash
pip install peft trl --break-system-packages

python -c "
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import SFTTrainer
print('All imports successful')
"
```

### 2.3 Storage Budget

| Item | Size | Notes |
|------|------|-------|
| Base model cache | ~14 GB | Already cached, read-only |
| PathVQA cache | ~4 GB | Already cached, training data |
| QLoRA checkpoints | ~0.2 GB | Adapters are only ~50MB each |
| Expanded FAISS index | ~0.1 GB | 35K docs x 768-dim |
| **Total new storage** | **~0.4 GB** | QLoRA is very storage-efficient |

---

## 3. Phase 6A: QLoRA Fine-Tuning

### 3.1 Research Foundation

Every top-performing Med-VQA system trains on VQA-RAD + SLAKE + PathVQA together:

- PeFoMed (arXiv:2401.02797) used all three datasets for stage 2 fine-tuning
- BioMedBLIP (JMIR 2024) fine-tuned across all three, achieving SOTA on 15/20 tasks
- BaMCo (Springer 2025) trained jointly: 85.8% SLAKE, 76.7% VQA-RAD, 60.0% PathVQA
- CMMO (J. Biomed. Informatics 2024) pre-trained across all three: 79.6%, 65.7%, 87.2%

Kandamali et al. (Applied Soft Computing 2025) confirmed 90-93% accuracy with LoRA/AdaLoRA
on hardware with 15GB VRAM - exactly matching our Tesla T4.

### 3.2 Training Data (Including PathVQA)

| Dataset | Split | QA Pairs | Images | Image Type | License |
|---------|-------|----------|--------|------------|---------|
| VQA-RAD | Train | 3,064 | 315 | X-ray, CT, MRI | Research |
| SLAKE | Train (EN) | 4,918 | 642 | X-ray, CT, MRI | CC BY 4.0 |
| PathVQA | Train | 19,654 | 2,599 | Pathology slides | MIT |
| **Total** | | **27,636** | **3,556** | **Mixed medical** | |

**Validation:** 10% of VQA-RAD train (~306). SLAKE val (1,053) as secondary.
**Test:** VQA-RAD test (451 samples) NEVER TOUCHED during training.

**Why PathVQA despite being pathology (not radiology):**
Our vision encoder is FROZEN during training. Only the LM's LoRA adapters train.
PathVQA's 19K samples teach the LM to produce concise medical answers, handle
yes/no structure, and map medical terminology -- all skills that transfer to
radiology. With 27K samples (vs 7K), overfitting risk drops significantly.

PathVQA uses identical structure to our other datasets (image, question, answer
fields on HuggingFace) so integration is 3 lines of code -- zero complexity added.

### 6A-1: Data Preparation (Half day)

**New files:** src/radiology_vqa/training/dataset.py, collator.py

Key implementation details:

1. **MedVQADataset class:** Loads all 3 datasets, normalizes answers (lowercase,
   strip articles/punctuation), formats as LLaVA conversations, creates train/val split

2. **Answer normalization:**
   - "The Lungs." -> "lungs"
   - "A CT scan" -> "ct scan"
   - "Yes." -> "yes"

3. **LlavaDataCollator:** Applies chat template, processes images through LLaVA
   processor, masks padding tokens with -100 in labels, max_length=256

4. **CRITICAL test:** Verify zero overlap between training questions and
   VQA-RAD test set (data leakage prevention)

### 6A-2: Training Script (1-2 days)

**New file:** scripts/finetune_qlora.py

**Model loading:** 4-bit NF4 with double quantization via BitsAndBytesConfig

**LoRA config (community-verified for LLaVA v1.6):**
- rank=16, alpha=32, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Vision encoder and multi_modal_projector EXCLUDED (frozen)
- Expected trainable: ~13M / ~7B total (0.18%)

**Training args:**
- batch_size=1, gradient_accumulation=8 (effective batch=8)
- lr=2e-4, cosine schedule, warmup 3%
- gradient_checkpointing=True, paged_adamw_8bit optimizer
- fp16=True, max_grad_norm=0.3
- 3 epochs, save best 2 checkpoints by val loss

**VRAM budget:** ~9-11 GB (T4 has 15 GB, ~4-6 GB safety margin)

### 6A-3: Training Run (~10 hours overnight)

- 27,636 samples x 3 epochs = 82,908 forward passes
- / 8 gradient accumulation = ~10,364 optimizer steps
- @ ~3-4 sec/step on T4 = ~9-11 hours

Run in tmux/screen, monitor with `tail -f` and `nvidia-smi`.

### 6A-4: Pipeline Integration (Half day)

**Modified file:** src/radiology_vqa/vlm/llava.py

Single conditional block in LlavaBackend.__init__():
- If adapter_path configured: load with PeftModel.from_pretrained(), merge_and_unload()
- If not configured (or adapter missing): fall back to zero-shot with warning
- All other code unchanged (predict(), confidence extraction, prompting)

### 6A-5: Evaluation (3 configs, ~5 hours)

| Config | VLM | LoRA | Agent |
|--------|-----|------|-------|
| A | Zero-shot | No | No | Existing baseline (already done) |
| B | Fine-tuned | Yes | No | Isolate fine-tuning impact |
| C | Fine-tuned | Yes | Yes | Full 6A pipeline |

Key hypothesis: fine-tuned VLM produces medical terms (not "Tumor" for everything),
leading to better evidence agreement and lower over-abstention.

### 6A-6: Hyperparameter Tuning (1 day if needed)

| Problem | Fix |
|---------|-----|
| Val loss diverges after epoch 1 | Reduce epochs to 2, dropout to 0.10 |
| Large train-val gap | Reduce rank to 8 |
| Loss still dropping at epoch 3 | Increase to 5 epochs |
| Poor open-ended answers | Increase rank to 32, add lm_head to modules_to_save |

**Fallback:** LLaMA-Factory (ACL 2024) handles multimodal QLoRA automatically.

---

## 4. Phase 6B: Knowledge Graph Expansion + Embedding Agreement

**Dependency:** Complete AFTER Phase 6A (fine-tuned VLM changes agreement dynamics).

### 6B-1: RadLex Ontology Integration (1 day)

**New file:** src/radiology_vqa/knowledge/radlex.py

RadLex (RSNA) contains 46,000+ classes covering radiology anatomy, findings,
modalities, procedures. Free license. Available in OWL/Excel/FHIR from radlex.org.

Strategy: Download Excel format, parse relevant categories (imaging modalities,
anatomical regions, radiological findings, imaging planes), convert to documents,
embed with PubMedBert, add to FAISS index.

### 6B-2: QA Pseudo-Documents (Half day)

Convert ~27K training QA pairs into retrievable pseudo-documents:
"Q: {question} A: {answer}" format. Only training set (never test) to prevent leakage.

### Expected Coverage After Expansion

| Source | Documents |
|--------|-----------|
| SLAKE KG (existing) | 2,987 |
| RadLex terms | ~5,000-8,000 |
| QA pseudo-docs | ~27,636 |
| **Total** | **~35K-38K** |

### 6B-3: Embedding-Based Agreement (1 day)

**Modified:** src/radiology_vqa/agents/supervisor.py

Replace keyword overlap with cosine similarity using PubMedBERT Embeddings
(NeuML/pubmedbert-base-embeddings, 768-dim, ~440MB). Handles synonyms
(tumor/neoplasm), paraphrases (enlarged heart/cardiomegaly), related concepts.

Thresholds: >=0.7 strong, 0.5-0.7 moderate, <0.5 weak.
Tune on validation set. Keep keyword matching as configurable fallback.

Known limitation: biomedical embeddings can assign high similarity to negation
pairs. Combine with VLM confidence, not sole gating criterion.

---

## 5. Phase 6C: Confidence Recalibration

**Dependency:** AFTER Phase 6A+6B. Calibrate the final system.

### 6C-1: Temperature Scaling (2-4 hours)

**New file:** src/radiology_vqa/calibration/temperature.py

Based on Guo et al. (ICML 2017): divide logits by learned scalar T before softmax.
Single parameter fitted on validation set via L-BFGS to minimize NLL.
One demonstration showed ECE reduction from 2.10% to 0.25%.

Implementation: ~30 lines of PyTorch. Fit on held-out 306 VQA-RAD validation
samples. Integrate as config option (temperature: 1.42 or whatever is learned).

### 6C-2: Bin-wise Calibration (if needed, 2-4 hours)

If single T is insufficient, learn per-bin temperatures for different confidence ranges.

---

## 6. Phase 6D: Final Evaluation and Portfolio Packaging

### 6D-1: 6-Configuration Comprehensive Evaluation

| # | Config | VLM | LoRA | RAG | KG | Agreement | Calibration |
|---|--------|-----|------|-----|-----|-----------|-------------|
| 1 | baseline_vlm | Zero-shot | No | No | - | - | - |
| 2 | baseline_agent | Zero-shot | No | Yes | Original | Keyword | - |
| 3 | finetuned_vlm | Fine-tuned | Yes | No | - | - | - |
| 4 | finetuned_agent | Fine-tuned | Yes | Yes | Original | Keyword | - |
| 5 | full_pipeline | Fine-tuned | Yes | Yes | Expanded | Embedding | - |
| 6 | full_calibrated | Fine-tuned | Yes | Yes | Expanded | Embedding | Temp |

Clean ablation: fine-tuning (3v1), RAG with FT (4v3), KG expansion (5v4),
calibration (6v5), full vs baseline (6v1).

### 6D-2: Selective Prediction Narrative

A January 2026 paper (arXiv:2601.01008) proposed an agentic AI framework for
uncertainty-aware abstention in stroke imaging, noting most systems lack structured
mechanisms to abstain under ambiguous conditions.

Our target metrics:

| Metric | Pre Phase 6 | Target Post Phase 6 |
|--------|-------------|---------------------|
| Accuracy (when answered) | 47.9% | >=80% |
| Correct abstention rate | 73.2% | >=85% |
| ECE | 0.198 | <0.10 |
| Abstention rate | 31.5% | 10-15% |

### 6D-3: Final Repository Structure

```
radiology-vqa/
|-- README.md
|-- ARCHITECTURE.md
|-- EVALUATION.md
|-- Makefile
|-- pyproject.toml
|-- requirements.txt
|-- configs/
|   |-- base.yaml
|   |-- phase6_vlm_only.yaml
|   |-- phase6_agent.yaml
|   |-- phase6_full.yaml
|   +-- training/qlora.yaml
|-- src/radiology_vqa/
|   |-- data/
|   |-- knowledge/
|   |-- vlm/
|   |-- agents/
|   |-- training/
|   |-- calibration/
|   +-- evaluation/
|-- scripts/
|   |-- finetune_qlora.py
|   |-- evaluate.py
|   |-- compare.py
|   |-- calibrate.py
|   +-- build_index.py
|-- tests/ (337+ tests)
|-- checkpoints/
|-- data/
|-- logs/
+-- docs/
```

---

## 7. Software Engineering Practices

### 7.1 Version Control

```
main                         # Stable, tested code
|-- feature/6a-training      # QLoRA fine-tuning
|-- feature/6b-kg            # KG expansion + embedding agreement
|-- feature/6c-calibration   # Temperature scaling
+-- feature/6d-packaging     # Final eval + cleanup
```

Each branch: includes new tests, does NOT break existing 337 tests,
merges to main via squash commit.

### 7.2 Testing Strategy

New tests for Phase 6:
- test_normalize_answer(): verify answer cleaning
- test_dataset_sizes(): verify expected sample counts (~27K train)
- test_conversation_format(): verify LLaVA structure
- test_no_test_leakage(): CRITICAL - zero overlap with VQA-RAD test
- test_batch_shapes(): verify collator output tensors
- test_label_masking(): verify -100 padding mask

### 7.3 Experiment Tracking (Lightweight)

JSON metadata for each run: experiment_id, timestamp, config dict, results
(train/val loss, accuracy, training time, peak GPU memory), hardware info.
No W&B/MLflow overhead.

### 7.4 Configuration Management

All configs in YAML files, never hardcoded. Config files checked into git.
Example training config:
- model.id, model.quantization
- lora.rank, lora.alpha, lora.dropout, lora.target_modules
- training.epochs, training.batch_size, training.learning_rate
- data.include_vqa_rad, data.include_slake, data.include_pathvqa, data.seed

### 7.5 Error Handling

Graceful degradation: if adapter_path not found, log warning and fall back to
zero-shot. If RadLex parsing fails, continue with existing KG. If temperature
scaling makes ECE worse, revert to T=1.0.

### 7.6 Documentation Standards

Every new module: module docstring, class docstrings with data contracts,
method docstrings (args/returns/raises), type hints on all public methods.

---

## 8. Timeline, Dependencies, and Risk Mitigation

### 8.1 Timeline (10 days)

```
Day 0:   Pre-flight (cleanup + install)
Day 1:   6A-1 Data preparation + tests
Day 2:   6A-2 Training script + dry run
Day 2-3: 6A-3 Training overnight (~10 hours)
Day 3:   6A-4 Integration + 20-sample validation
Day 4:   6A-5 Full evaluation (3 configs) + 6A-6 tuning if needed
Day 5:   6B-1 RadLex + 6B-2 QA pseudo-docs
Day 6:   6B-3 Embedding agreement + re-evaluate
Day 7:   6C-1 Temperature scaling + 6C-2 if needed
Day 8:   6D-1 Full 6-config evaluation
Day 9:   6D-2 Repo cleanup + README + architecture diagram
Day 10:  Buffer / polish
```

### 8.2 Critical Path

Pre-flight -> 6A-1 -> 6A-2 -> 6A-3 (overnight) -> 6A-4 -> 6A-5
Then: 6B-1 -> 6B-2 -> 6B-3 -> 6C-1 -> 6D-1 -> 6D-2

### 8.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CUDA OOM | Reduce max_length to 128, rank to 8, verify gradient checkpointing |
| Overfitting | Monitor val loss, early stopping, increase dropout |
| Collator crash | Fallback: LLaMA-Factory |
| Fine-tuned worse | 20-sample pilot first; check data format |
| PathVQA hurts radiology | Ablation: train with/without, compare on VQA-RAD test |
| RadLex parsing issues | Start with Excel format, only terms with definitions |
| Embedding agreement too lax | Tune threshold on val set, keep keyword fallback |

---

## 9. References

### Fine-Tuning
1. Kandamali et al. "LoRA and AdaLoRA on VQA-RAD/SLAKE." Applied Soft Computing, 2025
2. PeFoMed (arXiv:2401.02797). All 3 datasets, 87.1% closed on VQA-RAD
3. LLaMA32-Med (PMLR 2026). 84.6% SLAKE, +48 pts over zero-shot
4. BioMedBLIP (JMIR 2024). SOTA on 15/20 tasks across 3 datasets
5. BaMCo (Springer 2025). 85.8% SLAKE, 76.7% VQA-RAD, 60.0% PathVQA
6. CMMO (J. Biomed. Informatics 2024). 79.6%, 65.7%, 87.2%
7. HuggingFace LLaVA v1.6 community LoRA config
8. Philschmid "Fine-tune LLMs 2025": rank 16, lr 2e-4
9. LLaMA-Factory (ACL 2024): fallback option

### Knowledge Graph
10. RadLex, RSNA (radlex.org): 46K+ classes
11. Chepelev et al. RadioGraphics 2023

### Embeddings
12. NeuML/pubmedbert-base-embeddings (HuggingFace)
13. Chen et al. BMC Bioinformatics 2019
14. Bouscarrat et al. arXiv:2401.01943, 2024

### Calibration
15. Guo et al. "On Calibration of Modern NNs." ICML 2017
16. Minderer et al. NeurIPS 2021
17. gpleiss/temperature_scaling (GitHub)

### Clinical AI Safety
18. arXiv:2601.01008 (Jan 2026): agentic abstention in stroke imaging
19. SelectLLM (OpenReview, Oct 2025): selective prediction for medical QA

### Datasets
20. Lau et al. VQA-RAD. Scientific Data, 2018
21. Liu et al. SLAKE. ISBI 2021
22. He et al. PathVQA. ACL 2021. MIT License
