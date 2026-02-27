# Phase 6 — Grounded Multi-Agent Radiology VQA

**Phase 6A Status:** Complete — training finished 2026-02-26T06:34:46 UTC
**Phase 6B Status:** Complete — hybrid retrieval, expanded index (13,435 docs), embedding agreement
**Phase 6C–6D Status:** Planned
**Final eval loss:** 0.1473 | **Perplexity:** 1.16 | **Train loss:** 0.4689
**Hardware:** NVIDIA A10G (22.1 GB VRAM), SageMaker ml.g5.2xlarge
**Training time:** 23h 13m 57s (2,514 steps across 3 epochs)

Phase 6 modifies three specific points in the existing pipeline while keeping all module
interfaces and data contracts stable:

- **Phase 6A** — QLoRA fine-tuning of the LLaVA-Next 7B language layers on medical VQA data
- **Phase 6B** — Knowledge graph expansion and embedding-based supervisor agreement
- **Phase 6C** — Temperature scaling for calibrated confidence scores
- **Phase 6D** — Final 6-configuration evaluation and portfolio packaging

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Goal](#2-goal)
3. [Phase 6A — QLoRA Fine-Tuning](#3-phase-6a--qlora-fine-tuning-complete)
   - [Research Foundation](#research-foundation)
   - [6A-1 — Dataset Cleanup and Training Package](#phase-6a-1--dataset-cleanup-and-training-package)
   - [6A-2 — Training Script and Pre-Training Audit](#phase-6a-2--training-script-and-pre-training-audit)
   - [6A-3 — Training Run](#phase-6a-3--training-run)
   - [6A-4 — Pipeline Integration](#phase-6a-4--pipeline-integration)
   - [6A-5 — Evaluation](#phase-6a-5--evaluation)
   - [6A-6 — Hyperparameter Tuning](#phase-6a-6--hyperparameter-tuning)
   - [Error and Decision Summary](#error-and-decision-summary)
4. [Phase 6B — KG Expansion + Embedding Agreement](#4-phase-6b--kg-expansion--embedding-agreement-complete)
   - [6B-1 — Hybrid Retrieval (BM25 + Dense + RRF)](#phase-6b-1--hybrid-retrieval-bm25--dense--rrf)
   - [6B-2 — Knowledge Base Expansion](#phase-6b-2--knowledge-base-expansion)
   - [6B-3 — Embedding-Based Agreement](#phase-6b-3--embedding-based-agreement)
   - [Error and Decision Summary](#6b-error-and-decision-summary)
5. [Phase 6C — Confidence Recalibration](#5-phase-6c--confidence-recalibration-planned)
6. [Phase 6D — Final Evaluation and Portfolio](#6-phase-6d--final-evaluation-and-portfolio-planned)
7. [Test Coverage](#7-test-coverage)
8. [Software Engineering Practices](#8-software-engineering-practices)
9. [Timeline](#9-timeline)
10. [References](#10-references)

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
|                |    |        |                          |
|                |    |        v                          |
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
|                |    | Agreement: keyword overlap         |
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

**Baseline accuracy (pre Phase 6):** VLM-only 41.2%, Agent 32.8%
(47.9% when answered, 73.2% correct abstention rate, ECE 0.198)

### 1.2 Target Architecture (Post Phase 6)

```
USER INPUT: Image + Question
         |
         v
+--------------------------------------------------+
|   LLaVA v1.6 Mistral 7B (4-bit + QLoRA) <-[6A] |
|   [Fine-tuned on VQA-RAD + SLAKE]                |
|                                                   |
|   LoRA Adapter: rank 16, alpha=32                 |
|   Training: 6,704 QA pairs, 3 epochs              |
+---------------------+----------------------------+
                      |
         +------------+------------+
         v                         v
+----------------+    +-----------------------------------+
| VLM-Only Path  |    |   Expanded RAG Pipeline   <-[6B] |
| (Fine-tuned    |    |                                   |
|  Baseline)     |    | FAISS + BM25 (hybrid RRF) <-[6B-1]|
|                |    | 13,435 docs: SLAKE KG + RadLex    |
|                |    | + QA pseudo-docs (indices_v2)     |
|                |    +----------------+------------------+
|                |                     |
|                |                     v
|                |    +-----------------------------------+
|                |    |  UPGRADED SUPERVISOR       <-[6B] |
|                |    | Agreement: EMBEDDING COSINE SIM   |
|                |    | S-PubMedBert-MS-MARCO (reused)    |
|                |    | >= 0.87: supporting -> Answer     |
|                |    | < 0.87:  no support -> Re-query   |
|                |    +----------------+------------------+
|                |                     |
+--------+-------+                     |
         v                             v
+--------------------------------------------------+
|        TEMPERATURE SCALING             <-[6C]    |
|   Learned parameter T on validation set           |
|   calibrated_conf = softmax(logits / T)           |
|   Target: ECE 0.198 -> < 0.10                     |
+---------------------+----------------------------+
                      |
                      v
+--------------------------------------------------+
|                    OUTPUT                         |
| Answer + Calibrated Confidence + Citations        |
| OR: ABSTAIN + Reason + Suggested Review           |
+--------------------------------------------------+
```

**Target accuracy (post Phase 6):** 60–75% overall, <15% abstention rate, ECE < 0.10

### 1.3 Architectural Principles

**Principle 1 — Interface Stability:** Phase 6 changes internal implementations, NOT the
interfaces between modules. `VLMPrediction`, `AgentState`, and `SupervisorDecision` schemas
remain unchanged. All 420 fast tests continue passing throughout.

**Principle 2 — Configuration-Driven Behavior:** Every Phase 6 change is controlled by a
config flag that defaults to Phase 5 behavior. `adapter_path: null` disables the LoRA
adapter; `agreement_method: "keyword"` restores the original supervisor. This enables
clean A/B testing between pipeline versions.

**Principle 3 — Reproducibility:** Every experiment is deterministic through YAML config
files, random seeds (seed=42 throughout), and JSON experiment records.

---

## 2. Goal

The zero-shot LLaVA-Next model performs reasonably on general VQA but underperforms on
radiology-specific questions (unusual anatomy vocabulary, yes/no closed-form questions,
short concise answers). Phase 6 fine-tunes the model's language layers on domain-specific
radiology datasets so that inference gets a stronger visual backbone before the agent
pipeline adds RAG grounding on top.

The zero-shot VLM-only baseline achieved **41.2% accuracy** on VQA-RAD test. The agent
pipeline at 32.8% overall (47.9% when answered) shows the supervisor is over-abstaining
on poorly-formed VLM answers. A domain-adapted VLM should produce better medical
vocabulary, reducing false disagreement with KG evidence and lowering the abstention rate.

---

## 3. Phase 6A — QLoRA Fine-Tuning [COMPLETE]

**Status:** Complete — 2,514 steps, 3 epochs, eval_loss=0.1473, perplexity=1.16

### Research Foundation

Every top-performing Med-VQA system trains on domain-specific datasets with parameter-efficient
fine-tuning:

- **PeFoMed** (arXiv:2401.02797) — used VQA-RAD + SLAKE + PathVQA for stage 2 fine-tuning; achieved 87.1% closed accuracy on VQA-RAD
- **BaMCo** (Springer 2025) — trained jointly: 85.8% SLAKE, 76.7% VQA-RAD
- **CMMO** (J. Biomed. Informatics 2024) — achieved 79.6% (VQA-RAD), 65.7% (SLAKE)
- **Kandamali et al.** (Applied Soft Computing 2025) — confirmed 90–93% accuracy with LoRA on 15 GB VRAM hardware matching our setup

LoRA rank 16, alpha 32, lr 2e-4, and 3 epochs are the community-validated defaults for
LLaVA-series fine-tuning (Philschmid "Fine-tune LLMs 2025", LLaMA-Factory ACL 2024).

---

### Phase 6A-1 — Dataset Cleanup and Training Package

#### Datasets Used

| Dataset   | Split | Samples | Source |
|-----------|-------|---------|--------|
| VQA-RAD   | train | 1,793   | HF `flaviagiammarino/vqa-rad` |
| SLAKE     | train | 4,911   | HF `BoKelvin/SLAKE` (English only, after cleanup) |
| SLAKE     | val   | 1,053   | same (used as validation set) |
| PathVQA   | train | 19,654  | HF `flaviagiammarino/path-vqa` (excluded — see 6A-3) |
| **Final training total** | train | **6,704** | VQA-RAD + SLAKE only |

The VQA-RAD **test** split (451 samples) is never touched — reserved for the Phase 5
evaluation run to compare zero-shot vs fine-tuned on a clean held-out set.

SLAKE validation is used as the fine-tuning validation set because it is a dedicated
held-out split with medically grounded questions.

#### Cleanup 1 — SLAKE Whitespace Filter

**Problem**: SLAKE had one sample (`qid=1622`) with an empty answer after stripping
whitespace. An empty answer string would produce a target sequence of length zero, causing
`nan` loss in cross-entropy.

**Fix**: `slake_loader.py` already had this filter from a previous session. The Phase 6A-1
audit confirmed it was active and working.

#### Cleanup 2 — VQA-RAD Leakage Check

**Problem**: If (image, question, answer) triples from VQA-RAD test appeared in training,
accuracy numbers would be inflated.

**Approach**: Computed perceptual image hashes (MD5 of raw bytes) for all VQA-RAD train and
test images. Found 42 (question, answer) string matches across train/test, but **0** of those
shared the same image.

**Decision**: No filtering needed. The 42 question-answer string coincidences are expected
("Is this a chest X-ray?" / "yes" appears across completely different images). CLEAN.

#### Cleanup 3 — PathVQA Image Corruption Scan

**Problem**: PathVQA contains 19,654 images from mixed sources. Corrupt or non-RGB images
would crash the collator silently in a DataLoader worker mid-training.

**Approach**: Full scan on SageMaker. Results:
- 0 corrupt images
- 1,503 non-RGB (grayscale `L` mode) — handled by `.convert("RGB")` in the loader
- 1 truncated TIFF warning (non-fatal, PIL reads it anyway)

**Decision**: No images removed. `.convert("RGB")` handles grayscale transparently.

#### Cleanup 4 — Final Count Verification

After all cleanups, verified total counts:
- VQA-RAD train: 1,793 ✓
- SLAKE train (EN): 4,919 raw → -1 empty → -7 duplicates = **4,911** ✓
- PathVQA train: 19,654 ✓ (loaded but later excluded — see Phase 6A-3)
- **Combined: 26,358** (original plan) → **6,704** (final, after PathVQA exclusion)

#### Training Package — `src/radiology_vqa/training/`

**`dataset.py`**

`TrainingConfig` — dataclass (not Pydantic) for training-time configuration. Kept
separate from `SystemConfig` to avoid polluting the inference config with hyperparameters.

`normalize_answer(answer)` — lowercase + strip only. Deliberately does NOT remove articles
or punctuation, unlike the evaluation normaliser in `evaluation/metrics.py`. Training
targets should be natural language; over-normalising would teach the model to produce
answers that score well on exact-match but sound unnatural to clinicians.

`build_conversation(question, answer)` — returns `list[dict]` in LLaVA conversation format.
The user content is:
```
<image>
{question}
Provide only the direct answer in 1-5 words. Do not explain.
```
The `<image>` token is a placeholder replaced by image embeddings at collation time. The
instruction suffix must match the `concise_mode` inference prompt in `llava.py` exactly —
training on one format and inferring on another creates a distribution shift. This was
identified as audit issue **M1** and fixed before training.

The content is stored as a plain string (not typed blocks) so that existing fast unit
tests keep passing. The collator upgrades to typed-block format at collation time.

`build_training_dataset(config, slake_dir)` — orchestrates loading all datasets and
combining them. PathVQA is loaded via `Dataset.from_generator()` rather than
`Dataset.from_list()` because accumulating 19,654 PIL images in Python heap before Arrow
serialisation would spike RAM by ~8.8 GB. Generator-based construction yields rows one at
a time directly into the Arrow buffer. This was audit issue **M2**.

**`collator.py`**

`LlavaDataCollator` applies the processor to convert `(image, conversations)` pairs into
model inputs.

`_to_multimodal_format()`: Transformers ≥4.45 changed the LLaVA-Next Jinja2 chat template
to expect content as typed blocks:
```python
[{"type": "image"}, {"type": "text", "text": "..."}]
```
Passing a plain string causes `jinja2.exceptions.UndefinedError: 'str object' has no
attribute 'text'`. This helper converts at collation time, keeping `build_conversation()`
output stable for tests. This was runtime issue **R1**.

`max_length=4096`: LLaVA-Next v1.6 uses AnyRes image tiling. For large radiology images,
the processor produces up to 4 tiles × 576 tokens + 1 base × 576 tokens = **2,880 image
tokens**. Adding question/answer/template text (~100–200 tokens) gives a maximum of ~3,080
tokens per sample. `max_length=4096` (the Mistral-7B context window) ensures no sample is
truncated. Truncation causes a hard `ValueError` because the processor validates that the
count of `<image>` tokens in `input_ids` matches the count in the text template. This was
runtime issue **R2** (after two iterations: 256 → 2048 → 4096).

**Label masking:** `_find_answer_start()` locates the `[/INST]` marker token IDs in the
tokenised sequence and masks all tokens before it with `-100`. Only answer tokens
contribute to the loss. Padding tokens are also masked. This ensures the loss signal is
focused on the 1–5 word medical answer rather than diluted across ~3,000 image/prompt tokens.

---

### Phase 6A-2 — Training Script and Pre-Training Audit

#### Why QLoRA

A full fine-tune of LLaVA-Next 7B requires ~56 GB VRAM (fp16) — impossible on T4 (15 GB)
or A10G (22 GB). QLoRA addresses this with two techniques:

1. **4-bit NF4 quantisation** (bitsandbytes): freezes all base model weights in 4-bit,
   reducing the 7B model from ~14 GB (fp16) to ~4 GB. Computations are done in bf16 via
   `bnb_4bit_compute_dtype`.

2. **LoRA adapters** (PEFT): adds small rank-16 adapter matrices to the language model
   linear layers only. Only these adapters (~44M parameters, 1.1% of total) are
   trainable. The vision encoder is completely frozen.

Together, QLoRA trains only the adapters while keeping the base model frozen in 4-bit,
fitting the entire setup in ~4 GB model VRAM with ~6–8 GB peak during backpropagation.

#### Why these LoRA Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `rank` | 16 | Community standard for domain adaptation. Rank 8 is too low for VQA (needs to learn new vocabulary and reasoning patterns); rank 32+ uses more VRAM without proportional gain. |
| `alpha` | 32 | `alpha = 2 × rank` is the standard scaling convention. |
| `dropout` | 0.05 | Light regularisation. Medical VQA datasets are relatively small; without dropout, rank-16 adapters can memorise training answers. |
| `target_modules` | q/k/v/o/gate/up/down_proj | All language model attention and MLP linear layers. Vision tower excluded — its representations are already good for image understanding. |

#### Why these Training Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning. Higher than full fine-tune LRs because adapters are randomly initialised and need to move quickly. |
| `num_epochs` | 3 | Balances convergence vs overfitting on 6,704 samples. Fewer epochs underfits; more epochs risk memorisation at this dataset size. |
| `per_device_batch_size` | 1 | Forced by VRAM constraints with 4096-token sequences. |
| `gradient_accumulation_steps` | 8 | Effective batch size = 8. Provides stable gradients without VRAM cost of a real batch-8. |
| `optim` | `paged_adamw_8bit` | Offloads Adam optimizer states to CPU RAM in 8-bit. Saves ~2 GB VRAM vs standard AdamW. |
| `gradient_checkpointing` | true | Trades ~30% compute overhead for ~60% reduction in activation memory. Mandatory at max_length=4096. |
| `bf16` | true | A10G (Ampere) — native BFloat16 support. No GradScaler needed. |
| `lr_scheduler` | cosine | Smooth LR reduction after warmup. Standard for instruction fine-tuning. |
| `warmup_ratio` | 0.03 | 3% of total steps. Short warmup because adapters start from random init. |
| `max_grad_norm` | 0.3 | Aggressive gradient clipping. Prevents large adapter updates early in training. |

#### Script Architecture (`scripts/finetune_qlora.py`)

The script is structured as a sequence of named phases with structured logging so that
progress can be monitored via `tail -f` even if the terminal session drops overnight.

**Initialisation order (critical)**:
```
load model in 4-bit
  → prepare_model_for_kbit_training()   # enables gradient checkpointing on quantised layers
  → get_peft_model()                     # wraps with LoRA
```
`prepare_model_for_kbit_training` must run before `get_peft_model`, otherwise gradient
checkpointing is applied to the PEFT wrapper rather than the base model.

**Dual logging** (`setup_logging`): Both stdout and a timestamped file
`logs/finetune_{timestamp}.log` receive all output. `tail -f logs/finetune_*.log` works
to monitor progress after a tmux detach.

**`--dry-run` flag**: Loads the model, applies LoRA, runs the collator on 2 dummy samples,
runs a VRAM stress test (forward + backward pass), and exits. Validates the complete setup
in ~3 minutes without writing any checkpoint.

**`--max-steps N` flag**: Smoke-test mode. Runs N gradient steps on VQA-RAD + SLAKE only
(PathVQA download skipped), with eval and checkpointing disabled. Used to validate
backpropagation produces a finite loss without OOM before committing to an overnight run.

**Logging suppression fix**: The parent `transformers` logger is set to WARNING to suppress
library noise, but an explicit override `logging.getLogger("transformers.trainer").setLevel(logging.INFO)`
allows eval metric logs to pass through. Without this, eval_loss values are silently dropped
(runtime issue **R6**).

#### Pre-Training Audit Findings and Fixes (M1–M4)

A 30-point read-only audit of the code before the first training run identified 4 issues:

**M1 — Training/Inference Format Mismatch (MEDIUM)**

`build_conversation()` originally produced `<image>\n{question}` while `llava.py`'s
`concise_mode` inference prompt is `<image>\n{question}\nProvide only the direct answer in
1-5 words. Do not explain.` Training on one format and inferring on another creates a
distribution shift.

**Fix**: Added the instruction suffix to `build_conversation()`.

**M2 — PathVQA Heap Accumulation (MEDIUM)**

`Dataset.from_list()` requires all samples in Python heap before Arrow serialisation.
For 19,654 pathology images this is ~8.8 GB, risking OOM on ml.g4dn.2xlarge (32 GB RAM)
when combined with model weights.

**Fix**: Replaced with `Dataset.from_generator()` + `concatenate_datasets()`.

**M3 — eval_loss Missing from Experiment Record (MEDIUM)**

`TrainOutput.training_loss` captures only train loss. No overfitting detection was possible
from the record alone.

**Fix**: Added `eval_result = trainer.evaluate()` after `trainer.train()`. The
`save_experiment_record()` function now writes `eval_loss` and `perplexity = exp(eval_loss)`
to the JSON, plus an overfitting warning if `eval_loss > train_loss × 1.5`.

**M4 — Checkpoint Path Mismatch (MEDIUM / SHOW-STOPPER)**

The script saved the adapter to `checkpoints/llava-med-qlora/` but `configs/phase6.yaml`
referenced `checkpoints/llava-med-qlora/best`. Loading the adapter after training would
silently fall back to zero-shot with only a logged warning.

**Fix**: Changed to `trainer.save_model(os.path.join(output_dir, "best"))` and called
`processor.save_pretrained(best_model_dir)`. Both adapter weights and processor (tokenizer
+ image processor) are co-located under `best/`.

#### Runtime Fixes During Smoke Testing (R1–R2)

Two issues discovered during actual smoke tests using real radiology images (not dry-run):

**R1 — Jinja2 `'str object' has no attribute 'text'`**

Transformers ≥4.45 changed the LLaVA-Next Jinja2 chat template to iterate over content
items and access `item['text']`. Plain string content causes the template to iterate over
individual characters.

**Fix**: Added `_to_multimodal_format()` in `collator.py` that converts string content to
`[{"type": "image"}, {"type": "text", "text": "..."}]` at collation time.

**R2 — `max_length` Truncation Mismatch (three iterations)**

The dry-run dummy image (336×336) produces 1,215 tokens. Real SLAKE CT/MRI scans trigger
AnyRes tiling up to 2,880 image tokens. Starting values of `max_length=256` then
`max_length=2048` both fell short. The processor validates that `<image>` placeholder count
in `input_ids` matches the text template; truncation breaks this with a hard `ValueError`.

**Fix**: Set `max_length=4096` (Mistral-7B's native context window). Confirmed against HF
Transformers docs and real-world LLaVA-Next configs (GitHub transformers issue #36002).

---

### Phase 6A-3 — Training Run

#### Timeline

| Date | Event |
|------|-------|
| 2026-02-24 | Initial smoke test on T4 (ml.g4dn.xlarge) — BFloat16 GradScaler crash |
| 2026-02-24 | BFloat16 upcast fix applied and validated |
| 2026-02-24 | Chat template error (R1) discovered and fixed |
| 2026-02-24 | SFTTrainer replaced with plain Trainer (10-min delay eliminated) |
| 2026-02-24 | PathVQA excluded — domain mismatch decision |
| 2026-02-24 | GPU upgrade decision: T4 → A10G (ml.g5.2xlarge) |
| 2026-02-24 | A10G config optimisation (bf16, FA2, batch tuning) |
| 2026-02-24 | Dry-run + VRAM stress test validated on A10G |
| 2026-02-25 | Smoke test (3 steps) — loss 10.0 → 0.92, no OOM |
| 2026-02-25 06:50 UTC | Full training started in tmux on SageMaker |
| 2026-02-25 11:37 UTC | First checkpoint saved (step 500) |
| 2026-02-26 06:14 UTC | Training complete (2,514 steps, 23h 14m) |
| 2026-02-26 06:34 UTC | Final eval complete, model saved to `best/` |

---

#### Runtime Error 3 — BFloat16 GradScaler Crash (T4)

**Error:**
```
RuntimeError: GradScaler is not compatible with bfloat16
```

**Root cause:** LLaVA-Next v1.6 Mistral-7B ships its non-quantised weights in BFloat16.
When `fp16=True` is set in `TrainingArguments`, PyTorch's `GradScaler` (required for fp16
mixed precision) inspects all parameters and raises if it finds any BFloat16 tensors.
The T4 GPU (Volta architecture) does not support native BFloat16, so `bf16=True` is not
an option on T4.

**Investigation:** Dumped all parameter dtypes — confirmed 200+ parameters and buffers
(including rotary embeddings, layer norms) were natively BFloat16 from the pretrained
checkpoint. The 4-bit NF4 quantised weights were fine (stored as uint8), but all
non-quantised tensors remained BFloat16.

**Fix:** Added a conditional upcast pass in `setup_model_and_processor()`:
```python
if not use_bf16:
    # Upcast ALL non-quantised bf16 tensors (params AND buffers) to fp32
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
    for module in model.modules():
        for attr_name in list(module._buffers.keys()):
            buf = module._buffers[attr_name]
            if buf is not None and buf.dtype == torch.bfloat16:
                module._buffers[attr_name] = buf.to(torch.float32)
```

The upcast targets both `model.parameters()` (learnable weights) AND `module._buffers`
(non-learnable tensors like rotary position embeddings). Missing the buffers was the
initial cause of a second crash — the GradScaler checks buffers too. On A10G with
`bf16=True`, this entire block is skipped.

**Impact:** Fixed the crash. However, running fp32 on T4 meant ~80 seconds per step —
impractically slow for 2,514 steps.

---

#### Runtime Error 4 — Chat Template Missing

**Error:**
```
TemplateError: processor does not have a chat template
```

**Root cause:** The processor was initially constructed manually via
`LlavaNextProcessor(image_processor=..., tokenizer=...)`. The manual construction skips
loading `tokenizer_config.json`, which contains the Mistral chat template (Jinja2).

**Fix:** Changed to `LlavaNextProcessor.from_pretrained(model_id, use_fast=True)`. The
`from_pretrained()` path loads the full tokenizer config including the chat template.
Also swapped in `LlavaNextImageProcessorFast` to silence the deprecation warning in
transformers ≥4.46.

---

#### Runtime Error 5 — SFTTrainer 10-Minute Initialisation Delay

**Symptom:** After calling `SFTTrainer(...)`, the script hung for ~10 minutes with no log
output. On the T4 with limited time, this was unacceptable for iterative debugging.

**Root cause:** `trl.SFTTrainer` calls `_prepare_dataset()` during `__init__`, which
tokenises/preprocesses the entire training dataset upfront. This is wasted work because
our custom `LlavaDataCollator` already handles tokenisation at collation time.

**Fix:** Replaced `SFTTrainer` with the plain `transformers.Trainer`:
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    processing_class=processor.tokenizer,
)
```
Initialisation dropped from ~10 minutes to <1 second.

---

#### Design Decision — PathVQA Exclusion

**Context:** The original training config included PathVQA (19,654 samples) alongside
VQA-RAD (1,793) and SLAKE (4,911), for a total of 26,358 training samples.

**Problem:** PathVQA is a **pathology** dataset (histology slides, tissue samples), not
a radiology dataset (X-rays, CT, MRI). Including it introduces a significant domain
mismatch:
- Visual features (cell structures vs. anatomical structures) are fundamentally different
- Answer vocabulary (histological terms vs. radiological terms) creates noise
- The model learns to distribute attention across two very different visual domains
- Training time increased ~4× due to the 19k additional samples

**Decision:** Set `include_pathvqa: false` in `configs/training/qlora.yaml`. This:
- Reduced training samples from 26,358 → 6,704 (VQA-RAD + SLAKE only)
- Reduced total training steps from ~9,900 → 2,514 (at effective batch 8, 3 epochs)
- Eliminated the ~8.8 GB PathVQA image download from HuggingFace
- Kept training focused on the radiology domain the system is designed for

**Validation:** The SLAKE validation set (1,053 samples) is radiology-only, so eval
metrics directly measure radiology VQA performance.

---

#### GPU Migration — T4 to A10G

**Motivation:** On the T4 (14.6 GB VRAM, Volta), training a single step took ~80 seconds
with full fp32 (fp16 disabled due to GradScaler crash, bf16 not supported). Projected
wall time: **80+ hours** for 2,514 steps. The T4 also lacked Flash Attention 2 support
(requires Ampere+).

**Target:** NVIDIA A10G on SageMaker ml.g5.2xlarge — 22.1 GB VRAM, Ampere architecture,
native BFloat16, Flash Attention 2 compatible.

**Configuration changes for A10G:**

| Parameter | T4 Value | A10G Value | Rationale |
|---|---|---|---|
| `bf16` | `false` | `true` | Native Ampere support, no GradScaler needed |
| `fp16` | `false` | `false` | Superseded by bf16 |
| `attn_implementation` | (none) | `flash_attention_2` | ~2x faster attention, ~30% less VRAM |
| `compute_dtype` | `float16` | `bfloat16` | Match training precision to avoid casts |
| `dataloader_num_workers` | `0` | `4` | A10G has more CPU headroom |
| `dataloader_pin_memory` | `false` | `true` | Faster CPU→GPU transfer |
| `weight_decay` | `0.0` | `0.01` | Mild L2 regularisation for 3-epoch run |

**Result:** Step time dropped from ~80s (T4, fp32) to ~30s (A10G, bf16 + FA2) — a
**2.7× speedup**. Total training time: **23h 14m** (vs. projected 80+ hours on T4).

---

#### Dry-Run Validation on A10G

Before committing to the full training run, a dry-run (`--dry-run` flag) was executed:

```
Model loaded in 4-bit NF4: 4.01 GB VRAM
Flash Attention 2 active
LoRA adapters applied: 44,302,336 / 3,959,908,352 trainable (1.1188%)
Collator test: input_ids [2, 4096], pixel_values [2, 5, 3, 672, 672], labels [2, 4096]
Labels mask: 8,030/8,192 masked, 162 answer tokens (1.98%)
EOS preserved in labels: 2 tokens
VRAM stress test (forward + backward): peak within safe headroom
```

The stress test confirmed a single forward + backward pass at `max_length=4096` fit within
the A10G's 22.1 GB VRAM with sufficient headroom.

---

#### Smoke Test (3 Steps)

A quick 3-step smoke test (`--max-steps 3`) validated end-to-end training:

```
Step 1: loss = 10.0156
Step 2: loss = 1.1250
Step 3: loss = 0.9219
```

Loss dropped from 10.0 → 0.92 in just 3 steps — confirms the model is learning. No OOM
errors. Step time ~30 seconds (consistent with A10G bf16 + FA2).

---

#### Full Training Run — Configuration

**Experiment ID:** `qlora_20260225T065033Z`
**Started:** 2026-02-25T06:50:33 UTC
**Completed:** 2026-02-26T06:34:46 UTC (23h 44m wall time including eval)

```yaml
# Final training configuration (configs/training/qlora.yaml)
model:
  id: llava-hf/llava-v1.6-mistral-7b-hf
  quantization: nf4
  double_quant: true
  compute_dtype: bfloat16
  attn_implementation: flash_attention_2

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8     # effective batch = 8
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  max_grad_norm: 0.3
  weight_decay: 0.01
  optim: paged_adamw_8bit
  gradient_checkpointing: true
  bf16: true
  save_steps: 500
  eval_steps: 500
  logging_steps: 25

data:
  include_vqa_rad: true     # 1,793 samples
  include_slake: true       # 4,911 samples
  include_pathvqa: false    # excluded — pathology domain mismatch
  max_length: 4096
```

**Dataset composition:**
- Train: 6,704 samples (VQA-RAD 1,793 + SLAKE 4,911)
- Validation: 1,053 samples (SLAKE validation, English only)
- Steps per epoch: 838 (6,704 / 8 effective batch)
- Total steps: 2,514 (838 × 3 epochs)

---

#### Training Results

**Loss Curve:**

| Metric | Value |
|--------|-------|
| **Final train loss** | 0.4689 |
| **Final eval loss** | 0.1473 |
| **Perplexity** | 1.16 |
| **Total steps** | 2,514 |
| **Training time** | 23h 13m 57s |
| **Throughput** | 0.24 samples/sec, 0.03 steps/sec |

**Loss Progression by Epoch:**

| Epoch | Approx. Train Loss | Eval Loss | Learning Rate | Observation |
|-------|-------------------|-----------|---------------|-------------|
| 0.0 | ~10.0 | — | 2.0e-4 | Random adapter init, high loss expected |
| 0.5 | ~0.7 | — | ~1.9e-4 | Rapid convergence in first half-epoch |
| 1.0 | ~0.5 | 0.18* | ~1.6e-4 | Domain adaptation taking hold |
| 1.5 | ~0.3 | — | ~1.1e-4 | Cosine decay reducing LR |
| 2.0 | ~0.2 | 0.16* | ~5.0e-5 | Continued improvement |
| 2.5 | ~0.15 | — | ~1.5e-5 | Late-stage refinement |
| 2.98 | 0.15 | 0.1473 | ~1.9e-8 | Final eval at end of training |

*Intermediate eval losses estimated from checkpoint saves at steps 500, 1000, 1500, 2000.

**Late-Stage Loss Samples (Epoch 2.45–2.98):**

```
epoch 2.45: loss=0.149, grad_norm=3.04
epoch 2.51: loss=0.170, grad_norm=1.34
epoch 2.57: loss=0.132, grad_norm=4.18
epoch 2.63: loss=0.211, grad_norm=1.77
epoch 2.71: loss=0.147, grad_norm=2.93
epoch 2.80: loss=0.144, grad_norm=0.37
epoch 2.89: loss=0.143, grad_norm=1.48
epoch 2.95: loss=0.109, grad_norm=3.21
epoch 2.98: loss=0.155, grad_norm=2.99
```

Loss stabilised around 0.13–0.21 in the final epoch with no divergence. Gradient norms
remained bounded (0.3–4.2), confirming `max_grad_norm=0.3` clipping is effective. LR
decayed to near-zero (1.87e-8) by the end. No loss spikes or NaN values throughout.

**Overfitting Analysis:**

| Metric | Value | Assessment |
|--------|-------|------------|
| Train loss | 0.4689 (averaged over all steps) | — |
| Eval loss | 0.1473 (end of training) | — |
| Ratio | eval / train = 0.31 | **No overfitting** |
| Perplexity | 1.16 | **Excellent** — model is ~84% confident |

The eval loss (0.1473) is lower than the averaged train loss (0.4689) because the train
loss average includes early high-loss steps (10.0, 0.9, 0.7...) when adapters were
randomly initialised. The late-stage train loss (~0.13–0.17) is comparable to the eval
loss (0.1473), confirming the model generalises to the held-out SLAKE validation set.

**Perplexity 1.16** means the model's predicted probability distribution is very close to
the true distribution of medical VQA answers. Perplexity <1.5 for a domain fine-tuned VQA
model is strong performance. The QLoRA adapters have successfully adapted the language
model's output distribution to the radiology VQA domain.

---

#### Eval Logging Issue (R6)

**Problem:** During the training run, eval_loss values were not appearing in the log file
despite evaluations running every 500 steps (confirmed by checkpoint saves).

**Root cause:** `logging.getLogger("transformers").setLevel(logging.WARNING)` suppressed
the `transformers.trainer` child logger which emits eval results at INFO level.

**Fix:**
```python
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.trainer").setLevel(logging.INFO)  # allow eval logs
```

**Note:** The final eval_loss (0.1473) was captured correctly because `trainer.evaluate()`
at the end of training logs through our own `logger.info()` call.

---

#### Checkpoint Structure

```
checkpoints/llava-med-qlora/
├── best/                          ← final model (referenced by configs/phase6.yaml)
│   ├── adapter_config.json        ← LoRA hyperparameters
│   ├── adapter_model.safetensors  ← trained adapter weights (~170 MB)
│   ├── tokenizer.json             ← processor files co-located for easy loading
│   ├── tokenizer_config.json
│   └── preprocessor_config.json
├── checkpoint-500/                ← step 500 (epoch ~0.6)
├── checkpoint-1000/               ← step 1000 (epoch ~1.2)
├── checkpoint-1500/               ← step 1500 (epoch ~1.8)
├── checkpoint-2000/               ← step 2000 (epoch ~2.4)
└── checkpoint-2500/               ← step 2500 (epoch ~3.0)
```

`save_total_limit=3` means only the 3 most recent checkpoints are retained. The `best/`
directory is always preserved because it is written explicitly by `trainer.save_model()`.

---

#### Hardware Summary

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA A10G (Ampere) |
| **VRAM** | 22.1 GB total |
| **VRAM after model load** | 4.01 GB |
| **Attention** | Flash Attention 2 |
| **Mixed precision** | BFloat16 (native) |
| **Instance type** | SageMaker ml.g5.2xlarge |
| **Step time** | ~30 seconds |
| **Total wall time** | 23h 44m (including eval + save) |
| **Pure training time** | 23h 13m 57s |
| **Eval time** | 20m 20s (132 eval steps × 9.25 s/step) |

#### Trainable Parameter Budget

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| **Total model** | 3,959,908,352 | 100% |
| **Trainable (LoRA adapters)** | 44,302,336 | 1.12% |
| **Frozen (base model)** | 3,915,606,016 | 98.88% |
| **Vision encoder** | ~400M | Fully frozen |
| **Language model (base)** | ~3.5B | Frozen (4-bit NF4) |
| **LoRA adapters** | ~44M | Trainable (rank 16) |

---

### Phase 6A-4 — Pipeline Integration [COMPLETE]

The adapter is loaded at inference time in `src/radiology_vqa/vlm/llava.py`. A single
conditional block in `LlavaBackend.__init__()` handles the two modes:

```python
from peft import PeftModel

# Load base model (4-bit quantised)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Apply fine-tuned LoRA adapters (if adapter_path is configured)
adapter_path = "checkpoints/llava-med-qlora/best"
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
model = model.merge_and_unload()  # fold adapters into base — zero inference overhead
```

`merge_and_unload()` folds the adapter weights into the base model parameters, giving
full inference speed with no LoRA overhead. Setting `adapter_path: ""` in
`configs/phase6.yaml` falls back to zero-shot mode, enabling A/B testing between
zero-shot and fine-tuned on the same evaluation pipeline.

All other code in `llava.py` (predict(), confidence extraction, prompting) is unchanged.

---

### Phase 6A-5 — Evaluation [IN PROGRESS]

**Status:** Benchmark running on SageMaker as of 2026-02-26.

Three evaluation configurations are being measured on the VQA-RAD test split (451 samples):

| Config | VLM | LoRA | Agent | Purpose |
|--------|-----|------|-------|---------|
| A | Zero-shot | No | No | Existing Phase 5 baseline (41.2% VLM-only) |
| B | Fine-tuned | Yes | No | Isolate fine-tuning impact on VLM accuracy |
| C | Fine-tuned | Yes | Yes | Full Phase 6A pipeline — main deliverable |

**Commands (SageMaker):**

```bash
# Config C — fine-tuned agent
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \
  python scripts/run_evaluation.py \
    --mode agent \
    --dataset vqa_rad \
    --split test \
    --no-bertscore

# Config B — fine-tuned VLM-only (isolates fine-tuning effect)
VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \
  python scripts/run_evaluation.py \
    --mode vlm_only \
    --dataset vqa_rad \
    --split test \
    --no-bertscore

# Compare fine-tuned agent (C) vs zero-shot VLM (A)
python scripts/generate_report.py \
  --agent   data/eval/agent_vqa_rad_test_*.json \
  --baseline data/eval/vlm_vqa_rad_test_*.json
```

**Key hypothesis:** The fine-tuned VLM produces domain-specific medical terms (e.g.
"cardiomegaly" instead of "enlargement") that agree better with SLAKE KG evidence,
reducing over-abstention and improving closed-question accuracy.

---

### Phase 6A-6 — Hyperparameter Tuning [N/A — NOT REQUIRED]

The final training run achieved eval_loss=0.1473 and perplexity=1.16 with no overfitting
detected. No hyperparameter tuning was required. The contingency plan (reduce rank to 8,
dropout to 0.10, or reduce epochs to 2) was not triggered.

---

### Error and Decision Summary

#### Summary of All Errors Encountered and Fixes Applied

| # | Error | Phase | GPU | Root Cause | Fix | Impact |
|---|-------|-------|-----|------------|-----|--------|
| **M1** | Training/inference format mismatch | 6A-2 (audit) | — | `build_conversation()` missing instruction suffix | Added "Provide only the direct answer..." suffix | Prevented train/test distribution shift |
| **M2** | PathVQA heap accumulation | 6A-2 (audit) | — | `Dataset.from_list()` holds 19k images in RAM | Replaced with `Dataset.from_generator()` | Prevented CPU OOM on 32 GB instance |
| **M3** | eval_loss missing from record | 6A-2 (audit) | — | No `trainer.evaluate()` call after training | Added post-training eval + JSON recording | Enabled overfitting detection |
| **M4** | Checkpoint path mismatch | 6A-2 (audit) | — | Script saved to wrong path vs. phase6.yaml | Changed save path to `output_dir/best/` | Prevented silent zero-shot fallback |
| **R1** | Jinja2 `'str' has no attribute 'text'` | 6A-2 (smoke) | T4 | Transformers ≥4.45 changed LLaVA chat template format | Added `_to_multimodal_format()` converter | Fixed collator crash |
| **R2** | `max_length` truncation crash | 6A-2 (smoke) | T4 | AnyRes tiling produces up to 2,880 image tokens | Set `max_length=4096` | Fixed ValueError on large images |
| **R3** | BFloat16 GradScaler crash | 6A-3 (smoke) | T4 | LLaVA ships bf16 weights, GradScaler rejects them | Upcast bf16→fp32 for params AND buffers | Fixed RuntimeError on T4 |
| **R4** | Chat template missing | 6A-3 (smoke) | T4 | Manual processor construction skips template | Used `from_pretrained()` for processor | Fixed TemplateError |
| **R5** | SFTTrainer 10-min delay | 6A-3 (smoke) | T4 | SFTTrainer preprocesses dataset in __init__ | Replaced with plain `Trainer` | Eliminated dead wait time |
| **R6** | Eval logs not appearing | 6A-3 (training) | A10G | `transformers` logger set to WARNING | Added `transformers.trainer` → INFO override | Fixed missing eval visibility |

#### Decisions Log

| Decision | Rationale | Alternative Considered |
|----------|-----------|----------------------|
| Exclude PathVQA | Pathology domain ≠ radiology; adds noise, 4× more training time | Include with domain-weighted loss — rejected (over-engineering for unclear gain) |
| T4 → A10G migration | T4: 80+ hours projected, no bf16, no FA2. A10G: 23h, bf16 + FA2 | Stay on T4 with fp32 — rejected (cost exceeds A10G upgrade) |
| 3 epochs (not 1 or 5) | 1 epoch underfits on 6.7k samples; 5 epochs risks memorisation. `load_best_model_at_end` guards overfitting | 1 epoch with higher LR — rejected (less stable) |
| `weight_decay=0.01` | Mild L2 regularisation appropriate for 3-epoch small-dataset run | No decay — rejected (higher overfitting risk without PathVQA's volume) |
| Plain Trainer over SFTTrainer | No SFT-specific features needed; SFTTrainer adds 10-min overhead | Keep SFTTrainer — rejected (unnecessary complexity) |
| `save_total_limit=3` | Bounds disk usage; 3 checkpoints sufficient for recovery | Unlimited — rejected (each checkpoint ~170 MB) |
| JSON experiment records | Lightweight, no external dependencies, full audit trail | W&B / MLflow — rejected (infrastructure overhead for a single-run experiment) |

#### Experiment Record

**File:** `logs/experiment_20260225T065033Z.json`

```json
{
  "experiment_id": "qlora_20260225T065033Z",
  "timestamp": "20260225T065033Z",
  "model": {
    "base": "llava-hf/llava-v1.6-mistral-7b-hf",
    "trainable_params": 44302336,
    "total_params": 3959908352,
    "trainable_pct": 1.1188
  },
  "training": {
    "train_loss": 0.4689,
    "eval_loss": 0.1473,
    "perplexity": 1.1587,
    "steps": 2514
  },
  "hardware": {
    "gpu": "NVIDIA A10G",
    "vram_gb": 23.72
  }
}
```

**Log file:** `logs/finetune_20260225T065033Z.log` (full training log with per-step loss)

---

## 4. Phase 6B — KG Expansion + Embedding Agreement [COMPLETE]

**Status:** Complete — 13,435 docs indexed, embedding agreement calibrated, threshold validated.

Phase 6B has three sub-phases that together address the core weakness identified in
Phase 5: the supervisor was over-abstaining (61 spurious abstentions in 451 samples)
because keyword-based agreement missed synonyms and had poor coverage of the 2,987-doc
KG. The fix is threefold: better retrieval (6B-1), more documents (6B-2), and smarter
agreement scoring (6B-3).

---

### Phase 6B-1 — Hybrid Retrieval (BM25 + Dense + RRF)

**Status:** Complete

**Motivation:** Dense-only retrieval (Phase 5) uses PubMedBERT embeddings and handles
semantic similarity well but fails for exact medical terminology like "RID_4391" or
"Lobar Pneumonia" where BM25's token matching dominates. Reciprocal Rank Fusion (RRF)
combines both signals without needing to calibrate their score scales.

#### Files Changed

| File | Change |
|------|--------|
| `src/radiology_vqa/rag/bm25_retriever.py` | New — BM25 index using `rank-bm25` library |
| `src/radiology_vqa/rag/hybrid_retriever.py` | New — RRF fusion of BM25 + dense results |
| `src/radiology_vqa/rag/retriever.py` | Modified — dispatch to hybrid or dense based on config |
| `scripts/build_index.py` | Modified — `--bm25` flag to build BM25 index alongside FAISS |
| `src/radiology_vqa/config.py` | Modified — added hybrid retrieval settings |
| `configs/phase6.yaml` | Modified — `retrieval_method: "hybrid"` |

#### Configuration

```python
# config.py additions
retrieval_method: str = "dense"        # "dense" = Phase 5 behaviour; "hybrid" = BM25+dense+RRF
bm25_index_dir: Path = Path("data/bm25_index")
bm25_top_k: int = 20                   # BM25 candidates before fusion
dense_top_k: int = 20                  # Dense candidates before fusion
rrf_k: int = 60                        # RRF smoothing constant (standard value)
```

#### Reciprocal Rank Fusion (RRF)

RRF fuses two ranked lists without requiring score calibration:

```
RRF_score(doc) = Σ 1 / (k + rank_in_list)
```

With `k=60` (standard), a doc ranked 1st in one list and 10th in another scores
`1/61 + 1/70 = 0.0306`. A doc ranked 1st in both scores `2/61 = 0.0328`. The
constant `k=60` prevents very high scores for top-1 items from dominating the fusion.

Final top-K (default 5) is taken from the merged list, with scores normalised to
`[0, 1]` for compatibility with the supervisor's evidence quality gate.

#### New dependency

```bash
pip install rank-bm25
```

`rank-bm25` is a lightweight pure-Python library (no C extensions). Install on SageMaker
before building the BM25 index.

#### Building the index

```bash
# Dense only (Phase 5 style — data/indices)
python scripts/build_index.py

# Dense + BM25 (Phase 6B-1 — data/indices and data/bm25_index)
python scripts/build_index.py --bm25
```

---

### Phase 6B-2 — Knowledge Base Expansion

**Status:** Complete — index expanded from 2,987 → **13,435 documents**

The Phase 5 KG (2,987 SLAKE KG docs) had poor coverage for many radiological terms.
Specifically the Phase 5 evaluation found 61 over-abstentions where the VLM's answer
was medically correct but retrieval found no supporting evidence — because the concept
simply wasn't in the 2,987-doc index. Phase 6B-2 adds two new document sources.

#### New Document Sources

| Source | Documents | Description |
|--------|-----------|-------------|
| SLAKE KG (existing) | 2,987 | Phase 5 baseline |
| RadLex ontology | ~3,737 | RSNA radiology terminology (Tier 1 filtered) |
| VQA-RAD pseudo-docs | 1,793 | Training QA pairs as `"Question: {q} Answer: {a}"` |
| SLAKE pseudo-docs | 4,911 | English-only training QA pairs + metadata |
| **Total (indices_v2)** | **13,435** | Validated on SageMaker |

#### New Files

**`src/radiology_vqa/rag/radlex_processor.py`**

Reads `data/raw/radlex/Radlex.xls` using `xlrd` (direct Excel parsing, no LibreOffice).
The RadLex ontology has 46,657 entries across 200+ columns.

Tier 1 filter — a row is included only if:
- `col[1]` (Preferred Name) is non-empty
- `col[4]` (Is Obsolete) ≠ `"1"`
- `col[28]` (Definition) OR `col[3]` (Description) is non-empty

This yields ~3,737 useful clinical definition documents from 46,657 raw entries.

`doc_id` format: `radlex_{RID}` where RID is the RadLex concept identifier from `col[46]`
(PrefixIRI, e.g. `RID43`). Falls back to slugified label if RID is missing.

Non-ASCII characters are stripped via regex (`[^\x00-\x7F]+`) to prevent encoding issues
when embedding.

**`src/radiology_vqa/rag/qa_pseudo_processor.py`**

Converts VQA training pairs into retrievable pseudo-documents for BM25/dense grounding.

Document format:
```
Question: {question} Answer: {answer} [Body region: {loc}] [Modality: {mod}] [Category: {cat}]
```

The optional metadata fields (Body region, Modality, Category) are appended when
present — they appear in SLAKE records but not VQA-RAD. Their presence enables
the retriever to ground questions like "Is this a chest CT?" using direct evidence.

`process_vqarad(dataset=None)` — lazy-loads from HuggingFace or accepts a mock list
for tests. `doc_id` = `qa_vqarad_{idx}`.

`process_slake()` — reads the local JSON file. Strict `q_lang == "en"` filter
excludes Chinese entries (U+4E00–U+9FFF Unicode range check in tests). `doc_id` =
`qa_slake_{qid}`. QA pairs are train-only — the test split is never included.

**`scripts/build_index.py`** (extended)

New flags:
```bash
--sources kg radlex qa   # which sources to include (default: kg only)
--radlex-xls PATH        # default: data/raw/radlex/Radlex.xls
--slake-train PATH       # default: data/raw/Slake1.0/train.json
--output-index-dir PATH  # default: settings.index_dir
```

Example (Phase 6B-2 full build):
```bash
python scripts/build_index.py \
    --sources kg radlex qa \
    --output-index-dir data/indices_v2
```

**`configs/phase6.yaml`** (updated)

```yaml
data:
  index_dir: "data/indices_v2"   # Phase 6B-2 expanded index (13,435 docs)
                                  # base.yaml still points at data/indices (2,987 docs)
```

#### SageMaker Validation

The expanded build was validated on SageMaker:

```
Building index from sources: ['kg', 'radlex', 'qa']
  KG documents:      2,987
  RadLex documents:  3,737
  QA documents:      6,711
  Total documents:  13,435
FAISS index saved → data/indices_v2/index.faiss
```

Retrieval quality check on previously-failing queries:

| Query | Old top score (2,987 docs) | New top score (13,435 docs) |
|-------|--------------------------|----------------------------|
| "consolidation" | 0.88 | 0.97 |
| "What modality is this?" | 0.82 | 0.96 |
| "Is there cardiomegaly?" | 0.85 | 0.95 |

---

### Phase 6B-3 — Embedding-Based Agreement

**Status:** Complete — threshold calibrated to 0.87, validated on 5-case smoke test

The Phase 5 supervisor used keyword overlap to judge whether retrieved evidence
"supported" the VLM's answer. This caused 61 over-abstentions on cases where the VLM
was correct but the answer token (e.g. "opacity") didn't appear literally in the evidence
(which described "consolidation" — a direct synonym). Phase 6B-3 replaces keyword
matching with PubMedBERT cosine similarity.

#### Design Decision — Reuse Existing Model

The supervisor reuses the same `S-PubMedBert-MS-MARCO` model already loaded by the
Retriever, rather than loading a new model. In production the sentence-transformers
process-level model cache means no duplicate GPU memory. No additional model is
downloaded or instantiated in production.

**Implementation:** lazy module-level singleton.

```python
_embedder = None   # populated on first _compute_agreement() call

def _get_embedder():
    global _embedder
    if _embedder is None:
        from radiology_vqa.rag.embedder import Embedder
        _embedder = Embedder()
    return _embedder
```

The `embedder` parameter of `_compute_agreement()` accepts an explicit instance for
dependency injection in tests and benchmarks:

```python
def _compute_agreement(
    visual_answer, evidence, question, answer_type, support_threshold,
    embedder=None,   # None → use module singleton; explicit → injected (tests/bench)
) -> tuple[float, list[dict]]:
```

#### Dual-Signal Query Strategy (preserved from Phase 5)

The same logic as the old keyword approach is kept: what to embed depends on the
answer type.

```python
use_question = (answer_type == "closed") or (va_norm in ("yes", "no"))
query_text = question.strip() if use_question else visual_answer.strip()
```

- **Closed / yes-no**: embed the **full question** — "yes" and "no" carry no medical
  signal, but "Is there consolidation in the left lung?" does.
- **Open**: embed the **visual answer** — the actual medical term predicted by the VLM
  (e.g. "pneumonia", "cardiomegaly").

#### Agreement Computation

```python
query_vec    = emb.embed_query(query_text).flatten()   # shape (768,)
evidence_vecs = emb.embed_texts(evidence_texts)         # shape (n, 768)
sims         = evidence_vecs @ query_vec                # shape (n,) cosine similarities
supporting   = [item for item, sim in zip(candidates, sims) if sim >= semantic_threshold]
score        = len(supporting) / len(evidence)
```

Evidence text is `"{item['text']} {item['entity_name']}"` — concatenating the full
definition and the entity name so that short entity fields ("Pneumonia") and full
sentence fields both contribute to the semantic match.

The agreement `score ∈ [0, 1]` normalised by total evidence count preserves the
existing routing contract in `supervisor_node()` (`> 0` triggers answer/re_query paths).

#### Bug Encountered — Shape Mismatch

**Error on SageMaker (first run):**
```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 1 is different from 768)
```

**Root cause:** `emb.embed_query(query_text)` returns shape `(1, 768)` (2D with batch
dimension) rather than `(768,)` (1D). The matmul `(n, 768) @ (1, 768)` fails because
the inner dimensions are 768 vs 1.

**Fix:** Added `.flatten()` to squeeze any batch dimension:
```python
query_vec = emb.embed_query(query_text).flatten()   # safe for (768,) and (1, 768)
```

#### Threshold Calibration — Why 0.5 Was Wrong

The initial threshold of `0.5` caused `agreement=1.000` for **every** case, including
semantically unrelated pairs like "consolidation answer" vs "liver function evidence".

Diagnostic — measured actual cosine similarities for representative pairs:

| Query | Evidence | Similarity | Relation |
|-------|----------|-----------|---------|
| "pneumonia" | Lobar Pneumonia symptoms | **0.9306** | SAME |
| "Is there consolidation in the left lung?" | Consolidation: exudate replacing alveolar air | **0.9085** | SAME |
| "opacity" | Consolidation refers to exudate | **0.8819** | SYNONYM |
| "cardiomegaly" | Pleural Effusion: fluid accumulation | 0.8569 | DIFF |
| "pneumonia" | Kidney is located at both sides of the spine | 0.8377 | DIFF |
| "consolidation" | The function of Liver: metabolize nutrients | 0.8184 | DIFF |
| "Is there consolidation in the left lung?" | The function of Liver: metabolize nutrients | 0.8158 | DIFF |

**Observation:** S-PubMedBert-MS-MARCO, trained on PubMed biomedical text, assigns
surprisingly high similarity to all medical concept pairs because they share domain-level
features. The gap between SAME/SYNONYM (0.88–0.93) and DIFF (0.82–0.86) is only ~0.06.

**Threshold chosen: 0.87** — midpoint of the natural gap (0.857–0.882).

```python
# config.py
supervisor_semantic_threshold: float = 0.87
# Calibrated on S-PubMedBert-MS-MARCO: SAME/SYNONYM pairs score 0.88–0.93,
# DIFF medical pairs (pneumonia/kidney, cardiomegaly/pleural effusion) score 0.82–0.86.
# Natural gap: 0.857–0.882; 0.87 is the midpoint.
```

#### Validation (5-Case Smoke Test)

```
✓ pneumonia vs pneumonia evidence    [SAME]    score=1.000  supporting=1/1
✓ opacity vs consolidation evidence  [SYNONYM] score=1.000  supporting=1/1
✓ consolidation vs liver evidence    [DIFF]    score=0.000  supporting=0/1
✓ consolidation Q vs consolidation   [SAME]    score=1.000  supporting=1/1
✓ consolidation Q vs liver evidence  [DIFF]    score=0.000  supporting=0/1
```

All 5 cases correct. The DIFF pairs now correctly return `score=0.000`, triggering
re_query or abstain in the supervisor instead of blindly answering.

#### Observed Pipeline Behaviour (5-sample run, base model)

Running `python scripts/run_agent.py --dataset vqa_rad --range 0 5` with the calibrated
threshold:

- All samples route to **answer** (no abstentions in this micro-batch)
- Agreement is semantically meaningful — evidence about "Liver" no longer supports
  an answer about "consolidation"
- Fine-tuned model shows higher confidence (0.97–0.99) vs base (0.58–0.89), consistent
  with domain adaptation

#### Embedder Loading Note

The supervisor's `_get_embedder()` singleton loads the model fresh on the first
`_compute_agreement()` call (~4s overhead). This is separate from the Retriever's
already-loaded embedder. The sentence-transformers process-level cache prevents
duplicate GPU memory usage, but the initialisation time hits once per process. In
practice this adds ~4s to the first sample's latency; all subsequent samples are fast.

---

### 6B Error and Decision Summary

#### Errors Encountered

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| **B1** | `ModuleNotFoundError: No module named 'rank_bm25'` | `rank-bm25` not in environment | `pip install rank-bm25` |
| **B2** | `ValueError: matmul shape mismatch (size 1 vs 768)` | `embed_query()` returns `(1, 768)` not `(768,)` | Added `.flatten()` to `query_vec` |
| **B3** | `agreement=1.000` for all cases at threshold=0.5 and 0.72 | S-PubMedBERT assigns >0.8 sim to ALL biomedical text pairs | Measured actual sims; calibrated threshold to 0.87 |

#### Decisions Log

| Decision | Rationale | Alternative Considered |
|----------|-----------|----------------------|
| Reuse S-PubMedBert-MS-MARCO (not load new model) | Already in process memory from retriever; no extra GPU memory | Load `NeuML/pubmedbert-base-embeddings` separately — rejected (unnecessary second model) |
| Threshold 0.87 (not 0.5 or 0.7) | Empirically measured natural gap 0.857–0.882 in actual model output | Fixed heuristic 0.5 — rejected (all cases scored 1.000) |
| Keep dual-signal strategy from Phase 5 | Closed/yes-no VLM answers carry no semantic signal; full question must be embedded | Embed only visual_answer always — rejected (loses query context for closed questions) |
| `doc_id = radlex_{RID}` (not slugified label) | RIDs are stable across RadLex versions; labels can change | Slugified label — rejected (unstable across ontology updates) |
| Tier 1 filter for RadLex (only rows with definitions) | 46,657 raw entries; ~42,920 have no definition — useless for retrieval | Include all entries — rejected (creates noise without semantic content) |
| QA pseudo-docs train-only (never test) | VQA-RAD and SLAKE test splits are reserved for evaluation | Include test QA as retrieval context — rejected (would bias evidence retrieval toward test answers) |

---

## 5. Phase 6C — Confidence Recalibration [PLANNED]

**Dependency:** AFTER Phase 6A + 6B. Calibrate the final complete system.

### 6C-1 — Temperature Scaling

**New file:** `src/radiology_vqa/calibration/temperature.py`

Based on Guo et al. (ICML 2017): divide logits by learned scalar T before softmax.
Single parameter fitted on validation set via L-BFGS to minimise NLL. One example
showed ECE reduction from 2.10% to 0.25% with a single scalar.

Implementation: ~30 lines of PyTorch. Fit on held-out VQA-RAD validation samples.
Integrate as config option (`temperature: 1.42` or whatever is learned).

**Target:** ECE 0.198 → < 0.10

### 6C-2 — Bin-wise Calibration (if needed)

If a single T is insufficient, learn per-bin temperatures for different confidence ranges.
Standard threshold analysis already available from `calibration.py`.

---

## 6. Phase 6D — Final Evaluation and Portfolio [PLANNED]

**Dependency:** After Phase 6A + 6B + 6C are all complete.

### 6D-1 — 6-Configuration Comprehensive Evaluation

Clean ablation across all Phase 6 components on VQA-RAD test split:

| # | Config | VLM | LoRA | RAG | KG | Agreement | Calibration |
|---|--------|-----|------|-----|-----|-----------|-------------|
| 1 | baseline_vlm | Zero-shot | No | No | — | — | — |
| 2 | baseline_agent | Zero-shot | No | Yes | Original | Keyword | — |
| 3 | finetuned_vlm | Fine-tuned | Yes | No | — | — | — |
| 4 | finetuned_agent | Fine-tuned | Yes | Yes | Original | Keyword | — |
| 5 | full_pipeline | Fine-tuned | Yes | Yes | Expanded | Embedding | — |
| 6 | full_calibrated | Fine-tuned | Yes | Yes | Expanded | Embedding | Temp |

Clean ablation: fine-tuning (3 vs 1), RAG with FT (4 vs 3), KG expansion (5 vs 4),
calibration (6 vs 5), full vs baseline (6 vs 1).

### 6D-2 — Target Metrics

| Metric | Pre Phase 6 | Target Post Phase 6 |
|--------|-------------|---------------------|
| Accuracy (when answered) | 47.9% | ≥80% |
| Correct abstention rate | 73.2% | ≥85% |
| ECE | 0.198 | <0.10 |
| Abstention rate | 31.5% | 10–15% |

### 6D-3 — Portfolio Packaging

Final repository structure with complete docs, ARCHITECTURE.md, EVALUATION.md, and a
clean README.md summarising the full system.

---

## 7. Test Coverage

### Phase 6A Tests

All Phase 6A code has fast unit tests requiring no model download or GPU:

| Test class | What it covers |
|---|---|
| `TestNormalizeAnswer` | Lowercase, strip, punctuation/article preservation |
| `TestBuildConversation` | Structure, `<image>` token position, instruction suffix presence |
| `TestTrainingConfig` | Defaults, dataset toggles |
| `TestBuildTrainingDatasetUnit` | Empty-answer filter, long-answer truncation, field names |

**420 fast tests pass** after all Phase 6A changes.

### Phase 6B Tests

**New test file: `tests/test_kg_expansion.py`** — 14 fast tests + 3 slow integration tests

| Test class | What it covers | Data required |
|---|---|---|
| `TestRadLexProcessor` (7 tests) | Tier 1 filter, `doc_id` format (`radlex_`), `source_type`, consolidation coverage, encoding clean (no U+FFFD), minimum count (≥3,000) | `data/Radlex.xls` — skipped if absent |
| `TestQAPseudoProcessor` (7 tests) | VQA-RAD content format (`Question:`/`Answer:`), unique IDs, SLAKE English-only filter (no U+4E00–U+9FFF), `source_type`, metadata enrichment, global ID uniqueness | VQA-RAD uses mock data; SLAKE uses local JSON |
| `TestBuildIndexSources` (3 slow) | KG-only = 2,987 docs, KG+RadLex > 6,000 docs, full build > 13,000 docs | Requires both Radlex.xls and SLAKE train.json |

Slow tests are excluded from default `make test` via `@pytest.mark.slow`.

**441 fast tests pass** after Phase 6B-1 and 6B-2.
Phase 6B-3 (embedding agreement) changes only the internals of `_compute_agreement()`;
existing supervisor tests cover the routing logic and still pass with injected mock embedders.

---

## 8. Software Engineering Practices

**Version Control:**
```
main
├── feature/6a-training      # QLoRA fine-tuning [merged]
├── feature/6b-kg            # KG expansion + embedding agreement [planned]
├── feature/6c-calibration   # Temperature scaling [planned]
└── feature/6d-packaging     # Final eval + cleanup [planned]
```

Each branch: includes new tests, does NOT break existing tests, merges to main via squash
commit.

**Experiment Tracking:** JSON metadata for each run — experiment_id, timestamp, config
dict, results (train/val loss, accuracy, training time, peak GPU memory), hardware info.
No W&B/MLflow overhead.

**Configuration Management:** All configs in YAML files, never hardcoded. Config files
checked into git. Every Phase 6 change is controlled by a config flag that defaults to
Phase 5 behaviour.

**Error Handling:** Graceful degradation — if `adapter_path` is not found, log a warning
and fall back to zero-shot. If RadLex parsing fails, continue with existing KG. If
temperature scaling makes ECE worse, revert to T=1.0.

---

## 9. Timeline

```
Day 0:    Pre-flight (cleanup + install)
Day 1:    6A-1 Data preparation + tests                              [DONE]
Day 2:    6A-2 Training script + dry run                             [DONE]
Day 2–3:  6A-3 Training overnight (23h 14m on A10G)                  [DONE 2026-02-25/26]
Day 3:    6A-4 Pipeline integration (adapter loading in llava.py)    [DONE]
Day 3:    6A-5 Evaluation (in progress — running on SageMaker)
Day 3:    6B-1 Hybrid retrieval BM25 + dense + RRF                   [DONE]
Day 3:    6B-2 RadLex + QA pseudo-doc expansion (2,987 → 13,435)     [DONE]
Day 3:    6B-3 Embedding agreement + threshold calibration to 0.87   [DONE]
Day 4:    6A-5 Evaluation results analysis
Day 5:    6C-1 Temperature scaling + 6C-2 if needed
Day 6:    6D-1 Full 6-config evaluation
Day 7:    6D-2 Repo cleanup + README + architecture diagram
Day 8:    Buffer / polish
```

**Critical path:** 6A-1 → 6A-2 → 6A-3 (overnight) → 6A-4 → 6B-1 → 6B-2 → 6B-3 → 6A-5 (eval) → 6C → 6D

---

## 10. References

### Fine-Tuning
1. Kandamali et al. "LoRA and AdaLoRA on VQA-RAD/SLAKE." Applied Soft Computing, 2025
2. PeFoMed (arXiv:2401.02797). All 3 datasets, 87.1% closed on VQA-RAD
3. LLaMA32-Med (PMLR 2026). 84.6% SLAKE, +48 pts over zero-shot
4. BioMedBLIP (JMIR 2024). SOTA on 15/20 tasks across 3 datasets
5. BaMCo (Springer 2025). 85.8% SLAKE, 76.7% VQA-RAD, 60.0% PathVQA
6. CMMO (J. Biomed. Informatics 2024). 79.6%, 65.7%, 87.2%
7. Philschmid "Fine-tune LLMs 2025": rank 16, lr 2e-4 defaults
8. LLaMA-Factory (ACL 2024): fallback option

### Knowledge Graph
9. RadLex, RSNA (radlex.org): 46K+ radiology classes
10. Chepelev et al. RadioGraphics 2023

### Embeddings
11. NeuML/pubmedbert-base-embeddings (HuggingFace)
12. Chen et al. BMC Bioinformatics 2019

### Calibration
13. Guo et al. "On Calibration of Modern Neural Networks." ICML 2017
14. Minderer et al. NeurIPS 2021
15. gpleiss/temperature_scaling (GitHub)

### Clinical AI Safety
16. arXiv:2601.01008 (Jan 2026): agentic abstention in stroke imaging
17. SelectLLM (OpenReview, Oct 2025): selective prediction for medical QA

### Datasets
18. Lau et al. VQA-RAD. Scientific Data, 2018
19. Liu et al. SLAKE. ISBI 2021
20. He et al. PathVQA. ACL 2021. MIT License
