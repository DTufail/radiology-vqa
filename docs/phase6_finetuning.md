# Phase 6 — QLoRA Fine-Tuning

Phase 6 adapts the base LLaVA-Next v1.6 Mistral-7B model to the medical VQA domain using
parameter-efficient fine-tuning. It is split into two sub-phases:

- **Phase 6A-1** — Dataset cleanup and training dataset package
- **Phase 6A-2** — QLoRA fine-tuning script, pre-training audit, and runtime fixes

---

## Goal

The zero-shot LLaVA-Next model performs reasonably on general VQA but underperforms on
radiology-specific questions (unusual anatomy vocabulary, yes/no closed-form questions,
short concise answers). Phase 6 fine-tunes the model's language layers on three
domain-specific datasets so that inference (Phase 3/4) gets a stronger visual backbone
before the agent pipeline adds RAG grounding on top.

---

## Phase 6A-1 — Dataset Cleanup and Training Package

### Datasets Used

| Dataset   | Split | Samples | Source |
|-----------|-------|---------|--------|
| VQA-RAD   | train | 1,793   | HF `flaviagiammarino/vqa-rad` |
| SLAKE     | train | 4,911   | HF `BoKelvin/SLAKE` (English only, after cleanup) |
| SLAKE     | val   | 1,053   | same (used as validation set) |
| PathVQA   | train | 19,654  | HF `flaviagiammarino/path-vqa` |
| **Total** | train | **26,358** | VQA-RAD + SLAKE + PathVQA combined |

The VQA-RAD **test** split is never touched — it is reserved for the Phase 5 evaluation
run so that we can compare zero-shot vs fine-tuned on a clean held-out set.

SLAKE validation is used as the fine-tuning validation set because it is already a
dedicated held-out split (unlike VQA-RAD and PathVQA which have no standard validation
split), and because SLAKE contains the most medically grounded questions.

### Cleanup 1 — SLAKE Whitespace Filter

**Problem**: SLAKE had one sample (`qid=1622`) with an empty answer after stripping
whitespace. An empty answer string would pass Pydantic validation on the training side but
produce a target sequence of length zero, which causes a loss of `nan` in cross-entropy.

**Fix**: `slake_loader.py` already had this filter from a previous session (whitespace
stripping + skip if empty). The Phase 6A-1 audit confirmed it was active and working.

### Cleanup 2 — VQA-RAD Leakage Check

**Problem**: If (image, question, answer) triples from the VQA-RAD test split appeared in
training, accuracy numbers on the test split would be inflated.

**Approach**: Computed perceptual image hashes (MD5 of raw bytes) for all VQA-RAD train and
test images. Found 42 (question, answer) string matches across train/test, but **0** of those
shared the same image. Conclusion: CLEAN — no true leakage.

**Decision**: Do not add any filtering. The 42 question-answer string coincidences are
expected in medical VQA ("Is this a chest X-ray?" / "yes" appears in both splits on
completely different images).

### Cleanup 3 — PathVQA Image Corruption Scan

**Problem**: PathVQA contains 19,654 images from mixed sources including histopathology and
gross pathology. Corrupt or non-RGB images would crash the collator silently in a DataLoader
worker, producing an unhelpful error mid-training after hours of progress.

**Approach**: Ran a full scan on SageMaker (not locally to avoid OOM — 19k images × ~450 KB
is ~8.8 GB). Results:
- 0 corrupt images
- 1,503 non-RGB (grayscale `L` mode) — handled by `.convert("RGB")` in the loader
- 1 truncated TIFF warning (non-fatal, PIL reads it anyway)

**Decision**: No images removed. The `.convert("RGB")` call in the PathVQA loading path
handles grayscale transparently.

### Cleanup 4 — Final Count Verification

After all cleanups, verified total counts match expectations:
- VQA-RAD train: 1,793 ✓
- SLAKE train (EN): 4,919 raw → -1 empty → -7 duplicates = **4,911** ✓
- PathVQA train: 19,654 ✓
- **Combined: 26,358** ✓

### Training Package — `src/radiology_vqa/training/`

#### `dataset.py`

**`TrainingConfig`** — a dataclass (not a Pydantic model) for training-time configuration.
Kept separate from `SystemConfig` to avoid polluting the inference config with training
hyperparameters.

**`normalize_answer(answer)`** — lowercase + strip only. Deliberately does NOT remove
articles or punctuation, unlike the evaluation normaliser in
`src/radiology_vqa/evaluation/metrics.py`. Rationale: training targets should be natural
language that the model can learn; over-normalising (e.g. removing "the") would teach the
model to produce answers that score well on exact-match but sound unnatural to clinicians.

**`build_conversation(question, answer)`** — returns a `list[dict]` in the `[{"role":
"user", ...}, {"role": "assistant", ...}]` format. The user content is:

```
<image>
{question}
Provide only the direct answer in 1-5 words. Do not explain.
```

The `<image>` token is a placeholder that the collator replaces with image embeddings.
The instruction suffix is critical — it must match the `concise_mode` inference prompt in
`llava.py` exactly, otherwise the model is trained on one distribution and tested on
another. This was identified as audit issue **M1** and fixed before training.

The content is stored as a plain string (not as a list of typed blocks) so that the
existing fast unit tests (which check string membership) keep passing. The collator
upgrades to the proper typed-block format at collation time (see collator section below).

**`build_training_dataset(config, slake_dir)`** — orchestrates loading all three datasets
and combining them. PathVQA is loaded via `Dataset.from_generator()` rather than
`Dataset.from_list()` because accumulating all 19,654 PIL images in Python heap before
Arrow serialisation would spike RAM by ~8.8 GB. Generator-based construction yields rows
one at a time directly into the Arrow buffer. The VQA-RAD + SLAKE portion (small —
~6,700 samples) is built with `Dataset.from_list()` and then merged with PathVQA via
`concatenate_datasets()`. This was audit issue **M2**.

#### `collator.py`

**`LlavaDataCollator`** applies the processor to convert `(image, conversations)` pairs
into model inputs. Two design decisions are worth noting:

**Content format conversion** (`_to_multimodal_format`): Transformers ≥4.45 changed the
LLaVA-Next Jinja2 chat template to expect content as a list of typed blocks:
```python
[{"type": "image"}, {"type": "text", "text": "..."}]
```
rather than a plain string. Passing the plain string format causes:
```
jinja2.exceptions.UndefinedError: 'str object' has no attribute 'text'
```
The `_to_multimodal_format()` helper converts at collation time, keeping
`build_conversation()` output format stable for tests.

**`max_length: 4096`**: LLaVA-Next v1.6 Mistral uses any-resolution (AnyRes) image tiling.
For a large radiology image, the processor can produce up to 4 tiles × 576 tokens + 1 base
image × 576 tokens = **2,880 image tokens**. Adding question/answer/template text (~100–200
tokens) gives a maximum of ~3,080 tokens per sample. Setting `max_length=4096` (the
Mistral-7B context window) ensures no sample is truncated. Truncation at a smaller value
causes a hard error because the processor validates that the number of `<image>` tokens in
`input_ids` matches the number in the text template. This was confirmed against:
- HF Transformers docs (explicit warning about not truncating image tokens)
- Real-world LLaVA-Next fine-tuning configs using `model_max_length: 4096`
- GitHub transformers issue #36002

---

## Phase 6A-2 — QLoRA Fine-Tuning Script

### Why QLoRA

A full fine-tune of LLaVA-Next 7B requires ~56 GB VRAM (fp16) — impossible on a T4 (15
GB). QLoRA addresses this with two techniques:

1. **4-bit NF4 quantisation** (bitsandbytes): freezes all base model weights in 4-bit,
   reducing the 7B model from ~14 GB (fp16) to ~4 GB. The stored weights are quantised
   but computations are done in fp16 via the `bnb_4bit_compute_dtype` setting.

2. **LoRA adapters** (PEFT): adds small rank-16 adapter matrices to the language model
   linear layers only. Only these adapters (~44M parameters, 1.1% of total) are
   trainable. The vision encoder is completely frozen.

Together, QLoRA trains only the adapters while keeping the base model frozen in 4-bit,
fitting the entire setup in ~4 GB of model VRAM with ~6–8 GB peak during backpropagation.

### Why these LoRA Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `rank` | 16 | Community standard for domain adaptation. Rank 8 is too low for VQA (needs to learn new vocabulary and reasoning patterns); rank 32+ uses more VRAM without proportional gain. |
| `alpha` | 32 | `alpha = 2 × rank` is the standard scaling convention. Higher alpha amplifies adapter outputs. |
| `dropout` | 0.05 | Light regularisation. Medical VQA datasets are relatively small; without dropout, rank-16 adapters can memorise training answers. |
| `target_modules` | q/k/v/o/gate/up/down_proj | All language model attention and MLP linear layers. Vision tower excluded — its representations are already good enough for image understanding; fine-tuning it would require much more data to avoid catastrophic forgetting. |

### Why these Training Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning. Higher than typical full-fine-tune LRs because adapters are initialised randomly and need to move quickly. |
| `num_epochs` | 3 | Balances convergence vs overfitting on a 26k-sample dataset. Fewer epochs risk underfitting; more epochs on this dataset size risk memorisation. |
| `per_device_batch_size` | 1 | Forced by T4 VRAM constraints with 4096-token sequences. |
| `gradient_accumulation_steps` | 8 | Effective batch size = 8. Too small a batch (1) gives noisy gradients; 8 provides stable training without VRAM cost of a real batch-8. |
| `optim` | `paged_adamw_8bit` | Offloads Adam optimizer states (typically ~2× model size) to CPU RAM in 8-bit format. Saves ~2 GB VRAM compared to standard AdamW. |
| `gradient_checkpointing` | true | Recomputes layer activations during backward pass instead of storing them. Trades ~30% compute overhead for ~60% reduction in activation memory. Mandatory on T4. |
| `fp16` | true | T4 is Volta architecture — no BFloat16 support. fp16 is the appropriate mixed-precision mode. |
| `lr_scheduler` | cosine | Standard for instruction fine-tuning. Cosine decay provides a smooth LR reduction after warmup. |
| `warmup_ratio` | 0.03 | 3% of total steps. Short warmup because adapters start from random init and need to reach a useful regime quickly. |
| `max_grad_norm` | 0.3 | Aggressive gradient clipping. Prevents large adapter updates early in training when gradients are noisy. |

### Script Architecture (`scripts/finetune_qlora.py`)

The script is structured as a sequence of named phases with structured logging so that
progress can be monitored via `tail -f` even if the terminal session drops overnight.

**Initialisation order (critical)**:
```
load model in 4-bit
  → prepare_model_for_kbit_training()   # enables gradient checkpointing on quantised layers
  → get_peft_model()                     # wraps with LoRA
```
The order matters: `prepare_model_for_kbit_training` must run before `get_peft_model`,
otherwise gradient checkpointing is applied to the PEFT wrapper rather than the base model,
and backprop through the quantised layers fails.

**Dual logging** (`setup_logging`): Both stdout and a timestamped file
`logs/finetune_{timestamp}.log` receive all log output via separate handlers on the root
logger. This means `tail -f logs/finetune_*.log` works to monitor progress even after a
tmux detach.

**`--dry-run` flag**: Loads the model, applies LoRA, runs the collator on 2 dummy samples,
prints batch shapes, and exits with code 0. Runs in ~3 minutes and validates the entire
setup without writing any checkpoint.

**`--max-steps N` flag**: Smoke-test mode. Runs N gradient steps on VQA-RAD + SLAKE only
(PathVQA download skipped), with eval and checkpointing disabled. Used to validate that
backpropagation produces a finite loss without OOM before committing to an overnight run.

### Pre-Training Audit Findings and Fixes

A 30-point read-only audit of the code before the first training run identified 4 issues:

#### M1 — Training/Inference Format Mismatch (MEDIUM)

**Problem**: `build_conversation()` originally produced:
```
<image>
{question}
```
while `llava.py`'s `concise_mode` inference prompt is:
```
<image>
{question}
Provide only the direct answer in 1-5 words. Do not explain.
```
Training on one format and inferring on another creates a distribution shift — the model
learns to produce concise answers only when there is no suffix, but at inference the suffix
is always present.

**Fix**: Added the instruction suffix to `build_conversation()`. The `list[dict]` return
type was preserved to keep all 21 unit tests passing. The collator handles the conversion
to typed-block format separately.

#### M2 — PathVQA Heap Accumulation (MEDIUM)

**Problem**: `Dataset.from_list()` requires all samples to be in Python heap before Arrow
serialisation. For 19,654 pathology images this is ~8.8 GB, which could OOM on
ml.g4dn.2xlarge (32 GB CPU RAM) when combined with model weights during training.

**Fix**: Replaced with `Dataset.from_generator()` + `concatenate_datasets()`. The generator
yields rows one at a time; Arrow serialises them immediately and frees the PIL image from
Python heap.

#### M3 — eval_loss Missing from Experiment Record (MEDIUM)

**Problem**: `TrainOutput.training_loss` captures only train loss. The experiment JSON had
no eval_loss or perplexity, making it impossible to detect overfitting from the record
alone.

**Fix**: Added `eval_result = trainer.evaluate()` after `trainer.train()`. The
`save_experiment_record()` function now accepts `eval_result` and writes `eval_loss` and
`perplexity = exp(eval_loss)` to the JSON. An overfitting warning is emitted if
`eval_loss > train_loss × 1.5`.

#### M4 — Checkpoint Path Mismatch (MEDIUM / SHOW-STOPPER)

**Problem**: The script saved the adapter to `checkpoints/llava-med-qlora/` but
`configs/phase6.yaml` referenced `checkpoints/llava-med-qlora/best`. Loading the adapter
after training would silently fall back to zero-shot mode with a logged warning.

**Fix**: Changed to `trainer.save_model(os.path.join(output_dir, "best"))` and also called
`processor.save_pretrained(best_model_dir)`. Both the adapter weights and the processor
(tokenizer + image processor) are co-located under `best/` so that `PeftModel.from_pretrained`
finds everything it needs at the path `phase6.yaml` specifies.

### Runtime Fixes During Smoke Testing

Two additional issues were discovered during actual smoke tests (not dry-run), because the
dry-run uses a synthetic 336×336 image while real training uses heterogeneous radiology images.

#### R1 — Jinja2 `'str object' has no attribute 'text'`

**Root cause**: Transformers ≥4.45 changed the LLaVA-Next Jinja2 chat template to iterate
over content items and access `item['text']`. The old format (content as a plain string)
causes the template to iterate over individual characters.

**Fix**: Added `_to_multimodal_format()` in `collator.py` that converts string content to
`[{"type": "image"}, {"type": "text", "text": "..."}]` at collation time, keeping
`build_conversation()` format stable for tests.

#### R2 — `max_length` Truncation Mismatch

**Root cause**: The dry-run dummy image (336×336) produces 1,215 tokens. SLAKE contains
large CT/MRI scans that trigger AnyRes tiling up to 4 tiles + 1 base = 2,880 image tokens.
Starting with `max_length=256` and then `max_length=2048` both fell short. The processor
validates that the count of `<image>` placeholders in `input_ids` matches the count in the
text template; truncation breaks this invariant with a hard `ValueError`.

**Fix**: Set `max_length=4096` (Mistral-7B's native context window). This covers the
theoretical maximum of 2,880 image tokens + ~200 text tokens = ~3,080 with a 1,000-token
safety margin. Confirmed against HF Transformers documentation and real-world LLaVA-Next
fine-tuning configurations, which universally use `model_max_length: 4096` for this model.

---

## Checkpoint Structure After Training

```
checkpoints/llava-med-qlora/
├── best/                          ← referenced by configs/phase6.yaml
│   ├── adapter_config.json        ← LoRA hyperparameters
│   ├── adapter_model.safetensors  ← trained adapter weights (~170 MB)
│   ├── tokenizer.json             ← processor files co-located for easy loading
│   ├── tokenizer_config.json
│   └── preprocessor_config.json
├── checkpoint-838/                ← epoch 1 checkpoint (save_total_limit=2 keeps last 2)
└── checkpoint-1676/               ← epoch 2 checkpoint
```

At inference time, `llava.py` loads the base model and then:
```python
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
model = model.merge_and_unload()   # folds adapter into base weights, no runtime overhead
```

Setting `adapter_path: ""` (empty string) in `configs/phase6.yaml` falls back to zero-shot
mode. This allows A/B testing between zero-shot and fine-tuned on the same evaluation run.

---

## Experiment Record

After every training run (successful or failed), `logs/experiment_{timestamp}.json` is
written with:
- Full config dump
- Trainable / total parameter counts
- `train_loss`, `eval_loss`, `perplexity`
- GPU name and total VRAM
- `experiment_id` for cross-referencing with `logs/finetune_{timestamp}.log`

This provides a full audit trail without requiring Weights & Biases or MLflow.

---

## Test Coverage

All Phase 6A code has fast unit tests requiring no model download or GPU:

| Test class | What it covers |
|---|---|
| `TestNormalizeAnswer` | Lowercase, strip, punctuation/article preservation |
| `TestBuildConversation` | Structure, `<image>` token position, instruction suffix presence |
| `TestTrainingConfig` | Defaults, dataset toggles |
| `TestBuildTrainingDatasetUnit` | Empty-answer filter, long-answer truncation, field names |

420 fast tests pass after all Phase 6A changes. The collator tests use a `MockProcessor`
that never calls `apply_chat_template`, so the Jinja2 format change does not affect them.
