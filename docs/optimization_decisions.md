# Optimization Decisions — Phase 3 Backends & Data Pipeline

This document records every optimization made to the inference backends and data-loading
pipeline, the problem each one solves, and why the chosen fix is the right one.

---

## 1. `src/radiology_vqa/loader.py`

### 1.1 RGB guard on `load_vqa_rad`

**Problem**
VQA-RAD images are sourced from real DICOM studies. Some frames are grayscale (`mode="L"`)
or RGBA. `LlavaForConditionalGeneration` and `Blip2ForConditionalGeneration` both expect
3-channel RGB tensors. Feeding a grayscale image produces a shape mismatch inside the
processor that raises at runtime with a confusing error unrelated to the root cause.

`load_pathvqa` already called `.convert("RGB")` unconditionally. `load_vqa_rad` did not,
creating an inconsistency that only surfaced for non-RGB source images.

**Fix**
```python
if image.mode != "RGB":
    image = image.convert("RGB")
```
The guard is conditional rather than unconditional (like PathVQA) because most VQA-RAD
frames already arrive as RGB — skipping the no-op conversion avoids creating an unnecessary
copy for the common path.

---

### 1.2 `raise` instead of `return []` on exception

**Problem**
The original `except` block logged the error and returned an empty list. An empty list
is indistinguishable from a legitimate zero-sample split; downstream code (the benchmark
runner, tests) would silently evaluate nothing and report 0% accuracy with no warning
that data loading had failed.

**Fix**
```python
except Exception as e:
    logger.error("Failed to load VQA-RAD %s: %s", split, e)
    raise
```
Re-raising propagates the original exception with its full traceback. The caller decides
whether to handle or abort — not the loader. This matches standard library conventions
(`json.load`, `open`, etc.) and makes failures impossible to miss.

---

## 2. `src/radiology_vqa/slake_loader.py`

### 2.1 `raise` on missing or malformed split file

**Problem**
The original code checked `if not json_path.exists(): return []` and
`except (json.JSONDecodeError, OSError): return []`. Both silently returned an empty
list, hiding the fact that the entire split was unavailable. A benchmark run against an
empty list completes instantly with 0% accuracy and no diagnostics.

**Fix**
```python
if not json_path.exists():
    raise FileNotFoundError(f"SLAKE split file not found: {json_path}. ...")

try:
    with open(json_path, encoding="utf-8") as f:
        rows = json.load(f)
except (json.JSONDecodeError, OSError) as e:
    raise RuntimeError(f"Failed to parse SLAKE JSON {json_path}: {e}") from e
```
Missing individual images within a valid split still use `continue` (skip the sample)
because that is tested behaviour — SLAKE contains some missing images by design.
The split file itself missing is always a configuration error.

---

### 2.2 Module-level image cache

**Problem**
SLAKE has ~642 English questions but only ~222 unique images. Multiple QA pairs share
the same image. The original code opened and decoded each image on every `load_slake()`
call. A typical workflow loads train → test → validate sequentially; images shared
across splits were decoded 3× from disk.

**Fix**
```python
# Module-level: persists across load_slake() calls
_IMAGE_CACHE: dict[str, Image.Image] = {}

# Inside load_slake():
cache_key = str(img_path.resolve())
if cache_key in _IMAGE_CACHE:
    image = _IMAGE_CACHE[cache_key]
else:
    image = Image.open(img_path).convert("RGB")
    _IMAGE_CACHE[cache_key] = image
```
**Why module-level, not function-local?**
A function-local cache is garbage-collected after each call. The cache only helps if it
persists across calls. Module-level state survives the lifetime of the Python process,
which is exactly the scope needed for a multi-split evaluation loop.

**Why keyed by resolved absolute path?**
Different callers may pass different relative paths to the same physical file. Using
`img_path.resolve()` normalises symlinks and `..` components, so each image is decoded
at most once regardless of how the path was expressed.

**Safety note**: PIL images are shared by reference. Callers that modify an image
in-place (e.g., resize, crop) must call `image.copy()` first to avoid corrupting the
cache entry for other samples.

---

## 3. `scripts/download_datasets.py`

### 3.1 Direct `load_dataset()` instead of materialising VQASample objects

**Problem**
The original download script called `load_vqa_rad()` and `load_pathvqa()`, which decode
every image into a PIL object and build full `VQASample` instances. For VQA-RAD test
(451 samples) and PathVQA train (18,000+ samples) this consumes several gigabytes of
RAM during download, even though the purpose of the script is only to warm the HuggingFace
cache on disk.

**Fix**
```python
dataset = load_dataset(settings.vqa_rad_dataset, split=split)
_check_row(dataset[0], "VQA-RAD", split)
```
`load_dataset()` triggers the HF download and caching without decoding images into Python
objects. The dataset is accessed only to sanity-check the first row, not iterated.

---

### 3.2 Parallel downloads via `ThreadPoolExecutor`

**Problem**
VQA-RAD and PathVQA are hosted on independent HuggingFace endpoints. Downloading them
sequentially leaves one connection idle while the other transfers.

**Fix**
```python
with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
    futures = {pool.submit(fn): name for name, fn in tasks.items()}
    for future in as_completed(futures):
        ...
```
Both downloads run concurrently. Wall-clock time drops roughly proportionally to the
slower download dominating. `ThreadPoolExecutor` rather than `ProcessPoolExecutor`
is correct here because the bottleneck is network I/O (GIL-releasing), not CPU.

---

### 3.3 `_check_row()` function replaces `assert`

**Problem**
```python
assert s.question and s.answer
```
`assert` statements are disabled when Python runs with the `-O` (optimise) flag, making
the sanity check silently disappear in production or CI environments that use `-O`.

**Fix**
```python
def _check_row(row: dict, dataset: str, split: str) -> None:
    if not row.get("question") or not row.get("answer"):
        raise RuntimeError(
            f"Sanity check failed for {dataset} {split}: "
            "first row missing question or answer fields."
        )
```
`RuntimeError` is always raised regardless of interpreter optimisation flags, and the
message names the specific dataset and split, making the failure immediately actionable.

---

## 4. `src/radiology_vqa/vlm/llava_med.py`

### 4.1 Cache resolved device at load time

**Problem**
```python
# Original — called on every predict() invocation:
device = next(self._model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}
```
`next(self._model.parameters())` walks the module's parameter iterator every call. For
a 7B-parameter model with a long parameter list, this adds a non-trivial CPU-side scan
before every forward pass. In a 451-sample benchmark this is called 451 times.

**Fix**
```python
# __init__, called once after model load:
self._inferred_device: torch.device = next(self._model.parameters()).device

# predict() uses the cached attribute:
inputs = {k: v.to(self._inferred_device) for k, v in inputs.items()}
```
The model's device cannot change after `eval()` is called (no `.to()` is called
downstream), so caching is safe. The attribute name `_inferred_device` signals that
this was determined by inspecting the loaded weights, not assumed.

---

### 4.2 `torch.inference_mode()` instead of `torch.no_grad()`

**Problem**
`torch.no_grad()` disables gradient computation but still performs autograd version
tracking on tensors. For pure inference this tracking is unnecessary overhead.

**Fix**
```python
with torch.inference_mode():
    output = self._model.generate(...)
```
`torch.inference_mode()` disables both gradient computation *and* version tracking.
PyTorch documentation explicitly states it is the preferred context for inference:
it is strictly more efficient. The trade-off is that tensors created inside it cannot
be used in autograd graphs later — which is exactly what we want for inference-only code.

---

### 4.3 Batched confidence extraction

**Problem**
The original confidence calculation used a Python loop with one `softmax` call per
generated token:
```python
confidence = 0.0
for t, score in enumerate(scores):
    probs = torch.softmax(score[0], dim=-1)
    confidence += probs[generated_ids[t]].item()
confidence /= len(scores)
```
For a 128-token response this is 128 sequential GPU→CPU round-trips. Each `.item()`
call synchronises the CUDA stream, causing the GPU to stall.

**Fix**
```python
scores_tensor = torch.stack(scores).squeeze(1)  # (T, vocab_size)
probs = torch.softmax(scores_tensor, dim=-1)     # (T, vocab_size) — one kernel launch
token_probs = probs[
    torch.arange(len(scores), device=probs.device),
    generated_ids[: len(scores)],
]  # (T,) — one gather kernel
return token_probs.mean().item()                 # one synchronisation point
```
All token probabilities are computed in two fused GPU operations (softmax + gather),
then the mean reduces to a single scalar, triggering exactly one GPU→CPU sync. This
replaces T softmax kernel launches and T `item()` syncs with 2 kernels + 1 sync.

---

## 5. `src/radiology_vqa/vlm/blip2.py`

### 5.1 Cache resolved device at load time

**Problem and Fix**: Identical to §4.1 for LLaVA-Med. BLIP-2 had the same
`next(self._model.parameters()).device` call inside `predict()` on every invocation.

```python
# __init__, after model.eval():
self._inferred_device: torch.device = next(self._model.parameters()).device
logger.info("BLIP-2 loaded on device=%s.", self._inferred_device)
```

---

### 5.2 `torch.inference_mode()` instead of `torch.no_grad()`

**Problem and Fix**: Identical to §4.2. Eliminates unnecessary autograd version
tracking during BLIP-2 forward passes.

---

### 5.3 Fix `input_ids` access pattern

**Problem**
```python
# Original:
input_len = inputs.get("input_ids", torch.tensor([[]])).shape[1]
```
`inputs.get("input_ids", torch.tensor([[]]))`  creates a new CPU tensor on every call
as the default value, even though `input_ids` is always present in the processor output.
The `.get()` with a default tensor is a defensive pattern that provides no real safety —
if `input_ids` were missing the decode would be wrong regardless.

**Fix**
```python
input_len = inputs["input_ids"].shape[1]
```
Direct key access. Raises `KeyError` if the processor ever stops returning `input_ids`,
which is the correct failure mode (loud error vs. silent wrong decode).

---

### 5.4 Remove answer lowercasing

**Problem**
```python
# Original:
answer = raw_output.lower().strip() or "unknown"
```
LLaVA-Med did not lowercase; BLIP-2 did. This created inconsistency:
- `normalize_answer()` in `metrics.py` already applies `.lower()` before comparison
- Storing a pre-lowercased answer in `VLMPrediction.answer` makes the raw output
  unrecoverable and breaks any downstream code that reads predictions for display

**Fix**
```python
answer = raw_output.strip() or "unknown"
```
`answer` stores the model's actual output, capitalisation preserved. Normalisation for
comparison is the exclusive responsibility of `metrics.normalize_answer()`. This matches
LLaVA-Med's behaviour and the single-responsibility principle.

---

### 5.5 Real confidence extraction replacing hardcoded `0.5`

**Problem**
```python
# Original:
confidence=0.5,  # BLIP-2 generate() doesn't expose usable logprobs
```
The comment was incorrect. `Blip2ForConditionalGeneration.generate()` does support
`output_scores=True, return_dict_in_generate=True` — the same API used by LLaVA-Med.
A hardcoded `0.5` for every sample makes the `confidence` field useless for ranking,
thresholding, or any downstream calibration work.

**Fix**
```python
# In _infer_batch():
output = self._model.generate(
    **inputs,
    max_new_tokens=self._max_new_tokens,
    output_scores=True,
    return_dict_in_generate=True,
)

# _extract_confidence() — same batched approach as LLaVA-Med but handles batch dim:
scores_tensor = torch.stack(scores)           # (T, batch_size, vocab_size)
sample_scores = scores_tensor[:, sample_idx, :]   # (T, vocab_size)
probs = torch.softmax(sample_scores, dim=-1)
token_probs = probs[torch.arange(len(scores)), generated_ids[:len(scores)]]
return token_probs.mean().item()
```
The key difference from LLaVA-Med's extractor is the batch dimension: BLIP-2's score
tensors are `(batch_size, vocab_size)` per step (not `(1, vocab_size)`), so `sample_idx`
selects the correct row before the softmax. The `.squeeze(1)` from llava_med.py is
replaced by explicit indexing to correctly support batch_size > 1.

---

### 5.6 True batching via `_infer_batch()` + chunked `predict_batch()`

**Problem**
```python
# Original predict_batch():
return [self.predict(image, question) for image, question in samples]
```
Each call to `predict()` launched a separate forward pass, with:
- Per-call processor overhead (tokenisation, image feature extraction)
- Per-call GPU kernel launch latency
- No GPU utilisation between samples while the CPU prepared the next input

BLIP-2's `Blip2Processor` natively supports batched inputs with padding, making
it straightforward to process multiple samples in a single forward pass.

**Fix**
```python
def predict(self, image, question):
    """Delegate to _infer_batch to eliminate code duplication."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return self._infer_batch([image], [f"Question: {question} Answer:"])[0]

def _infer_batch(self, images, prompts):
    """Single forward pass for N images."""
    inputs = self._processor(
        images=images, text=prompts, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(self._inferred_device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = self._model.generate(...)
    # Decode each sample from output.sequences[idx, input_len:]

def predict_batch(self, samples, batch_size=8):
    """Chunk samples and call _infer_batch per chunk."""
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        results.extend(self._infer_batch(images, prompts))
```

**Why `batch_size=8` as default?**
BLIP-2 OPT-2.7B in 8-bit quantisation uses ~3.5 GB VRAM. On a 16 GB GPU (T4/A10),
batch_size=8 adds roughly 1 GB of activation memory — well within budget. Larger
batches reduce per-sample latency up to the point where memory bandwidth saturates.
The default is conservative; users on A100/H100 may increase to 16–32.

**Why `_infer_batch()` as a private method?**
Both `predict()` (batch_size=1) and `predict_batch()` (batch_size=N) need the same
forward-pass logic. Extracting it prevents duplication and ensures that any future
changes to the generation call (e.g., adding `temperature`, `top_p`) are applied
in one place.

**Latency reporting with batching**
Each `VLMPrediction.latency_seconds` is set to `batch_latency / batch_size`, so the
benchmark runner's per-sample latency stats remain meaningful. The total benchmark
wall time equals `sum(batch_latency_for_each_chunk)`, which is the true elapsed time.

---

## 6. `src/radiology_vqa/benchmark/runner.py`

### 6.1 `batch_size` parameter on `run()`

**Problem**
The runner called `self._vlm.predict()` in a Python loop, precluding batched inference
regardless of backend capability. A BLIP-2 benchmark with `batch_size=8` should be
~5× faster than sequential predict calls; the runner had no way to express this.

**Fix**
```python
def run(self, samples, dataset_name, split, max_samples=None, batch_size=1):
    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start : chunk_start + batch_size]
        if batch_size == 1:
            predictions = [self._vlm.predict(chunk[0].image, chunk[0].question)]
        else:
            predictions = self._vlm.predict_batch([(s.image, s.question) for s in chunk])
        # per-sample record building is identical for both paths
```

**Why keep `predict()` for batch_size=1?**
`predict()` is the canonical single-sample interface defined in `VLMInterface`. Using
it for the default case avoids calling `predict_batch([(image, question)])` which would
work but feels semantically wrong and could confuse protocol implementations that have
optimised only one path.

**Backward compatibility**: `batch_size` defaults to `1`. All existing call sites and
tests that do not pass `batch_size` continue to behave identically.

**Progress logging**: The `done % 50 < batch_size` condition fires when the count of
completed samples crosses a multiple-of-50 boundary within a chunk. This correctly
handles batch sizes larger than 50 and ensures the final sample count is always logged.

**`config` dict**: `batch_size` is stored in the saved JSON result under `config`, so
benchmark outputs are self-describing and can be compared across runs with different
batch sizes.
