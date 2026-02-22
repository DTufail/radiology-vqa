# LLaVA-Med Loading Fixes: Problem Log and Solutions

This document records every runtime failure encountered when loading
`microsoft/llava-med-v1.5-mistral-7b` on a SageMaker T4 instance
(16 GB VRAM, Python 3.12, transformers 5.x) and the fix applied for each.

---

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA T4 · 16 GB VRAM (14.56 GB usable) |
| CPU RAM | 70 GB total · ~19 GB free at load time |
| Python | 3.12 |
| transformers | 5.x (latest) |
| accelerate | installed via `pip install -e ".[quant]"` |
| bitsandbytes | installed |
| Checkpoint | `microsoft/llava-med-v1.5-mistral-7b` (written for transformers 4.36.2) |

---

## Fix 1 — Missing `sentencepiece` package

### Error
```
ValueError: Error parsing line b'\x0e' in tokenizer.model
```

### Root cause
The Mistral-7B tokenizer uses a SentencePiece `.model` file.
`transformers` tries to convert it to the fast Rust tokenizer format using
`SentencePieceExtractor`, which requires the standalone `protobuf` package.
When `protobuf` is missing, `transformers` silently falls back to TikToken,
which cannot parse binary SentencePiece files and crashes.

The `sentencepiece` Python package itself bundles its own protobuf for model
loading, but `transformers`'s fast-tokenizer converter needs the *standalone*
`protobuf` package separately.

### Fix
1. Added `sentencepiece>=0.1.99` to `pyproject.toml` core dependencies.
2. Added `protobuf>=3.20` to `pyproject.toml` core dependencies.
3. Added `use_fast=False` fallback in `_load_processor` Tier 3: if the fast
   tokenizer conversion still fails (e.g., missing protobuf), the slow tokenizer
   uses the sentencepiece Python binding's own bundled protobuf — no standalone
   protobuf required.

### Files changed
- `pyproject.toml` — added `sentencepiece>=0.1.99`, `protobuf>=3.20`
- `src/radiology_vqa/vlm/llava_med.py` — `_load_processor` Tier 3 fallback

---

## Fix 2 — CUDA out of memory during weight remapping

### Error
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 24.81 MiB is free.
```
Traceback pointed to `_apply_remapped_weights` at `val.to(dtype=dtype, device=device)`.

### Root cause
Two compounding issues:

**Issue A — All 4 shards loaded into CPU RAM simultaneously.**
The original `_load_checkpoint_state_dict` read every safetensors shard into a
single Python dict before any transfer occurred:
```python
state_dict: dict[str, Tensor] = {}
for shard in shard_files:
    for key in shard.keys():
        state_dict[key] = shard.get_tensor(key)   # 14 GB in RAM at once
```

**Issue B — GPU already full before remapping started.**
`from_pretrained(device_map="auto", torch_dtype=float16)` with ALL MISSING
checkpoint keys (old flat format ≠ new nested format) falls back to random
initialisation directly on the GPU.  With fp16 Mistral-7B (~14 GB) the T4
had only ~24 MB free before the remapping loop even began.

### Fix
Rewrote `_apply_remapped_weights` to stream one shard (~3.5 GB) at a time and
use `accelerate.utils.set_module_tensor_to_device` for **in-place** GPU
replacement:
- Old (random) tensor on GPU is freed.
- New (checkpoint) tensor is allocated on the same device.
- Net VRAM change ≈ 0 per parameter.
- Peak CPU RAM ≈ 3.5 GB (one shard), not 14 GB.

### Files changed
- `src/radiology_vqa/vlm/llava_med.py` — `_apply_remapped_weights` rewritten;
  `_remap_key()` and `_set_param_direct()` helpers added.

---

## Fix 3 — Shape mismatch: `[4096, 14336]` vs `[4096, 11008]`

### Error
```
ValueError: Trying to set a tensor of shape torch.Size([4096, 14336]) in
"weight" (which has shape torch.Size([4096, 11008])), this looks incorrect.
```

### Root cause
**transformers 5.x silently remaps `model_type="llava_mistral"` to `"llava"`**
before returning from `AutoConfig.from_pretrained`.  The generic `"llava"`
fallback creates a default `LlamaConfig` with `intermediate_size=11008` (the
LLaMA-7B default).  Every FFN weight in the checkpoint has the Mistral-7B size
of 14336, so every tensor write fails with a shape mismatch.

**The LLaVA-Med `config.json` uses a flat format** (written for transformers
4.36.2): all Mistral-7B fields (`hidden_size`, `intermediate_size`, etc.) sit
at the top level alongside `mm_*` LLaVA fields, with no nested `text_config`
sub-object.  Our first attempt to detect `model_type == "llava_mistral"` via
`AutoConfig` failed because transformers had already remapped the type before
returning.

### Fix
When `old_format=True` (detected from the shard index, not from `AutoConfig`):

1. Read `config.json` directly as raw JSON via `hf_hub_download` + `json.load`,
   bypassing `AutoConfig`'s type remapping entirely.
2. Build a typed `MistralConfig` object with the correct dimensions extracted
   from the raw JSON (`intermediate_size=14336`, `num_key_value_heads=8`, etc.).
3. Wrap it in `LlavaConfig(text_config=mistral_cfg, vision_config=clip_cfg)`.
4. Pass as `config=` to `from_pretrained` (or to `LlavaForConditionalGeneration`
   directly in the new loading path).

This ensures the model skeleton is built with Mistral-7B dimensions, not
LLaMA-7B defaults.

### Files changed
- `src/radiology_vqa/vlm/llava_med.py` — config pre-processing block in
  `_load_model`; reads `config.json` as raw JSON, builds `MistralConfig`.

---

## Fix 4 — Garbage model output after successful weight loading

### Symptom
Model loaded without errors.  Log showed:
```
Weight remapping complete: 686/689 model params applied.
GPU memory after model load: 12.22 GB
```
But inference output was random subword garbage:
```
Answer: utes ▁ambit elt illy ▁occasion ners 归 uten burg …
Confidence: 0.000
```
Confidence of exactly 0.000 means the model assigned near-zero probability to
every token — consistent with running entirely random weights.

### Root cause — `AlignDevicesHook.weights_map` staleness

The loading sequence was:

```
from_pretrained(device_map="auto")
  → ALL keys MISSING → random init on GPU/CPU
  → accelerate calls dispatch_model() internally
  → AlignDevicesHook attached to CPU-offloaded layers
  → hook.weights_map  ←  initialized from RANDOM parameter tensors  ← BUG
```

Then `_apply_remapped_weights` ran:
```
set_module_tensor_to_device(model, key, device, value=checkpoint_tensor)
  → updates module._parameters[leaf]  ✓
  → does NOT update hook.weights_map  ✗
```

On every forward pass, `AlignDevicesHook.pre_forward()` executes:
```python
value = self.weights_map[name]          # ← stale RANDOM tensor
set_module_tensor_to_device(module, name, exec_device, value=value)
```
The hook reloaded the random weights from its stale `weights_map` and
overwrote the correctly-loaded checkpoint tensors before every single layer
computation.  The model effectively ran with random weights throughout.

### Fix — reverse the order: weights before hooks

The correct sequence is:

```
init_empty_weights()         → meta model, zero memory, NO hooks yet
infer_auto_device_map()      → compute GPU/CPU placement
set_module_tensor_to_device  → materialise meta → real device (no hooks, no staleness)
dispatch_model()             → attach AlignDevicesHook NOW
                               hook.weights_map initialised from CORRECT weights ✓
```

When `dispatch_model` is called **after** all weights are loaded, its hook
initialisation reads the already-correct parameter tensors into `weights_map`.
There is no stale state.

Implemented as the new `_load_with_remapping` method, called from `_load_model`
whenever `old_format=True` and CUDA is available.

### Files changed
- `src/radiology_vqa/vlm/llava_med.py` — new `_load_with_remapping` method;
  `_load_model` updated to use it for the `old_format + CUDA` path.

---

## Processor loading fallback chain

Because `processor_config.json` does not exist in the LLaVA-Med repo, three
tiers of fallback were added to `_load_processor`:

| Tier | Method | Works with |
|------|--------|------------|
| 1 | `AutoProcessor.from_pretrained` | transformers 4.x |
| 2 | `LlavaProcessor.from_pretrained` | transformers 4.x/5.x without processor_config |
| 3 | Manual: `CLIPImageProcessor` + `AutoTokenizer` (with `use_fast=False` fallback) | transformers 5.x, any protobuf situation |

---

## Summary of all `pyproject.toml` additions

```toml
"sentencepiece>=0.1.99",   # Mistral-7B tokenizer
"protobuf>=3.20",          # fast tokenizer conversion (SentencePieceExtractor)
"safetensors>=0.4",        # shard streaming for weight remapping
```

---

## Final loading sequence on SageMaker T4

```
_load_processor()
  Tier 1 → AutoProcessor (likely fails on transformers 5.x)
  Tier 2 → LlavaProcessor.from_pretrained (likely fails)
  Tier 3 → CLIPImageProcessor + AutoTokenizer (succeeds)

_detect_old_key_format()
  Reads model.safetensors.index.json (73 KB, already cached)
  Checks for "model.embed_tokens.weight" → True (old format)

_load_model() — config pre-processing
  Reads config.json as raw JSON (bypasses AutoConfig remapping)
  Builds MistralConfig(intermediate_size=14336, ...)
  Builds LlavaConfig(text_config=mistral_cfg, vision_config=clip_cfg)

_load_with_remapping() — 4-step sequence
  Step 1: init_empty_weights() → LlavaForConditionalGeneration(config)
  Step 2: infer_auto_device_map() → GPU layers + CPU-offloaded layers
  Step 3: stream 4 shards, _remap_key() each, set_module_tensor_to_device()
  Step 4: dispatch_model() → hooks initialised from correct weights

model.eval()
GPU memory: ~12 GB
```

Expected log output:
```
INFO  Built explicit LlavaConfig (MistralConfig text_config): intermediate_size=14336
INFO  device_map inferred: N entries (GPU+CPU split).
INFO  Streaming 4 shard(s) → meta-to-device materialisation (no hooks yet)…
INFO  Weight materialisation complete: 686/686 params loaded.
INFO  GPU memory after model load: 12.xx GB
INFO  LLaVA-Med loaded on device=cuda:0.
```
