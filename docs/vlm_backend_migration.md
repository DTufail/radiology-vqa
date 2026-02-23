# VLM Backend Migration: LLaVA-Med → LLaVA v1.6

## Why we moved away from LLaVA-Med

`microsoft/llava-med-v1.5-mistral-7b` was the original VLM backend for this project. It is a
biomedical fine-tune of LLaVA v1.5 (Mistral-7B) trained on medical image-question pairs. On
paper it is a good fit for radiology VQA. In practice, the checkpoint has fundamental
incompatibilities with any modern Python environment.

### The checkpoint was released for transformers 4.36.2 (December 2023)

The LLaVA-Med repo has not been updated since then. Running it under transformers ≥ 4.45 or
transformers 5.x requires working around four separate breaking changes, each of which surfaces
only after the previous one is fixed:

| # | Error | Root cause |
|---|-------|------------|
| 1 | `ValueError: Error parsing line b'\x0e'` | Missing `sentencepiece` + `protobuf`; transformers fell back to TikToken which cannot parse the SentencePiece `.model` file. |
| 2 | `torch.OutOfMemoryError: CUDA out of memory` | All 4 checkpoint shards (~14 GB) loaded into CPU RAM simultaneously, then `.to(device="cuda")` attempted while GPU was already full (random fp16 init). |
| 3 | `ValueError: Trying to set a tensor of shape [4096,14336] in "weight" (which has shape [4096,11008])` | transformers 5.x silently remaps `model_type="llava_mistral"` → `"llava"`, creating a `LlamaConfig` with the wrong `intermediate_size` (11008 LLaMA default instead of 14336 Mistral). |
| 4 | Coherent load but random garbage output, `Confidence: 0.000` | `AlignDevicesHook.weights_map` staleness: `from_pretrained(device_map="auto")` with all keys MISSING attaches accelerate hooks initialised from random tensors; `set_module_tensor_to_device` later updates `module._parameters` but not `hook.weights_map`, so the hook reloads random weights before every forward pass. |

Each fix required deep knowledge of transformers internals, accelerate hook lifecycle, and the
specific checkpoint format. The full fix log is in [llava_med_loading_fixes.md](llava_med_loading_fixes.md).

### The final loading sequence was still fragile

Even after all four fixes, the loading path was:

```
Read config.json as raw JSON (bypass AutoConfig)
  → build MistralConfig manually
  → wrap in LlavaConfig
  → init_empty_weights()
  → infer_auto_device_map()
  → stream 4 shards, remap keys, set_module_tensor_to_device
  → dispatch_model()
```

This is ~300 lines of compatibility shim code maintaining a brittle dependency on the exact
internal behaviour of three libraries (transformers, accelerate, bitsandbytes). Any upgrade to
any of these three could break it silently.

### The decisive failure

After implementing all four fixes, `dispatch_model` in accelerate's `big_modeling.py`
called `model.to(device)` on the quantized model immediately after dispatch, which
bitsandbytes explicitly forbids:

```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
```

This is an accelerate ≥ 0.27 / < 0.30 bug that affects **only** checkpoints that trigger the
single-device shortcut in `dispatch_model` (i.e. models that fit entirely on one GPU). On a T4
with 4-bit quantization, the ~4 GB model fits easily, triggering the bug every time.

At this point the cost-benefit calculus was clear: no amount of further shimming would produce a
stable, maintainable backend for a checkpoint that was abandoned two years before the current
environment was built.

---

## The replacement: LLaVA v1.6 Mistral-7B

`llava-hf/llava-v1.6-mistral-7b-hf` is the HuggingFace-native version of LLaVA v1.6 (also
known as LLaVA-NeXT), using the same Mistral-7B language backbone.

### Why it works cleanly

| Property | LLaVA-Med v1.5 | LLaVA v1.6 (HF) |
|----------|----------------|-----------------|
| `processor_config.json` | Missing (404) | Present |
| `tokenizer_config.json` | Old format | Standard |
| Weight key format | Old flat (`model.layers.*`) | New nested (`model.language_model.layers.*`) |
| `model_type` in config | `"llava_mistral"` (not recognised by transformers 5.x) | `"llava_next"` (fully supported) |
| `AutoProcessor.from_pretrained` | Fails on transformers 5.x | Works |
| `BitsAndBytesConfig` 4-bit | Fails (keys MISSING → random init → quant state broken) | Works |
| `from_pretrained(device_map=...)` | Triggers dispatch bugs | Works |
| Key remapping required | Yes (~50 lines) | No |

### The loading code

```python
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map={"": 0},   # direct GPU 0 placement — avoids dispatch_model .to() bug
)
```

The `device_map={"": 0}` pattern places all parameters on GPU 0 directly, bypassing accelerate's
`dispatch_model` function (which is the source of the `.to()` crash). This is safe for any
model that fits entirely on one GPU.

### VRAM usage

| Precision | VRAM | Fits on T4 (16 GB)? |
|-----------|------|---------------------|
| fp16 (no quant) | ~14.3 GB | Barely — risky |
| 4-bit (bitsandbytes) | ~4.5 GB | Yes, with 11 GB headroom |
| 8-bit (bitsandbytes) | ~8 GB | Yes, with 8 GB headroom |

4-bit is the default and recommended setting.

### Prompt format

LLaVA v1.6 Mistral-7B uses the Mistral instruction format:

```
[INST] <image>
{question} [/INST]
```

---

## What changed in the codebase

### New file: `src/radiology_vqa/vlm/llava.py`

Clean backend using `LlavaNextForConditionalGeneration` and `LlavaNextProcessor`.
No compatibility shims. ~180 lines total vs ~800 lines for `llava_med.py`.

### Updated: `src/radiology_vqa/vlm/factory.py`

- `"llava"` → routes to `LLaVABackend` in `llava.py`
- `"llava_med"` → backward-compat alias, also routes to `LLaVABackend`
- Both auto-redirect the old `microsoft/llava-med-v1.5-mistral-7b` model_id to the new checkpoint

### Updated: `src/radiology_vqa/config.py`

```python
vlm_backend: str = "llava"                            # was "llava_med"
vlm_model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"  # was "microsoft/llava-med-v1.5-mistral-7b"
```

### Updated: `scripts/quick_inference.py`, `scripts/run_benchmark.py`

`--backend` choices updated from `["llava_med", "blip2"]` to `["llava", "llava_med", "blip2"]`.

### Retained: `src/radiology_vqa/vlm/llava_med.py`

Kept for reference but no longer invoked. The four-fix loading sequence is documented there
and in `llava_med_loading_fixes.md` for completeness.

---

## Differences in model capability

LLaVA v1.6 is a generation newer than LLaVA v1.5 and uses higher-resolution image tiling
(up to 672×672 effective input), which can improve performance on detail-heavy tasks like
radiology. LLaVA-Med was fine-tuned on medical data but its base model (LLaVA v1.5) is weaker.

The benchmark results will determine whether the domain-general LLaVA v1.6 base outperforms
the domain-specific but architecturally older LLaVA-Med. This is an empirical question that
the benchmark runner (`scripts/run_benchmark.py`) is designed to answer.
