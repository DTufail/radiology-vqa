# Dependency Decisions — pyproject.toml

This document records every dependency constraint in [pyproject.toml](../pyproject.toml),
the problem each one solves, and the reasoning behind it.

---

## Why this file exists

The initial pyproject.toml had hard version pins copied from the local development
machine (Intel Mac, no CUDA). Running `pip install -e .` on Google Colab caused a
cascade of package downgrades that broke the GPU environment:

| Package | Colab had | pip downgraded to | Breakage |
|---|---|---|---|
| torch | 2.7.1+cu118 | 2.2.2 | torchvision and torchaudio both pin torch==2.7.1 |
| transformers | 5.2.0 | 4.48.3 | intentional, but triggered the next row |
| huggingface_hub | 1.4.1 | 0.36.2 | transformers 4.48.x requires huggingface_hub<1.0 |

The root cause was that environment limits from the local Mac were encoded as package
constraints, which is wrong. Package constraints should reflect code requirements, not
what happens to be installed in one specific environment.

---

## Constraint-by-constraint decisions

### `torch>=2.0`

**Old**: `torch>=2.0,<2.3`

**Problem**: The `<2.3` upper-bound was the maximum version available on the developer's
Intel Mac. It is not a code requirement — every `torch` API used in this codebase
(`torch.inference_mode`, `torch.softmax`, `torch.stack`, `torch.arange`) has been
stable since torch 1.x. Encoding this as a hard upper-bound downgraded Colab's
GPU-optimised `torch 2.7.1+cu118` to `2.2.2`, which broke `torchvision` and
`torchaudio` (both require `torch==2.7.1`).

**Fix**: Remove the upper-bound. Let each environment use its installed torch.
The local Mac installs 2.2.2 (the latest available there). Colab keeps its 2.7.1.

---

### `numpy>=1.24,<2.0`

**Old**: `numpy>=1.24` (no upper-bound)

**Problem**: PyTorch 2.0–2.3.x is compiled against the NumPy 1.x C-API. NumPy 2.0
changed that API; using NumPy 2.x with torch 2.2.x causes silent wrong results or
runtime crashes. Without an upper-bound, `pip` could resolve NumPy 2.x on a fresh
install.

**Fix**: Add `<2.0`. Colab already has `numpy 1.26.4` so this never triggers a
downgrade. Users with newer torch (≥2.4, which supports NumPy 2.x) are unaffected
because `numpy 1.26.4` is still valid for them.

---

### `sentence-transformers>=2.2`

**Old**: `sentence-transformers>=2.2,<4.0`

**Problem**: The `<4.0` upper-bound was precautionary. sentence-transformers 5.x
requires `torch>=2.4`, which is only available on Colab (not on Mac with torch 2.2).
But the `<4.0` cap prevented Colab (torch 2.7) from using the newer, faster 5.x.
Our usage of the library is minimal — `SentenceTransformer(model_id).encode(texts)` —
which is backward-compatible across all versions.

**Fix**: Remove the upper-bound. On Mac (torch 2.2) pip naturally resolves to 3.x
(below 5.x's torch requirement). On Colab (torch 2.7) it can use 5.x.

---

### `transformers>=4.37`

**Old**: `transformers>=4.40,<4.49`

**Problem**: The `<4.49` upper-bound was motivated by two concerns that no longer apply:

1. **CVE-2025-32434**: transformers 4.49+ blocks `torch.load()` for checkpoints that
   only ship `.bin` weights (no safetensors). This was relevant when the embedding
   model was BiomedBERT (`.bin`-only). After switching to
   `pritamdeka/S-PubMedBert-MS-MARCO`, and given that LLaVA-Med and BLIP-2 both
   ship safetensors weights, the CVE guard never fires for any model in this pipeline.

2. **LLaVA-Med weight key renaming**: `transformers ≥4.45` renamed
   `LlavaForConditionalGeneration`'s expected weight paths from flat
   (`model.layers.*`) to nested (`model.language_model.layers.*`). This caused all
   checkpoint weights to be UNEXPECTED/MISSING, leaving the model randomly
   initialised — then bitsandbytes crashed on the first forward pass.
   This is now handled in `llava_med.py` via `_detect_old_key_format()` and
   `_apply_remapped_weights()`, which detect and fix the mismatch automatically at
   load time regardless of transformers version.

**Collateral damage of `<4.49`**: Colab's `transformers 5.2.0` was downgraded to
`4.48.3`. `transformers 4.48.x` requires `huggingface_hub<1.0`, which forced pip
to downgrade Colab's `huggingface_hub 1.4.1` to `0.36.2`. That in turn broke other
Colab tools that depend on the newer hub API.

**Fix**: Remove the upper-bound. Lower bound relaxed from `4.40` to `4.37` (the
minimum version that introduced `LlavaForConditionalGeneration`).

---

### `safetensors>=0.4`

**Old**: not listed (was a transitive dependency).

**Problem**: `llava_med.py` calls `safetensors.torch.safe_open` directly to read
weight shards for key remapping (when the checkpoint uses old flat naming and the
installed transformers uses new nested naming). An implicit transitive dependency is
fragile — if the resolution chain changes, the import fails at runtime with no
clear message.

**Fix**: Declare it explicitly. `safetensors` is a lightweight, stable library with
no meaningful upper-bound concerns.

---

### `huggingface_hub` — removed from explicit dependencies

**Old**: `huggingface_hub>=0.20` (explicit)

**Problem**: The explicit pin was intended to ensure `hf_hub_download` and
`snapshot_download` are available. However, declaring it as an explicit dependency
caused pip to participate in resolving it, which conflicted with `transformers 4.48.x`
requiring `huggingface_hub<1.0`. The explicit `>=0.20` constraint was satisfied by
`0.36.2`, so pip chose that — downgrading from `1.4.1`.

**Fix**: Remove the explicit declaration. `huggingface_hub` remains a required
transitive dependency of both `transformers` and `datasets`. The APIs we use
(`hf_hub_download`, `snapshot_download`, `local_files_only`) have been available
since ≥0.14.

---

### `bitsandbytes>=0.41` and `accelerate>=0.26` — moved to `[quant]` optional group

**Old**: both in main `dependencies` (required for everyone)

**Problem**: Both packages are CUDA-only. On CPU-only machines (Intel Mac, CI without
GPU), they either fail to install cleanly or install but produce incorrect behaviour
at runtime. They were not installed in the local development venv at all, meaning
`pip install -e .` would silently skip them or fail depending on the environment.
Declaring CUDA-only packages as universally required is a misuse of the dependency
system.

**Fix**: Move to `[project.optional-dependencies]` under the `quant` group.

```bash
# GPU machine (Colab, Linux with NVIDIA GPU):
pip install -e ".[quant]"

# CPU-only machine (Mac, CI, testing):
pip install -e "."
```

---

## Install guide by environment

### Google Colab (GPU)
```bash
pip install -e ".[quant]"
```
Colab's pre-installed torch, torchvision, torchaudio, and huggingface_hub are
preserved without downgrade. bitsandbytes and accelerate are installed for 4-bit
quantization.

### Local Mac / CPU-only machine
```bash
pip install -e "."           # runtime only
pip install -e ".[dev]"      # + pytest and ruff
```
bitsandbytes and accelerate are intentionally absent. torch 2.2.2 is the highest
version available on Intel Mac and must be installed separately (the constraint
`torch>=2.0` allows it).

### Fresh GPU Linux / cloud VM
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # install torch first
pip install -e ".[quant]"
```

---

## Dependency graph (simplified)

```
radiology-vqa
├── pydantic, pydantic-settings, python-dotenv   (config)
├── Pillow, pydicom                              (image I/O)
├── datasets                                     (HF datasets)
├── torch>=2.0, numpy>=1.24,<2.0                (ML runtime)
├── sentence-transformers>=2.2, faiss-cpu        (RAG)
├── transformers>=4.37, safetensors>=0.4         (VLM backends)
│
├── [quant]
│   ├── bitsandbytes>=0.41                       (4-bit/8-bit quantization, CUDA)
│   └── accelerate>=0.26                         (device_map, multi-GPU, CUDA)
│
└── [dev]
    ├── pytest>=7.0, pytest-cov>=4.0
    └── ruff>=0.1
```
