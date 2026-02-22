# SageMaker Setup Guide — ml.g4dn.xlarge

Step-by-step instructions for running the full pipeline (data → RAG index → VLM inference)
using the JupyterNotebook terminal on AWS SageMaker ml.g4dn.xlarge.

**Instance specs**: 1× NVIDIA T4 (16 GB VRAM), 4 vCPUs, 16 GB RAM, 125 GB NVMe SSD
**Persistent storage**: `/home/ec2-user/SageMaker/` — everything else is lost on restart.

---

## Step 1 — Open a terminal

In JupyterLab: **Launcher → Terminal** (or **File → New → Terminal**).

All commands below are run in that terminal.

---

## Step 2 — Verify the environment

```bash
# Confirm GPU is visible
nvidia-smi

# Check CUDA version
nvcc --version

# Check available disk space (should be > 40 GB free)
df -h /home/ec2-user/SageMaker

# Check Python version (need 3.11+)
python3 --version
```

Expected output from `nvidia-smi`: a T4 GPU with 16 GB total memory.
If disk space is low, go to the SageMaker console and increase the EBS volume size.

---

## Step 3 — Get the code onto SageMaker

All project files must be under `/home/ec2-user/SageMaker/` — that is the EBS-backed
persistent volume. Files written elsewhere (e.g. `/tmp/`) are lost when the instance stops.

```bash
cd /home/ec2-user/SageMaker
```

**Option A — git clone (if the repo is on GitHub/GitLab)**
```bash
git clone https://github.com/YOUR_USERNAME/radiology-vqa.git
cd radiology-vqa
```

**Option B — upload a zip file**

1. In JupyterLab left sidebar, navigate to `SageMaker/`
2. Drag and drop `radiology-vqa.zip` (or use **Upload Files** button)
3. Back in the terminal:

```bash
cd /home/ec2-user/SageMaker
unzip radiology-vqa.zip
cd radiology-vqa
```

Confirm you are in the right place:
```bash
pwd          # should print /home/ec2-user/SageMaker/radiology-vqa
ls           # should show: src/ scripts/ tests/ data/ docs/ pyproject.toml Makefile
```

---

## Step 4 — Create a Python virtual environment

Create the venv inside SageMaker persistent storage so it survives restarts.

```bash
python3 -m venv /home/ec2-user/SageMaker/vqa-env
source /home/ec2-user/SageMaker/vqa-env/bin/activate
```

Your prompt should now start with `(vqa-env)`. **You must run this activate command
every time you open a new terminal session.**

Upgrade pip:
```bash
pip install --upgrade pip
```

---

## Step 5 — Install dependencies

Install with the `[quant]` group since CUDA is available on this instance.

```bash
pip install -e ".[quant]"
```

This installs:
- All core dependencies (torch, transformers, sentence-transformers, faiss-cpu, etc.)
- `sentencepiece` — required for the Mistral-7B tokenizer used by LLaVA-Med
- `protobuf` — required by transformers to convert the SentencePiece `.model` file to the fast tokenizer format; without it transformers silently falls back to TikToken which crashes on binary `.model` files
- `safetensors` — used for weight key remapping when loading LLaVA-Med
- `bitsandbytes` — 4-bit / 8-bit quantization
- `accelerate` — required for `device_map="auto"`

Expected duration: 3–6 minutes (downloads ~800 MB).

Verify the install:
```bash
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
import transformers, sentence_transformers, bitsandbytes
print('transformers:', transformers.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
"
```

Expected output:
```
torch: 2.x.x
CUDA available: True
GPU: Tesla T4
```

---

## Step 6 — Set a HuggingFace token (recommended)

Without a token, HuggingFace rate-limits unauthenticated downloads.
LLaVA-Med is a 15 GB model — hitting a rate limit mid-download wastes time.

Get a token at https://huggingface.co/settings/tokens (free account, read-only token).

```bash
# Add to your shell rc so it persists across sessions
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
echo $HF_TOKEN   # should print your token
```

---

## Step 7 — Verify SLAKE data is present

SLAKE is not on HuggingFace — it must be in the repo under `data/raw/Slake1.0/`.

```bash
ls data/raw/Slake1.0/
# should show: KG/  imgs/  train.json  test.json  validate.json
```

If the directory is missing, the project root might have a `data.zip`:
```bash
# If you see data.zip in the project root:
ls data.zip
unzip data.zip -d data/
ls data/raw/Slake1.0/
```

If you do not have the SLAKE data at all, download it from the official source:
```bash
# Download from the SLAKE repository
pip install gdown
gdown --fuzzy "https://drive.google.com/drive/folders/1EZ0WgbpcE8FGBxL85ojo_GsXBi0ODsO9" \
      -O data/raw/ --folder
```

---

## Step 8 — Download VQA-RAD and PathVQA

These are downloaded automatically from HuggingFace into `~/.cache/huggingface/`.

```bash
# Must be run from the project root directory
python scripts/download_datasets.py
```

Expected output:
```
Downloading VQA-RAD...
  VQA-RAD train: 3515 samples
  Sanity check OK: ...
  VQA-RAD test: 451 samples
  ...
Downloading PathVQA...
  PathVQA train: 18000 samples
  ...
SLAKE must be downloaded manually.
```

Duration: 2–5 minutes depending on connection speed.

---

## Step 9 — Run the test suite

Verify the entire codebase is working before touching models.

```bash
pytest tests/ -m "not slow" -v
```

Expected: **81 passed** in under 10 seconds.

If any tests fail, stop here and investigate before proceeding — a later step will fail too.

---

## Step 10 — Build the FAISS knowledge index

This downloads the embedding model (`pritamdeka/S-PubMedBert-MS-MARCO`, ~440 MB)
and builds the retrieval index from the SLAKE knowledge graph.

```bash
python scripts/build_index.py
```

Expected output:
```
Loading KG triples from data/raw/Slake1.0 ...
  Loaded 1010 triples.
Processing triples into documents ...
  Generated 2100 documents.
Initializing embedding model (pritamdeka/S-PubMedBert-MS-MARCO) ...
Building FAISS index ...
Saving index to data/indices ...
--- Index Summary ---
  disease              1200 docs
  organ                 900 docs
  TOTAL                2100 docs
  index.faiss size:     X.XX MB
  Build time:          XX.Xs
```

Duration: 3–8 minutes (first run downloads the embedding model).

Verify the index was created:
```bash
ls data/indices/
# should show: index.faiss  documents.jsonl  index_meta.json
```

---

## Step 11 — Test retrieval quality (optional but recommended)

```bash
python scripts/test_retrieval.py
```

This runs 10 validation queries and shows the top-3 retrieved documents per query.
Scores should be in the range 0.85–0.99 with the correct answer at rank 1.

---

## Step 12 — Run quick inference (sanity check)

Before running a full benchmark, test a single sample to confirm the VLM loads and
generates output correctly.

**Option A — BLIP-2 (2.7B, safe for T4, ~4 GB VRAM in 4-bit)**
```bash
python scripts/quick_inference.py \
    --dataset vqa_rad \
    --index 0 \
    --backend blip2
```

**Option B — LLaVA-Med (7B, ~14 GB VRAM in fp16 on this instance)**
```bash
python scripts/quick_inference.py \
    --dataset vqa_rad \
    --index 0 \
    --backend llava_med
```

> **Note on LLaVA-Med**: The codebase automatically detects the old weight key format
> in the LLaVA-Med checkpoint and applies key remapping. With newer transformers it
> falls back to fp16 (instead of 4-bit) and logs a warning. fp16 uses ~14 GB of the
> T4's 16 GB — it fits, but barely. Monitor with `nvidia-smi` in a second terminal.

Expected output:
```
Model: blip2-opt-2.7b-4bit (or llava-med-v1.5-mistral-7b-fp16)
Question: Is there evidence of an aortic aneurysm?
Answer: no
Confidence: 0.72
Latency: 1.23s
```

Model download happens on first run only (~5 GB for BLIP-2, ~15 GB for LLaVA-Med).

---

## Step 13 — Run a full benchmark

**Quick run (50 samples) — confirm everything works end-to-end:**
```bash
python scripts/run_benchmark.py \
    --dataset vqa_rad \
    --split test \
    --backend blip2 \
    --max-samples 50
```

**Full VQA-RAD test benchmark (451 samples):**
```bash
python scripts/run_benchmark.py \
    --dataset vqa_rad \
    --split test \
    --backend blip2
```

**BLIP-2 with batching (faster on T4 — uses native batch support):**
```bash
python scripts/run_benchmark.py \
    --dataset vqa_rad \
    --split test \
    --backend blip2 \
    --batch-size 8
```

Results are saved to `data/benchmarks/` as a timestamped JSON file.

**View results:**
```bash
ls data/benchmarks/
cat data/benchmarks/blip2-opt-2.7b-4bit_vqa_rad_test_*.json | python3 -m json.tool | head -30
```

**Compare multiple benchmark runs:**
```bash
python scripts/run_benchmark.py --compare
```

---

## Step 14 — Monitor GPU usage

While a benchmark is running, open a second terminal and watch GPU utilisation:

```bash
watch -n 2 nvidia-smi
```

For BLIP-2 4-bit you should see ~4 GB VRAM used, ~80–100% GPU utilisation during
inference. For LLaVA-Med fp16 expect ~14 GB VRAM.

---

## Keeping the environment across restarts

When SageMaker stops and restarts the instance, the code and data in
`/home/ec2-user/SageMaker/` persists. The virtual environment also persists.
You only need to re-activate it:

```bash
source /home/ec2-user/SageMaker/vqa-env/bin/activate
cd /home/ec2-user/SageMaker/radiology-vqa
```

Add this to `~/.bashrc` to activate automatically on login:
```bash
echo 'source /home/ec2-user/SageMaker/vqa-env/bin/activate' >> ~/.bashrc
echo 'cd /home/ec2-user/SageMaker/radiology-vqa' >> ~/.bashrc
```

---

## Reference — all key commands

```bash
# Activate environment (every new terminal)
source /home/ec2-user/SageMaker/vqa-env/bin/activate
cd /home/ec2-user/SageMaker/radiology-vqa

# Download datasets
python scripts/download_datasets.py

# Build index
python scripts/build_index.py

# Run tests
pytest tests/ -m "not slow" -v

# Quick inference — one sample
python scripts/quick_inference.py --dataset vqa_rad --index 0 --backend blip2
python scripts/quick_inference.py --dataset vqa_rad --index 0 --backend llava_med

# Full benchmark
python scripts/run_benchmark.py --dataset vqa_rad --split test --backend blip2
python scripts/run_benchmark.py --dataset vqa_rad --split test --backend llava_med

# Monitor GPU
watch -n 2 nvidia-smi
```

---

## VRAM reference for ml.g4dn.xlarge (T4 — 16 GB)

| Backend | Precision | VRAM | Fits T4? |
|---|---|---|---|
| BLIP-2 OPT-2.7B | 4-bit | ~4 GB | Yes (comfortable) |
| BLIP-2 OPT-2.7B | 8-bit | ~6 GB | Yes |
| LLaVA-Med 7B | 4-bit (requires transformers<4.45) | ~5 GB | Yes |
| LLaVA-Med 7B | fp16 (auto fallback with newer transformers) | ~14 GB | Yes (tight) |
| LLaVA-Med 7B | fp32 | ~28 GB | No — OOM |

Start with BLIP-2 to verify the pipeline. Move to LLaVA-Med once confirmed working.
