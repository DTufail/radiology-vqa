# Phase 1: Data Foundation & Project Skeleton

**Status:** Complete
**Tests:** 23/23 pass (`make test-slow`)
**Python:** ≥3.11

---

## 1. Overview

Phase 1 establishes the data layer of the Grounded Multi-Agent Radiology VQA system. It provides:

- A unified data schema (`VQASample`) that normalises three heterogeneous VQA datasets into a single interface
- Dataset-specific loaders with graceful error handling
- A SLAKE Knowledge Graph (KG) loader that parses medical entity triples from local CSV files
- DICOM image utilities (windowing, anonymisation, PIL conversion)
- Validated, typed configuration via `pydantic-settings`

No ML models are loaded in this phase. The output of Phase 1 is a reliable, typed data pipeline that every downstream component depends on.

---

## 2. Dataset Overview

| Dataset | Source | Domain | Size | Splits | Notes |
|---------|--------|--------|------|--------|-------|
| VQA-RAD | HuggingFace (`flaviagiammarino/vqa-rad`) | Radiology (X-ray, CT, MRI) | 2,244 QA pairs | train (1,793), test (451) | No modality labels; answer type inferred |
| SLAKE | Local (`data/raw/Slake1.0/`) | Radiology (CT, X-ray, MRI) | 7,033 English QA pairs | train (4,919), val (1,061), test (1,053) | Bilingual; filter `q_lang == "en"` |
| PathVQA | HuggingFace (`flaviagiammarino/path-vqa`) | Pathology (histology, tissue) | 32,632 QA pairs | train (19,654), val (6,259), test (6,719) | Some images are CMYK — must convert to RGB |

> **Domain note:** PathVQA is pathology, not radiology. Its visual domain is distinct from X-rays and CT scans. It is included to broaden the answer space but should be treated separately in evaluation.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Data Sources                         │
│                                                       │
│   HuggingFace                  Local Filesystem       │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│   │  VQA-RAD │  │ PathVQA  │  │  Slake1.0/       │   │
│   └────┬─────┘  └────┬─────┘  │  ├── train.json  │   │
│        │             │        │  ├── imgs/        │   │
│        │             │        │  └── KG/          │   │
│        │             │        └────────┬─────────┘   │
└────────┼─────────────┼─────────────────┼─────────────┘
         │             │                 │
         ▼             ▼                 ▼
   ┌──────────┐  ┌──────────┐    ┌───────────────┐
   │ loader.py│  │ loader.py│    │slake_loader.py│
   │load_vqa_ │  │load_path │    │  load_slake() │
   │   rad()  │  │  vqa()   │    └───────┬───────┘
   └────┬─────┘  └────┬─────┘            │
        │             │             ┌────┴─────┐
        │             │             │kg_loader │
        │             │             │load_kg() │
        │             │             └────┬─────┘
        ▼             ▼                  ▼
   ┌──────────┐  ┌──────────┐    ┌───────────────┐
   │VQASample │  │VQASample │    │  SLAKESample  │
   │  list    │  │  list    │    │  +KGTriple    │
   └──────────┘  └──────────┘    └───────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   schema.py     │
               │  (shared types) │
               └─────────────────┘
```

---

## 4. Data Schemas

### 4.1 `VQASample` — Unified VQA Record

```python
class VQASample(BaseModel):
    image: Any          # PIL.Image.Image (RGB)
    question: str
    answer: str
    answer_type: str    # "closed" | "open"
    modality: str       # "xray" | "ct" | "mri" | "pathology" | "unknown"
    source: str         # "vqa_rad" | "slake" | "pathvqa"
    sample_id: str      # "{source}_{split}_{index}"
```

This is the **primary data contract** across Phase 1–4. All loaders return `list[VQASample]`. Downstream components must not depend on dataset-specific fields unless they explicitly downcast to `SLAKESample`.

**Normalisation rules:**
- `answer_type`: inferred as `"closed"` iff `answer.strip().lower() in {"yes", "no"}`, else `"open"`
- `modality`: always lowercase. VQA-RAD has no labels → `"unknown"`. PathVQA → `"pathology"`.
- Images: always RGB (CMYK conversion applied for PathVQA)

### 4.2 `SLAKESample` — SLAKE Extension

```python
class SLAKESample(VQASample):
    location: str       # body region, e.g. "Abdomen", "Chest"
    content_type: str   # question category, e.g. "Modality", "Organ"
    triple: list[str]   # KG triple ["head", "relation", "tail"]
    img_name: str       # relative image path, e.g. "xmlab1/source.jpg"
```

`SLAKESample` extends `VQASample` — it is a strict superset. Code that works with `VQASample` will accept `SLAKESample` without modification.

### 4.3 `KGTriple` — Knowledge Graph Triple

```python
class KGTriple(BaseModel):
    head: str       # entity, e.g. "Pneumonia"
    relation: str   # attribute, e.g. "symptom", "cause", "function"
    tail: str       # value, e.g. "fever, cough, dyspnea"
    category: str   # "disease" | "organ" | "organ_rel"
```

`KGTriple` represents a single fact from the SLAKE Knowledge Graph. It is produced by `kg_loader.py` and consumed by Phase 2's `KGProcessor`.

---

## 5. Module Reference

### 5.1 `config.py`

Configuration is managed via `pydantic-settings` and loads from a `.env` file. A singleton `settings` object is imported across all modules.

```python
from radiology_vqa.config import settings

settings.slake_dir      # Path: ./data/raw/Slake1.0
settings.index_dir      # Path: ./data/indices  (Phase 2)
settings.embedding_model  # str: BiomedBERT model ID  (Phase 2)
```

All configurable values have environment variable overrides. Prefix the field name with the variable name directly (e.g. `SLAKE_DIR=./my/path`).

| Setting | Default | Description |
|---------|---------|-------------|
| `data_dir` | `./data` | Root data directory |
| `slake_dir` | `./data/raw/Slake1.0` | Local SLAKE root |
| `vqa_rad_dataset` | `flaviagiammarino/vqa-rad` | HF dataset ID |
| `pathvqa_dataset` | `flaviagiammarino/path-vqa` | HF dataset ID |
| `log_level` | `INFO` | Python logging level |

### 5.2 `data/loader.py`

```python
def load_vqa_rad(split: str = "train") -> list[VQASample]: ...
def load_pathvqa(split: str = "train") -> list[VQASample]: ...
```

Both functions catch all exceptions, log the error, and return `[]` rather than crashing. This ensures the rest of the pipeline can continue if one source is unavailable.

**PathVQA CMYK handling:** Some PathVQA images are stored as CMYK JPEG (common in scanned medical literature). The loader calls `.convert("RGB")` on every image before constructing the `VQASample`, guaranteeing all downstream code sees 3-channel images.

### 5.3 `data/slake_loader.py`

```python
def load_slake(slake_dir: Path, split: str = "train") -> list[SLAKESample]: ...
```

**Split file mapping:**

| `split` argument | File read |
|-----------------|-----------|
| `"train"` | `train.json` |
| `"validation"` | `validate.json` |
| `"test"` | `test.json` |

**Key behaviours:**
- Filters to `q_lang == "en"` only — Chinese rows are discarded
- Caches loaded PIL images by `img_name` — 642 unique images are shared across ~7,000 QA pairs; without caching, memory and load time blow up
- Missing images: logs a `WARNING` and skips the affected QA pairs — does not crash
- Modality normalisation: `"X-Ray"` → `"xray"`, `"CT"` → `"ct"`, `"MRI"` → `"mri"`

### 5.4 `data/kg_loader.py`

```python
def load_kg(slake_dir: Path) -> list[KGTriple]: ...
```

Loads the three English KG files:

| File | Content | Rows |
|------|---------|------|
| `KG/en_disease.csv` | Disease attributes (symptom, cause, treatment, …) | 2,215 |
| `KG/en_organ.csv` | Organ attributes (function, position, definition, …) | 280 |
| `KG/en_organ_rel.csv` | Organ-to-system membership (all "belong to") | 102 |

**Delimiter:** All three files use `#` as the column separator (not comma). The loader uses `csv.reader(f, delimiter='#')`.

**macOS resource forks:** The `KG/` directory may contain `._` prefixed shadow files created by macOS. The loader explicitly skips any file whose name starts with `._`.

**Column detection:** The loader attempts to identify `head`, `relation`, and `tail` columns by name from the header row. On failure, it falls back to positional indices (0, 1, 2), which is correct for all three SLAKE KG files.

### 5.5 `data/dicom_handler.py`

Utilities for loading and processing DICOM files. These are not used by the VQA loaders but are provided for clinical deployment scenarios where raw DICOM inputs are ingested.

```python
def load_dicom(path: Path) -> dict:
    # Returns {"pixel_array": np.ndarray, "metadata": dict}

def apply_windowing(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float
) -> np.ndarray:
    # Returns uint8 array clipped and scaled to [0, 255]

def dicom_to_pil(path: Path) -> Image.Image:
    # Full pipeline: load → window → convert to RGB PIL Image

def anonymize_metadata(metadata: dict) -> dict:
    # Strips 9 PHI fields, returns cleaned copy
```

**PHI fields stripped by `anonymize_metadata`:**
`PatientName`, `PatientID`, `PatientBirthDate`, `PatientSex`, `PatientAge`, `InstitutionName`, `ReferringPhysicianName`, `StudyDate`, `AccessionNumber`

**Windowing fallback:** If `WindowCenter` / `WindowWidth` tags are absent from the DICOM header, the pixel array is min-max normalised to [0, 255].

**Multi-frame rejection:** Multi-frame DICOMs (e.g. CT volumes encoded as a single file) are rejected with a `ValueError` and a logged warning. Only single-frame DICOMs are supported.

---

## 6. SLAKE Directory Structure

```
Slake1.0/
├── train.json          # 9,835 rows total (4,919 English)
├── validate.json       # 2,099 rows total (1,061 English)
├── test.json           # 2,094 rows total (1,053 English)
├── imgs/
│   └── xmlab{N}/
│       ├── source.jpg      # radiology image (the one we load)
│       ├── detection.json  # organ bounding boxes (Phase 3+)
│       ├── question.json   # per-image QA index
│       └── mask.png        # segmentation mask (Phase 3+)
├── KG/
│   ├── en_disease.csv      # English disease triples
│   ├── en_organ.csv        # English organ triples
│   └── en_organ_rel.csv    # English organ-system membership
└── mask.txt                # pixel-label mapping for masks
```

---

## 7. Setup & Usage

### Installation

```bash
make install
# Equivalent to: pip install -e ".[dev]"
```

### Downloading HuggingFace Datasets

```bash
make download-data
# Downloads VQA-RAD and PathVQA to HF cache (~2GB)
```

SLAKE requires manual download. Place the `Slake1.0/` directory at `data/raw/Slake1.0/`.

### Validating All Datasets

```bash
make validate-data
# Loads all splits, reports statistics, checks for integrity issues
# Exits with code 1 on any critical error
```

### Running Tests

```bash
make test        # fast tests only — no downloads (~1s)
make test-slow   # includes HF loader tests (~8 min first run)
```

---

## 8. Testing Strategy

Tests are split into two tiers by the `@pytest.mark.slow` marker:

| Tier | Marker | Coverage | Duration |
|------|--------|----------|----------|
| Fast | _(none)_ | Config, schema validation, SLAKE loader (fixtures), KG loader (fixtures), DICOM math | <2s |
| Slow | `@pytest.mark.slow` | VQA-RAD and PathVQA HF downloads, full loader integration | ~8 min (cached after first run) |

Fast tests use `tmp_path` fixtures with synthetic data — they never touch the network or filesystem beyond `tmp/`.

All fixtures are defined in `tests/conftest.py`:

| Fixture | Purpose |
|---------|---------|
| `sample_vqa_sample` | Valid `VQASample` with synthetic PIL image |
| `sample_slake_sample` | Valid `SLAKESample` with all SLAKE fields |
| `sample_kg_triple` | Valid `KGTriple` |
| `tmp_slake_dir` | Temp dir mimicking `Slake1.0/` with 3 English + 1 Chinese row |
| `synthetic_dicom` | Minimal DICOM file with pixel data and windowing tags |

---

## 9. Design Decisions

### Why pydantic-settings for configuration?
Type-safe configuration with `.env` support, validation at startup, and IDE autocomplete. The alternative (argparse / raw `os.getenv`) is error-prone at scale and provides no validation.

### Why PIL images inside Pydantic models?
PIL images are not JSON-serialisable, which requires `ConfigDict(arbitrary_types_allowed=True)`. This is a deliberate trade-off: keeping the image co-located with its metadata avoids the complexity of an image registry and matches how `datasets` returns data.

### Why a separate `SLAKESample` rather than optional fields on `VQASample`?
SLAKE metadata (`location`, `content_type`, `triple`) is structurally absent from VQA-RAD and PathVQA — not just null. Optional fields would require every consumer to null-check fields that are semantically meaningless for those datasets. Subclassing preserves the Liskov Substitution Principle: `SLAKESample` can be used anywhere `VQASample` is expected.

### Why filter SLAKE to English only?
The system is English-only at this stage. Chinese QA pairs would contaminate any embedding-based retrieval and confuse answer generation. Filtering at load time rather than at inference time prevents accidental inclusion.

### Why cache images in `slake_loader.py`?
Each SLAKE image is referenced by multiple QA pairs (typically 10–20 questions per image). Loading 642 unique images once and reusing them reduces I/O from ~7,000 disk reads to 642.

---

## 10. Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| VQA-RAD has no modality labels | `modality = "unknown"` for all VQA-RAD samples | Visual modality classifier deferred to Phase 3 |
| SLAKE bounding boxes not loaded | `detection.json` ignored | Phase 3 will load for spatial grounding |
| PathVQA is pathology, not radiology | Domain mismatch with X-ray/CT samples | Separate evaluation splits recommended |
| No deduplication across datasets | Overlapping question-answer pairs possible | Dataset-level train/val/test splits prevent leakage |
| DICOM handler not integrated with loaders | Loaders work with HF/JPEG images | Phase 3+ will add a DICOM ingestion path |
