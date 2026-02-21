# Phase 2: RAG Knowledge Base & Retrieval Pipeline

**Status:** Complete
**Tests:** 53/53 pass (`make test-slow`)
**Depends on:** Phase 1 (data schemas, KG loader)

---

## 1. Overview

Phase 2 implements the retrieval backbone of the system. Every factual claim the VQA system makes must be grounded in retrieved evidence — this phase provides the mechanism for that grounding.

The pipeline transforms raw SLAKE Knowledge Graph triples into a searchable vector index, then exposes a typed retrieval interface that downstream agents will call at inference time. Retrieval quality in this phase directly determines the factual accuracy of all agent outputs in Phases 3 and 4.

**What is built:**
- A `Document` model with full provenance metadata (source file, entity, attribute)
- Natural-language templates that convert KG triples into grammatical English sentences
- A dual-document indexing strategy (per-attribute precision + entity summaries for recall)
- A FAISS `IndexFlatIP` vector store with exact cosine similarity search
- A `Retriever` class that is the stable contract consumed by agents in Phase 4

**What is deliberately deferred:**
- PubMed abstract indexing (architecture supports it without refactoring)
- SLAKE QA-derived knowledge statements (same)

---

## 2. Architecture

```
  Raw Sources                Processing               Index              Inference
  ──────────                 ──────────               ─────              ─────────

  en_disease.csv ──┐
  en_organ.csv  ───┤──► KGProcessor ──► list[Document] ──► Embedder ──► FAISSIndexer
  en_organ_rel  ───┘   (templates +                    (S-PubMedBert)  (IndexFlatIP)
                        summaries)                                            │
                                                                             │ save()
                                                                             ▼
  [Future]                                                           data/indices/
  PubMed abs  ────────► SourceProcessor ──► list[Document] ──►     ├─ index.faiss
  SLAKE QA    ────────► SourceProcessor ──►  (same schema)          ├─ documents.jsonl
                                                                     └─ index_meta.json
                                                                             │
                                                                             │ load()
                                                                             ▼
                                                                        Retriever
                                                                             │
                                                          ┌──────────────────┘
                                                          │
                                                          ▼
                                              retrieve(query, top_k)
                                                          │
                                                          ▼
                                              list[RetrievalResult]
                                              ├─ document.text     ← evidence
                                              ├─ document.meta     ← citation
                                              └─ score             ← confidence
```

---

## 3. Data Models

### 3.1 `DocumentMeta` — Provenance

```python
class DocumentMeta(BaseModel):
    source_type: str    # "kg_disease" | "kg_organ" | "kg_organ_rel" | "pubmed" | "slake_qa"
    entity_name: str    # e.g. "Pneumonia", "Liver"
    attribute: str      # e.g. "symptom", "function", "summary"
    source_file: str    # e.g. "en_disease.csv"
    chunk_index: int    # 0 for non-chunked; position within parent for chunked
```

Every indexed document carries this metadata. Without it, retrieval results are unattributable — useless for clinical citation.

### 3.2 `Document` — Indexed Unit

```python
class Document(BaseModel):
    text: str = Field(min_length=1)     # the text that gets embedded
    meta: DocumentMeta
    doc_id: str                         # unique, deterministic
```

`doc_id` is built deterministically as:
```
{source_type}_{slugified_entity}_{slugified_attribute}_{chunk_index}
```

Example: `kg_disease_lobar_pneumonia_symptom_0`

Deterministic IDs enable re-indexing without orphaned records. `min_length=1` on `text` is validated at construction time — an empty document would waste an embedding slot and pollute results.

### 3.3 `RetrievalResult` — Query Output

```python
class RetrievalResult(BaseModel):
    document: Document
    score: float    # cosine similarity in [−1, 1]; higher is better
    rank: int       # 1-indexed position in the returned list
```

This is the **stable API contract** that Phase 4 agents consume. Do not change this schema after Phase 2.

---

## 4. Module Reference

### 4.1 `rag/embedder.py` — `Embedder`

Wraps `sentence-transformers` with a biomedical sentence similarity model. All embeddings are L2-normalised so that inner product equals cosine similarity — this is required for `IndexFlatIP` to give meaningful scores.

```python
class Embedder:
    def __init__(self, model_name: str | None = None) -> None: ...
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...   # 768
    @property
    def model_name(self) -> str: ...
```

**Model:** `pritamdeka/S-PubMedBert-MS-MARCO`

PubMedBERT fine-tuned on the MS-MARCO passage retrieval dataset via sentence-transformers contrastive training. This is a proper sentence similarity model: it was trained with a supervised objective to place semantically related passages close in embedding space, which is exactly what retrieval requires.

The original model choice (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) was a masked language model (MLM) used with default mean pooling — not trained for retrieval. Validation confirmed this produced compressed, non-discriminative scores (all results in the 0.956–0.981 range) where "function of liver" ranked "Heart belongs to Circulatory System" above "Liver / function". After the swap to S-PubMedBert-MS-MARCO, scores span a meaningful range (0.88–0.99) and all 10 validation queries return semantically correct top-1 results.

**Dependency constraints** (see `pyproject.toml`):
- `sentence-transformers>=2.2,<4.0` — v5+ requires `torch>=2.4` which is unavailable in this environment
- `torch>=2.0,<2.3` — max available for this environment
- `transformers>=4.40,<4.49` — v4.49+ enforces CVE-2025-32434 which blocks loading `.bin` weights with `torch<2.6`

**Return shapes:**
- `embed_texts(n_texts)` → `(N, 768)` float32, L2-normalised
- `embed_query(query)` → `(1, 768)` float32, L2-normalised

### 4.2 `rag/chunker.py` — `TextChunker`

Word-based text splitter with configurable overlap. KG entries are short and never trigger chunking in practice. This component exists to handle PubMed abstracts in future iterations.

```python
class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None: ...
    def chunk(self, text: str) -> list[str]: ...
```

**Behaviour:**
- Texts shorter than `chunk_size` words → returned as a single-element list
- Empty or whitespace-only text → returns `[]`
- Adjacent chunks share `chunk_overlap` words at boundaries

**Word-based, not token-based:** Tokeniser-based chunking would require loading the embedding model just to split text. Word splitting is a practical approximation that works well when `chunk_size` is set conservatively (512 words ≈ 300–400 BiomedBERT tokens for biomedical text).

### 4.3 `rag/kg_processor.py` — `KGProcessor`

Converts `KGTriple` objects (produced by Phase 1's `kg_loader`) into `Document` objects ready for indexing.

```python
class KGProcessor:
    def process_diseases(self, triples: list[KGTriple]) -> list[Document]: ...
    def process_organs(self, triples: list[KGTriple]) -> list[Document]: ...
    def process_organ_relations(self, triples: list[KGTriple]) -> list[Document]: ...
    def process_all(self, triples: list[KGTriple]) -> list[Document]: ...
```

#### Dual-Document Strategy

Each entity is indexed with two document types:

**Per-attribute documents** (precision):
```
Pneumonia symptoms include: chills, high fever, chest pain, cough, rusty sputum.
Treatment for Pneumonia: antibiotic drug therapy.
Pneumonia is caused by: infection due to streptococcus pneumoniae.
```
These are targeted. A query for "pneumonia treatment" will strongly match the treatment document and ignore the symptom document.

**Entity summary documents** (recall):
```
Pneumonia. Location: Lung. Symptoms: chills, high fever, chest pain.
Causes: streptococcus pneumoniae. Treatment: antibiotic drug therapy.
Prevention: avoid bacterial infection.
```
These are broad. A query like "tell me about pneumonia" or one that partially overlaps multiple attributes will match the summary even when the per-attribute match is weak.

The dual strategy trades a 2× index size for significantly improved recall on broad and ambiguous queries.

#### Natural Language Templates

KG triples are stored as structured data (`head#relation#tail`). Raw triple strings are poor candidates for embedding — their semantics are implicit in the relation name, not expressed as natural language. Templates convert them:

| Attribute | Template |
|-----------|---------|
| `symptom` | `"{entity} symptoms include: {value}"` |
| `cause` | `"{entity} is caused by: {value}"` |
| `treatment` | `"Treatment for {entity}: {value}"` |
| `location` | `"{entity} is located in the {value}"` |
| `description` | `"{entity}: {value}"` |
| `prevention` | `"Prevention of {entity}: {value}"` |
| `infectivity` | `"{entity} infectivity: {value}"` |
| `susceptible_population` | `"{entity} commonly affects: {value}"` |
| `function` | `"The function of {entity}: {value}"` |
| `position` | `"{entity} is located at: {value}"` |
| `definition` | `"{entity}: {value}"` |
| `component` | `"{entity} consists of: {value}"` |
| `belong to` | `"{entity} belongs to the {value}."` |

Templates produce grammatical sentences that match the style of biomedical questions. "Pneumonia symptoms include: fever" will embed close to "What are the symptoms of pneumonia?" because both are natural English. The raw triple "Pneumonia#symptom#fever" would not.

#### Document Counts (from real SLAKE KG)

| Source | Per-attribute docs | Summary docs | Total |
|--------|-------------------|--------------|-------|
| `en_disease.csv` | ~2,215 | ~302 | ~2,517 |
| `en_organ.csv` | ~280 | ~105 | ~385 |
| `en_organ_rel.csv` | 102 | — | 102 |
| **Total** | | | **~3,004** |

### 4.4 `rag/indexer.py` — `FAISSIndexer`

Builds and persists the FAISS vector index. This is an offline/batch component — it runs once during index construction, not at inference time.

```python
class FAISSIndexer:
    def __init__(self, embedder: Embedder) -> None: ...
    def build_index(self, documents: list[Document]) -> None: ...
    def save(self, index_dir: Path) -> None: ...

    @classmethod
    def load(cls, index_dir: Path) -> tuple[faiss.Index, list[Document], dict]: ...
```

**Index type: `IndexFlatIP`**

With ~3,000 documents and 768 dimensions, an approximate index (IVF, HNSW) would add engineering complexity with no meaningful speedup. `IndexFlatIP` does exact nearest-neighbour search — every query scans all 3,000 vectors. At 768 dimensions, this takes <5ms on CPU. Exact search eliminates recall loss from approximation.

Inner product on L2-normalised vectors is mathematically equivalent to cosine similarity. All embeddings are normalised at the `Embedder` layer, so `IndexFlatIP` scores are cosine similarities in [−1, 1].

**Persisted files:**

| File | Format | Contents |
|------|--------|---------|
| `index.faiss` | FAISS binary | Vector index (FAISS native format) |
| `documents.jsonl` | JSON Lines | One `Document.model_dump_json()` per line |
| `index_meta.json` | JSON | `doc_count`, `embedding_model`, `dimension`, `built_at` |

The document list is stored separately from FAISS because FAISS stores only vectors — the text and metadata must be retrieved by index position after a search.

**Dependency injection:** `FAISSIndexer` receives an `Embedder` via constructor. It does not instantiate one internally. This enables:
- Injecting a `MockEmbedder` in tests without downloading the model
- Sharing a single `Embedder` instance between the indexer and the retriever
- Future swapping of embedding models without modifying indexer code

### 4.5 `rag/retriever.py` — `Retriever`

The online query interface. Stateless after initialisation — all mutable state is the FAISS index loaded into memory.

```python
class Retriever:
    def __init__(self, index_dir: Path, embedder: Embedder | None = None) -> None: ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]: ...

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[RetrievalResult]: ...
```

#### `retrieve()` — Primary Interface

1. Embeds the query with `Embedder.embed_query()` → `(1, 768)` vector
2. Calls `index.search(query_vec, k)` → raw FAISS scores and indices
3. Filters out results with `index == -1` (FAISS padding for empty slots) and `score < min_score`
4. Returns `list[RetrievalResult]` with 1-indexed `rank`

Empty index (`ntotal == 0`) returns `[]` immediately without querying FAISS.

#### `retrieve_with_filter()` — Source-Type Filtered Interface

Over-fetches `top_k * 3` results from FAISS, then applies a metadata filter on `source_type`. Re-ranks the filtered list before returning.

This is acceptable at ≤5,000 documents. At larger scale, FAISS filtered search (using ID selectors) would be more efficient.

#### Phase 4 Agent Contract

This is the interface agents will use in Phase 4:

```python
retriever = Retriever(index_dir=settings.index_dir)

results = retriever.retrieve(
    query="visual findings + question",
    top_k=5,
    min_score=settings.retrieval_min_score,
)

for r in results:
    evidence_text = r.document.text      # inject into agent context
    citation = r.document.meta           # source_type, entity_name, attribute
    confidence = r.score                 # for re-ranking or threshold filtering
```

**This contract is frozen.** `RetrievalResult` schema and `Retriever.retrieve()` signature must not change after Phase 2.

---

## 5. CLI Scripts

### `scripts/build_index.py`

Builds the FAISS index from all available sources and saves it to `settings.index_dir`.

```bash
python scripts/build_index.py              # KG only (PubMed deferred)
python scripts/build_index.py --source kg  # explicit KG-only
python scripts/build_index.py --source all # KG + future sources
```

**Expected output:**
```
Loading KG triples from data/raw/Slake1.0 ...
  Loaded 2600 triples.
Processing triples into documents ...
  Generated 2987 documents.
Initializing embedding model (pritamdeka/S-PubMedBert-MS-MARCO) ...
Building FAISS index ...
Saving index to data/indices ...

--- Index Summary ---
  kg_disease           2501 docs
  kg_organ              383 docs
  kg_organ_rel          103 docs
  TOTAL                2987 docs
  index.faiss size:    8.75 MB
  Build time:          ~300s
  Index saved to:      data/indices
```

### `scripts/test_retrieval.py`

Validates retrieval quality against 10 representative medical queries.

```bash
python scripts/test_retrieval.py                              # run all validation queries
python scripts/test_retrieval.py --query "lung function"      # single query
python scripts/test_retrieval.py --interactive                # REPL mode
```

**Validation queries:**
1. What are the symptoms of pneumonia?
2. What is the function of the liver?
3. What causes lung cancer?
4. Where is the heart located?
5. What is the treatment for asthma?
6. Which organs belong to the digestive system?
7. What does consolidation in the lung indicate?
8. What are the symptoms of a brain tumor?
9. What is the function of the pancreas?
10. How is tuberculosis transmitted?

---

## 6. Configuration Reference

New settings added in Phase 2 (all in `Settings`):

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `pritamdeka/S-PubMedBert-MS-MARCO` | Sentence-transformers model |
| `index_dir` | `./data/indices` | Directory for persisted FAISS index |
| `chunk_size` | `512` | Words per chunk (for PubMed abstracts, future) |
| `chunk_overlap` | `50` | Overlap words between adjacent chunks |
| `retrieval_top_k` | `5` | Default number of results to return |
| `retrieval_min_score` | `0.3` | Default minimum cosine similarity threshold |

Override via environment:
```bash
RETRIEVAL_MIN_SCORE=0.5 python scripts/test_retrieval.py
```

---

## 7. Setup & Workflow

```bash
# 1. Install dependencies (includes sentence-transformers, faiss-cpu, torch)
make install

# 2. Build the index (~2 min on CPU)
make build-index-kg

# 3. Validate retrieval quality
make test-retrieval

# 4. Run all tests (fast tests: <2s; slow tests: ~8 min)
make test
make test-slow
```

---

## 8. Testing Strategy

### Test Tiers

| File | Tier | What it covers |
|------|------|----------------|
| `test_document.py` | Fast | Schema validation, `min_length` enforcement |
| `test_chunker.py` | Fast | Chunking correctness, edge cases |
| `test_kg_processor.py` | Fast | Template rendering, summary content, doc_id uniqueness |
| `test_indexer.py` | Fast | Index construction, save/load roundtrip, metadata fields |
| `test_embedder.py` | Slow | Model loading, vector shapes, L2-norm, semantic ordering |
| `test_retriever.py` | Slow | End-to-end retrieval, filter, empty index, score thresholding |

### Key Fast-Test Infrastructure

The `MockEmbedder` in `conftest.py` is a zero-dependency drop-in for `Embedder`:

```python
class MockEmbedder:
    model_name = "mock"
    _dim = 8

    def embed_texts(self, texts, batch_size=64) -> np.ndarray:
        # Deterministic random unit vectors (seed=42)

    def embed_query(self, query) -> np.ndarray:
        # Deterministic random unit vector (seed=123)
```

`FAISSIndexer` and `Retriever` accept any object that implements the embedder interface (duck typing). This allows fast tests to run without loading the ~440MB embedding model.

### Semantic Similarity Test

`test_semantic_similarity` (slow) verifies that the embedding model captures biomedical semantics:

```python
sim("lung disease symptoms", "pulmonary condition signs")
  > sim("lung disease symptoms", "knee replacement surgery")
```

This test would catch a model swap to a general-purpose model that happens to have compatible vector dimensions.

---

## 9. Design Decisions

### Why FAISS `IndexFlatIP` and not HNSW or IVF?

At 3,000 documents and 768 dimensions, a flat exact-search index:
- Queries in <5ms on CPU (benchmarked)
- Has zero approximation error — every result is the true nearest neighbour
- Has no training step or centroid selection to manage
- Is trivial to rebuild deterministically

HNSW and IVF introduce recall loss, require tuning (n_links, n_probes, n_centroids), and provide speed improvements only at 100k+ documents. At our scale, they are pure complexity with no benefit.

### Why `S-PubMedBert-MS-MARCO` over `all-MiniLM-L6-v2` or `BiomedBERT`?

**Over `all-MiniLM-L6-v2` (general-purpose):** General models are trained on web text. Medical terminology is underrepresented: "myocardial infarction" and "heart attack" may not be close neighbours, and "consolidation" (a radiological finding) would map near its everyday English meaning. A biomedical-domain model is required for this application.

**Over `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (plain MLM):** BiomedBERT is a masked language model, not a sentence similarity model. When used with default mean pooling in `sentence-transformers`, it has not been trained with any retrieval objective. In practice, this produced severely compressed cosine similarity scores (all results in the 0.956–0.981 range, a spread of only 0.025) where the ranking was effectively random — "function of liver" retrieved "Heart belongs to Circulatory System" as the top result.

`S-PubMedBert-MS-MARCO` is PubMedBERT fine-tuned via contrastive learning on MS-MARCO (a large passage retrieval benchmark). It has both properties we need: biomedical vocabulary from pre-training and discriminative retrieval geometry from fine-tuning. After the swap, validation scores span 0.88–0.99 and all 10 test queries return semantically correct top-1 results.

### Why natural language templates instead of raw triples?

Embedding models are trained on natural language, not structured data notation. The cosine similarity between:
- `"Pneumonia#symptom#fever, cough"` (raw triple)
- `"What are the symptoms of pneumonia?"` (natural language query)

is significantly lower than between:
- `"Pneumonia symptoms include: fever, cough"` (templated)
- `"What are the symptoms of pneumonia?"` (natural language query)

The template bridges the distributional gap between structured KG data and natural language queries.

### Why the builder/reader split (`FAISSIndexer` vs `Retriever`)?

Index construction (embedding ~3,000 texts) is a batch offline operation that takes ~90s. Query serving is an online operation that must complete in milliseconds. Separating them:
- Prevents accidental index rebuilds during serving
- Allows the index to be built once and loaded by multiple retriever instances
- Makes the offline build script independent of the serving code

### Why store documents as JSONL alongside the FAISS binary?

FAISS stores only float32 vectors — it has no built-in mechanism for storing associated metadata. After a search, FAISS returns integer indices into the vector array. The `documents.jsonl` file maps those indices back to the full `Document` objects (text, provenance) by line position.

JSONL is preferred over a single JSON array because it supports line-by-line streaming and is robust to partial writes — each line is an independent, valid JSON object.

---

## 10. Known Limitations

| Limitation | Impact | Planned mitigation |
|-----------|--------|--------------------|
| Index must be fully rebuilt to add new documents | No incremental indexing | Acceptable at <10k docs; Phase 5+ may add IVF or HNSW with periodic full rebuild |
| 768-dim vectors; index is ~8.75MB for 3k docs | Scales linearly with doc count | At 100k docs (~250MB index), move to IVF |
| `retrieve_with_filter` post-filters in Python (not in FAISS) | May miss results if `top_k * 3` is insufficient | Acceptable at <5k docs; use FAISS ID selectors at scale |
| KG covers 302 diseases and 105 organs | Gaps for rare conditions | PubMed ingestion (Phase 2 extension) will fill gaps |
| No query expansion or re-ranking | Single-pass retrieval | Cross-encoder re-ranker can be layered in Phase 4 |
| Retrieval min_score threshold (0.3) is a heuristic | May filter valid results for rare queries | Tune per query type; consider adaptive thresholds |
