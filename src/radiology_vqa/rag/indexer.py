import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from radiology_vqa.rag.document import Document
from radiology_vqa.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """Builds and persists a FAISS vector index from Documents."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index = None
        self._documents: list[Document] = []

    def build_index(self, documents: list[Document]) -> None:
        """Embed all document texts and build FAISS IndexFlatIP."""
        import faiss

        if not documents:
            logger.warning("No documents provided — building empty index.")
            # CPU index — search over ~3K docs is <1ms; faiss-gpu unnecessary at this scale
            self._index = faiss.IndexFlatIP(self._embedder.dimension)
            self._documents = []
            return

        texts = [doc.text for doc in documents]

        t0 = time.perf_counter()
        embeddings = self._embedder.embed_texts(texts)
        t_embed = time.perf_counter() - t0
        logger.info(
            "Embedded %d documents (dim=%d) in %.2fs",
            len(documents),
            embeddings.shape[1],
            t_embed,
        )

        t1 = time.perf_counter()
        # CPU index — search over ~3K docs is <1ms; faiss-gpu unnecessary at this scale
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings.astype(np.float32))
        t_index = time.perf_counter() - t1
        logger.info(
            "Built FAISS IndexFlatIP (%d vectors) in %.2fs", len(documents), t_index
        )

        self._documents = list(documents)

    def save(self, index_dir: Path) -> None:
        """Persist index, documents, and metadata to disk."""
        import faiss

        if self._index is None:
            raise RuntimeError("Call build_index() before save().")

        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_dir / "index.faiss"))

        with open(index_dir / "documents.jsonl", "w", encoding="utf-8") as f:
            for doc in self._documents:
                f.write(doc.model_dump_json() + "\n")

        meta = {
            "doc_count": len(self._documents),
            "embedding_model": self._embedder.model_name,
            "dimension": self._embedder.dimension,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(index_dir / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved index to %s (%d docs)", index_dir, len(self._documents))

    @classmethod
    def load(cls, index_dir: Path) -> tuple:
        """Load and return (faiss_index, documents_list, meta_dict)."""
        import faiss

        faiss_path = index_dir / "index.faiss"
        docs_path = index_dir / "documents.jsonl"
        meta_path = index_dir / "index_meta.json"

        for p in (faiss_path, docs_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Index file not found: {p}")

        index = faiss.read_index(str(faiss_path))

        documents: list[Document] = []
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(Document.model_validate_json(line))

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        return index, documents, meta
