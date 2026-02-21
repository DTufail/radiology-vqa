import json

from radiology_vqa.rag.document import Document, DocumentMeta
from radiology_vqa.rag.indexer import FAISSIndexer


def _make_doc(i: int, source_type: str = "kg_disease") -> Document:
    return Document(
        text=f"Medical fact number {i} about a condition.",
        meta=DocumentMeta(
            source_type=source_type,
            entity_name=f"Entity{i}",
            attribute="symptom",
            source_file="en_disease.csv",
        ),
        doc_id=f"{source_type}_entity{i}_symptom_0",
    )


def test_build_index_size(mock_embedder, tmp_index_dir):
    docs = [_make_doc(i) for i in range(5)]
    indexer = FAISSIndexer(mock_embedder)
    indexer.build_index(docs)
    assert indexer._index.ntotal == 5


def test_save_creates_all_files(mock_embedder, tmp_index_dir):
    docs = [_make_doc(i) for i in range(3)]
    indexer = FAISSIndexer(mock_embedder)
    indexer.build_index(docs)
    indexer.save(tmp_index_dir)

    assert (tmp_index_dir / "index.faiss").exists()
    assert (tmp_index_dir / "documents.jsonl").exists()
    assert (tmp_index_dir / "index_meta.json").exists()


def test_load_documents_match(mock_embedder, tmp_index_dir):
    docs = [_make_doc(i) for i in range(4)]
    indexer = FAISSIndexer(mock_embedder)
    indexer.build_index(docs)
    indexer.save(tmp_index_dir)

    _, loaded_docs, _ = FAISSIndexer.load(tmp_index_dir)
    assert len(loaded_docs) == 4
    for original, loaded in zip(docs, loaded_docs):
        assert original.doc_id == loaded.doc_id
        assert original.text == loaded.text


def test_index_meta_fields(mock_embedder, tmp_index_dir):
    docs = [_make_doc(i) for i in range(2)]
    indexer = FAISSIndexer(mock_embedder)
    indexer.build_index(docs)
    indexer.save(tmp_index_dir)

    with open(tmp_index_dir / "index_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    assert "doc_count" in meta
    assert "dimension" in meta
    assert "built_at" in meta
    assert meta["doc_count"] == 2
