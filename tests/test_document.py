import pytest
from pydantic import ValidationError

from radiology_vqa.rag.document import Document, DocumentMeta, RetrievalResult


def _meta(**kwargs) -> DocumentMeta:
    defaults = dict(
        source_type="kg_disease",
        entity_name="Pneumonia",
        attribute="symptom",
        source_file="en_disease.csv",
    )
    defaults.update(kwargs)
    return DocumentMeta(**defaults)


def test_document_meta_valid():
    meta = _meta()
    assert meta.source_type == "kg_disease"
    assert meta.chunk_index == 0


def test_document_valid():
    doc = Document(
        text="Pneumonia symptoms include: fever, cough.",
        meta=_meta(),
        doc_id="kg_disease_pneumonia_symptom_0",
    )
    assert doc.text.startswith("Pneumonia")
    assert doc.doc_id == "kg_disease_pneumonia_symptom_0"


def test_document_empty_text_raises():
    with pytest.raises(ValidationError):
        Document(text="", meta=_meta(), doc_id="kg_disease_pneumonia_symptom_0")


def test_retrieval_result_valid():
    doc = Document(
        text="Pneumonia symptoms include: fever, cough.",
        meta=_meta(),
        doc_id="kg_disease_pneumonia_symptom_0",
    )
    result = RetrievalResult(document=doc, score=0.92, rank=1)
    assert result.score == pytest.approx(0.92)
    assert result.rank == 1
    assert result.document.meta.entity_name == "Pneumonia"
