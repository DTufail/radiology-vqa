import pytest

from radiology_vqa.rag.document import Document, DocumentMeta
from radiology_vqa.rag.indexer import FAISSIndexer


def _make_doc(entity: str, attribute: str, text: str, source_type: str = "kg_organ") -> Document:
    slug = entity.lower().replace(" ", "_")
    return Document(
        text=text,
        meta=DocumentMeta(
            source_type=source_type,
            entity_name=entity,
            attribute=attribute,
            source_file="en_organ.csv",
        ),
        doc_id=f"{source_type}_{slug}_{attribute}_0",
    )


MEDICAL_DOCS = [
    _make_doc("Liver", "function", "The function of the liver: metabolize nutrients, detoxify blood, produce bile."),
    _make_doc("Lung", "function", "The function of the lung: gas exchange, oxygen intake, carbon dioxide removal."),
    _make_doc("Heart", "function", "The function of the heart: pump blood throughout the body."),
    _make_doc("Brain", "function", "The function of the brain: control body functions, cognition, memory."),
    _make_doc("Kidney", "function", "The function of the kidney: filter blood, produce urine, regulate electrolytes."),
    _make_doc("Stomach", "function", "The function of the stomach: digest food, secrete gastric acid and enzymes."),
    _make_doc("Pancreas", "function", "The function of the pancreas: secrete insulin, glucagon, and digestive enzymes."),
    _make_doc("Liver", "belong_to", "Liver belongs to the Digestive System.", source_type="kg_organ_rel"),
    _make_doc("Lung", "symptom", "Pneumonia symptoms include: fever, cough, shortness of breath.", source_type="kg_disease"),
    _make_doc("Bone", "definition", "Bone: rigid connective tissue forming the skeleton for structural support.", source_type="kg_organ"),
]


@pytest.mark.slow
def test_retrieve_liver_function(tmp_index_dir):
    from radiology_vqa.rag.embedder import Embedder
    from radiology_vqa.rag.retriever import Retriever

    embedder = Embedder()
    indexer = FAISSIndexer(embedder)
    indexer.build_index(MEDICAL_DOCS)
    indexer.save(tmp_index_dir)

    retriever = Retriever(tmp_index_dir, embedder=embedder)
    results = retriever.retrieve("liver function", top_k=3)

    assert len(results) > 0
    assert results[0].document.meta.entity_name == "Liver"


@pytest.mark.slow
def test_retrieve_high_min_score_fewer_results(tmp_index_dir):
    from radiology_vqa.rag.embedder import Embedder
    from radiology_vqa.rag.retriever import Retriever

    embedder = Embedder()
    indexer = FAISSIndexer(embedder)
    indexer.build_index(MEDICAL_DOCS)
    indexer.save(tmp_index_dir)

    retriever = Retriever(tmp_index_dir, embedder=embedder)
    default_results = retriever.retrieve("organ function", top_k=5, min_score=0.0)
    strict_results = retriever.retrieve("organ function", top_k=5, min_score=0.99)
    assert len(strict_results) <= len(default_results)


@pytest.mark.slow
def test_retrieve_with_filter_source_type(tmp_index_dir):
    from radiology_vqa.rag.embedder import Embedder
    from radiology_vqa.rag.retriever import Retriever

    embedder = Embedder()
    indexer = FAISSIndexer(embedder)
    indexer.build_index(MEDICAL_DOCS)
    indexer.save(tmp_index_dir)

    retriever = Retriever(tmp_index_dir, embedder=embedder)
    results = retriever.retrieve_with_filter("organ function", top_k=5, source_type="kg_organ")
    assert len(results) > 0
    assert all(r.document.meta.source_type == "kg_organ" for r in results)


@pytest.mark.slow
def test_retrieve_empty_index(tmp_index_dir):
    from radiology_vqa.rag.embedder import Embedder
    from radiology_vqa.rag.retriever import Retriever

    embedder = Embedder()
    indexer = FAISSIndexer(embedder)
    indexer.build_index([])
    indexer.save(tmp_index_dir)

    retriever = Retriever(tmp_index_dir, embedder=embedder)
    results = retriever.retrieve("anything", top_k=5)
    assert results == []
