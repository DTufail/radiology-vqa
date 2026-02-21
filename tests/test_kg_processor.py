from radiology_vqa.rag.kg_processor import KGProcessor
from radiology_vqa.schema import KGTriple


def _triple(head: str, relation: str, tail: str, category: str) -> KGTriple:
    return KGTriple(head=head, relation=relation, tail=tail, category=category)


def test_disease_symptom_template():
    processor = KGProcessor()
    triples = [_triple("Pneumonia", "symptom", "fever, cough", "disease")]
    docs = processor.process_diseases(triples)
    attr_docs = [d for d in docs if d.meta.attribute == "symptom"]
    assert len(attr_docs) == 1
    assert "symptoms include:" in attr_docs[0].text


def test_disease_cause_template():
    processor = KGProcessor()
    triples = [_triple("Tuberculosis", "cause", "Mycobacterium tuberculosis", "disease")]
    docs = processor.process_diseases(triples)
    attr_docs = [d for d in docs if d.meta.attribute == "cause"]
    assert len(attr_docs) == 1
    assert "is caused by:" in attr_docs[0].text


def test_organ_function_template():
    processor = KGProcessor()
    triples = [_triple("Liver", "function", "metabolize nutrients, detoxify blood", "organ")]
    docs = processor.process_organs(triples)
    attr_docs = [d for d in docs if d.meta.attribute == "function"]
    assert len(attr_docs) == 1
    assert "function of" in attr_docs[0].text.lower()


def test_organ_relation_template():
    processor = KGProcessor()
    triples = [_triple("Liver", "belong to", "Digestive System", "organ_rel")]
    docs = processor.process_organ_relations(triples)
    assert len(docs) == 1
    assert "belongs to the" in docs[0].text


def test_disease_summary_contains_all_attributes():
    processor = KGProcessor()
    triples = [
        _triple("Lobar Pneumonia", "symptom", "chills, fever, chest pain", "disease"),
        _triple("Lobar Pneumonia", "cause", "streptococcus pneumoniae", "disease"),
        _triple("Lobar Pneumonia", "treatment", "antibiotic therapy", "disease"),
    ]
    docs = processor.process_diseases(triples)
    summary_docs = [d for d in docs if d.meta.attribute == "summary"]
    assert len(summary_docs) == 1
    summary_text = summary_docs[0].text
    assert "chills, fever, chest pain" in summary_text
    assert "streptococcus pneumoniae" in summary_text
    assert "antibiotic therapy" in summary_text


def test_all_doc_ids_unique():
    processor = KGProcessor()
    # Mix of all categories
    triples = []
    for i in range(35):
        triples.append(_triple(f"Disease{i}", "symptom", f"symptom{i}", "disease"))
        triples.append(_triple(f"Organ{i}", "function", f"func{i}", "organ"))
        triples.append(_triple(f"Organ{i}", "belong to", f"System{i}", "organ_rel"))

    docs = processor.process_all(triples)
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids found"


def test_correct_source_types():
    processor = KGProcessor()
    triples = [
        _triple("Pneumonia", "symptom", "fever", "disease"),
        _triple("Liver", "function", "metabolize", "organ"),
        _triple("Liver", "belong to", "Digestive System", "organ_rel"),
    ]
    docs = processor.process_all(triples)
    source_types = {d.meta.source_type for d in docs}
    assert "kg_disease" in source_types
    assert "kg_organ" in source_types
    assert "kg_organ_rel" in source_types
