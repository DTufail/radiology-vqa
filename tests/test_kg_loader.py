def test_load_kg_parses_csv(tmp_slake_dir):
    from radiology_vqa.kg_loader import load_kg

    triples = load_kg(tmp_slake_dir)
    # en_disease.csv has 2 data rows; en_organ.csv and en_organ_rel.csv are absent
    assert len(triples) == 2


def test_load_kg_skips_dot_files(tmp_slake_dir):
    from radiology_vqa.kg_loader import load_kg

    triples = load_kg(tmp_slake_dir)
    # ._en_disease.csv must be ignored; all categories must be valid
    assert all(t.category in {"disease", "organ", "organ_rel"} for t in triples)


def test_load_kg_triple_fields(tmp_slake_dir):
    from radiology_vqa.kg_loader import load_kg

    triples = load_kg(tmp_slake_dir)
    assert triples[0].head == "pneumonia"
    assert triples[0].relation == "is_a"
    assert triples[0].tail == "disease"
    assert triples[0].category == "disease"
