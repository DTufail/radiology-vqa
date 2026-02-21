import json

from radiology_vqa.schema import SLAKESample


def test_load_slake_english_only(tmp_slake_dir):
    from radiology_vqa.slake_loader import load_slake

    samples = load_slake(tmp_slake_dir, "train")
    # 3 English rows; Chinese row is filtered out
    assert len(samples) == 3
    assert all(isinstance(s, SLAKESample) for s in samples)


def test_slake_modality_normalized(tmp_slake_dir):
    from radiology_vqa.slake_loader import load_slake

    samples = load_slake(tmp_slake_dir, "train")
    for s in samples:
        assert s.modality == s.modality.lower()
        assert s.modality in {"xray", "ct", "mri", "pathology", "unknown"}


def test_slake_answer_type_normalized(tmp_slake_dir):
    from radiology_vqa.slake_loader import load_slake

    samples = load_slake(tmp_slake_dir, "train")
    for s in samples:
        assert s.answer_type in {"open", "closed"}


def test_slake_missing_image_skipped(tmp_slake_dir):
    # Append a row pointing to a non-existent image
    train_path = tmp_slake_dir / "train.json"
    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)

    data.append(
        {
            "qid": 99,
            "img_name": "nonexistent/image.jpg",
            "question": "Does this exist?",
            "answer": "no",
            "answer_type": "CLOSED",
            "q_lang": "en",
            "modality": "CT",
            "location": "",
            "content_type": "",
            "triple": [],
        }
    )
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    from radiology_vqa.slake_loader import load_slake

    samples = load_slake(tmp_slake_dir, "train")
    # 3 valid English rows; missing-image row skipped
    assert len(samples) == 3


def test_slake_sample_fields_populated(tmp_slake_dir):
    from radiology_vqa.slake_loader import load_slake

    samples = load_slake(tmp_slake_dir, "train")
    first = samples[0]
    assert first.location == "chest"
    assert first.content_type == "abnormality"
    assert first.img_name == "xmlab_test/source.jpg"
    assert first.source == "slake"
