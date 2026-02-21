import pytest

from radiology_vqa.schema import VQASample


@pytest.mark.slow
def test_load_vqa_rad_train():
    from radiology_vqa.loader import load_vqa_rad

    samples = load_vqa_rad("train")
    assert len(samples) > 0
    assert all(isinstance(s, VQASample) for s in samples)


@pytest.mark.slow
def test_vqa_rad_non_empty_fields():
    from radiology_vqa.loader import load_vqa_rad

    samples = load_vqa_rad("train")
    for s in samples:
        assert s.question, f"Empty question in {s.sample_id}"
        assert s.answer, f"Empty answer in {s.sample_id}"


@pytest.mark.slow
def test_vqa_rad_answer_types():
    from radiology_vqa.loader import load_vqa_rad

    samples = load_vqa_rad("train")
    answer_types = {s.answer_type for s in samples}
    assert answer_types.issubset({"closed", "open"})


@pytest.mark.slow
def test_pathvqa_images_rgb():
    from radiology_vqa.loader import load_pathvqa

    samples = load_pathvqa("train")
    assert len(samples) > 0
    for s in samples:
        assert s.image.mode == "RGB", f"Non-RGB image in {s.sample_id}: {s.image.mode}"
