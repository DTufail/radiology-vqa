import pytest
from PIL import Image
from pydantic import ValidationError


def _make_image() -> Image.Image:
    return Image.new("RGB", (32, 32))


def test_vqa_sample_valid():
    from radiology_vqa.schema import VQASample

    sample = VQASample(
        image=_make_image(),
        question="Is this normal?",
        answer="yes",
        answer_type="closed",
        modality="xray",
        source="vqa_rad",
        sample_id="vqa_rad_train_0",
    )
    assert sample.question == "Is this normal?"
    assert sample.answer_type == "closed"


def test_vqa_sample_missing_question():
    from radiology_vqa.schema import VQASample

    with pytest.raises(ValidationError):
        VQASample(
            image=_make_image(),
            answer="yes",
            answer_type="closed",
            modality="xray",
            source="vqa_rad",
            sample_id="vqa_rad_train_0",
        )


def test_slake_sample_extra_fields():
    from radiology_vqa.schema import SLAKESample

    sample = SLAKESample(
        image=_make_image(),
        question="What organ?",
        answer="liver",
        answer_type="open",
        modality="ct",
        source="slake",
        sample_id="slake_train_1",
        location="abdomen",
        content_type="organ",
        triple=["liver", "is_a", "organ"],
        img_name="test.jpg",
    )
    assert sample.location == "abdomen"
    assert sample.content_type == "organ"
    assert sample.triple == ["liver", "is_a", "organ"]
    assert sample.img_name == "test.jpg"


def test_kg_triple_valid(sample_kg_triple):
    assert sample_kg_triple.head == "pneumonia"
    assert sample_kg_triple.relation == "is_a"
    assert sample_kg_triple.tail == "disease"
    assert sample_kg_triple.category == "disease"
