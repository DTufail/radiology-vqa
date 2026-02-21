"""Tests for VLMInterface protocol and VLMPrediction data model."""

import pytest
from pydantic import ValidationError

from radiology_vqa.vlm.interface import VLMInterface, VLMPrediction


# ---------------------------------------------------------------------------
# VLMPrediction schema validation
# ---------------------------------------------------------------------------


def test_vlm_prediction_valid_fields():
    p = VLMPrediction(
        answer="yes",
        confidence=0.8,
        raw_output="yes",
        model_name="test-model",
        latency_seconds=0.25,
    )
    assert p.answer == "yes"
    assert p.confidence == pytest.approx(0.8)
    assert p.model_name == "test-model"
    assert p.latency_seconds == pytest.approx(0.25)


def test_vlm_prediction_confidence_boundary_zero():
    p = VLMPrediction(
        answer="no", confidence=0.0, raw_output="no", model_name="m", latency_seconds=0.1
    )
    assert p.confidence == 0.0


def test_vlm_prediction_confidence_boundary_one():
    p = VLMPrediction(
        answer="yes", confidence=1.0, raw_output="yes", model_name="m", latency_seconds=0.1
    )
    assert p.confidence == 1.0


def test_vlm_prediction_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        VLMPrediction(
            answer="yes", confidence=1.1, raw_output="yes", model_name="m", latency_seconds=0.1
        )


def test_vlm_prediction_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        VLMPrediction(
            answer="yes",
            confidence=-0.01,
            raw_output="yes",
            model_name="m",
            latency_seconds=0.1,
        )


def test_vlm_prediction_rejects_empty_answer():
    with pytest.raises(ValidationError):
        VLMPrediction(
            answer="", confidence=0.5, raw_output="", model_name="m", latency_seconds=0.1
        )


# ---------------------------------------------------------------------------
# VLMInterface protocol check (runtime_checkable)
# ---------------------------------------------------------------------------


def test_mock_backend_satisfies_vlm_interface(mock_vlm_backend):
    assert isinstance(mock_vlm_backend, VLMInterface)


def test_mock_backend_predict_returns_vlm_prediction(mock_vlm_backend):
    from PIL import Image

    img = Image.new("RGB", (16, 16))
    result = mock_vlm_backend.predict(img, "Is there a fracture?")

    assert isinstance(result, VLMPrediction)
    assert result.answer == "yes"
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_seconds >= 0.0


def test_mock_backend_predict_batch(mock_vlm_backend):
    from PIL import Image

    img = Image.new("RGB", (16, 16))
    samples = [(img, "question one"), (img, "question two")]
    results = mock_vlm_backend.predict_batch(samples)

    assert len(results) == 2
    assert all(isinstance(r, VLMPrediction) for r in results)


def test_mock_backend_model_name(mock_vlm_backend):
    assert isinstance(mock_vlm_backend.model_name, str)
    assert len(mock_vlm_backend.model_name) > 0
