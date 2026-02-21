"""Tests for the VLM backend factory."""

import pytest

from radiology_vqa.vlm.interface import VLMInterface


def test_factory_unknown_backend_raises():
    from radiology_vqa.config import Settings
    from radiology_vqa.vlm.factory import create_vlm_backend

    config = Settings(vlm_backend="not_a_real_backend")
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        create_vlm_backend(config)


@pytest.mark.slow
def test_factory_llava_med_returns_vlm_interface():
    from radiology_vqa.config import Settings
    from radiology_vqa.vlm.factory import create_vlm_backend

    config = Settings(vlm_backend="llava_med", vlm_quantize="none")
    backend = create_vlm_backend(config)
    assert isinstance(backend, VLMInterface)


@pytest.mark.slow
def test_factory_blip2_returns_vlm_interface():
    from radiology_vqa.config import Settings
    from radiology_vqa.vlm.factory import create_vlm_backend

    config = Settings(vlm_backend="blip2", vlm_quantize="none")
    backend = create_vlm_backend(config)
    assert isinstance(backend, VLMInterface)
