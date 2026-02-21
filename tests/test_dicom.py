import numpy as np
from PIL import Image


def test_apply_windowing_range():
    from radiology_vqa.dicom_handler import apply_windowing

    pixel_array = np.linspace(0, 255, 256, dtype=np.float64)
    result = apply_windowing(pixel_array, window_center=128.0, window_width=256.0)
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_apply_windowing_clips():
    from radiology_vqa.dicom_handler import apply_windowing

    # Window: center=128, width=128 → low=64, high=192
    pixel_array = np.array([0.0, 50.0, 128.0, 200.0, 300.0])
    result = apply_windowing(pixel_array, window_center=128.0, window_width=128.0)
    assert result[0] == 0    # 0 < 64 → clipped to low
    assert result[4] == 255  # 300 > 192 → clipped to high


def test_anonymize_metadata_strips_phi():
    from radiology_vqa.dicom_handler import anonymize_metadata

    metadata = {
        "PatientName": "John Doe",
        "PatientID": "12345",
        "Modality": "CR",
        "StudyDate": "20240101",
        "InstitutionName": "Test Hospital",
        "SOPClassUID": "1.2.840.10008.5.1.4.1.1.7",
    }
    cleaned = anonymize_metadata(metadata)
    assert "PatientName" not in cleaned
    assert "PatientID" not in cleaned
    assert "StudyDate" not in cleaned
    assert "InstitutionName" not in cleaned
    assert cleaned["Modality"] == "CR"
    assert cleaned["SOPClassUID"] == "1.2.840.10008.5.1.4.1.1.7"


def test_load_dicom_returns_dict(synthetic_dicom):
    from radiology_vqa.dicom_handler import load_dicom

    result = load_dicom(synthetic_dicom)
    assert "pixel_array" in result
    assert "metadata" in result
    assert isinstance(result["pixel_array"], np.ndarray)
    assert isinstance(result["metadata"], dict)


def test_dicom_to_pil_returns_rgb(synthetic_dicom):
    from radiology_vqa.dicom_handler import dicom_to_pil

    image = dicom_to_pil(synthetic_dicom)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
