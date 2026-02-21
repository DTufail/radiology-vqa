import logging
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)

_PHI_FIELDS = frozenset(
    {
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "InstitutionName",
        "ReferringPhysicianName",
        "StudyDate",
        "AccessionNumber",
    }
)


def load_dicom(path: Path) -> dict:
    """Returns {"pixel_array": np.ndarray, "metadata": dict}"""
    ds = pydicom.dcmread(str(path))

    if hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
        logger.warning("Multi-frame DICOM rejected: %s", path)
        raise ValueError(f"Multi-frame DICOM not supported: {path}")

    pixel_array = ds.pixel_array.astype(np.float64)
    metadata = {
        elem.keyword: str(elem.value)
        for elem in ds
        if elem.keyword and not elem.keyword.startswith("Pixel")
    }

    return {"pixel_array": pixel_array, "metadata": metadata}


def apply_windowing(
    pixel_array: np.ndarray, window_center: float, window_width: float
) -> np.ndarray:
    """Apply DICOM windowing, return normalized uint8 array."""
    low = window_center - window_width / 2.0
    high = window_center + window_width / 2.0
    windowed = np.clip(pixel_array, low, high)
    if high == low:
        return np.zeros_like(windowed, dtype=np.uint8)
    normalized = ((windowed - low) / (high - low) * 255.0).astype(np.uint8)
    return normalized


def dicom_to_pil(path: Path) -> Image.Image:
    """Load DICOM → apply windowing → convert to PIL RGB Image."""
    ds = pydicom.dcmread(str(path))

    if hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
        logger.warning("Multi-frame DICOM rejected: %s", path)
        raise ValueError(f"Multi-frame DICOM not supported: {path}")

    pixel_array = ds.pixel_array.astype(np.float64)

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        # Handle multi-value tags (DS sequence)
        if hasattr(wc, "__iter__") and not isinstance(wc, str):
            wc = float(list(wc)[0])
        else:
            wc = float(wc)
        if hasattr(ww, "__iter__") and not isinstance(ww, str):
            ww = float(list(ww)[0])
        else:
            ww = float(ww)
        normalized = apply_windowing(pixel_array, wc, ww)
    else:
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax == pmin:
            normalized = np.zeros_like(pixel_array, dtype=np.uint8)
        else:
            normalized = ((pixel_array - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)

    return Image.fromarray(normalized).convert("RGB")


def anonymize_metadata(metadata: dict) -> dict:
    """Strip PHI fields, return cleaned copy."""
    return {k: v for k, v in metadata.items() if k not in _PHI_FIELDS}
