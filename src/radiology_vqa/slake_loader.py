import json
import logging
from pathlib import Path

from PIL import Image

from radiology_vqa.schema import SLAKESample

logger = logging.getLogger(__name__)

_SPLIT_FILE_MAP: dict[str, str] = {
    "train": "train.json",
    "validation": "validate.json",
    "test": "test.json",
}

_MODALITY_MAP: dict[str, str] = {
    "X-Ray": "xray",
    "CT": "ct",
    "MRI": "mri",
}

_ANSWER_TYPE_MAP: dict[str, str] = {
    "OPEN": "open",
    "CLOSED": "closed",
}

# Module-level image cache: persists across load_slake() calls so train -> test
# loads do not re-read shared images from disk.  Keyed by resolved absolute
# path so different slake_dir values stay independent.
# Images are shared by reference -- PIL read ops are safe; callers must not
# mutate (resize/crop in-place) without first calling image.copy().
_IMAGE_CACHE: dict[str, Image.Image] = {}


def load_slake(slake_dir: Path, split: str = "train") -> list[SLAKESample]:
    split_file = _SPLIT_FILE_MAP.get(split)
    if split_file is None:
        raise ValueError(f"Unknown split: {split!r}. Expected one of {list(_SPLIT_FILE_MAP)}")

    json_path = slake_dir / split_file
    if not json_path.exists():
        raise FileNotFoundError(
            f"SLAKE split file not found: {json_path}. "
            "Ensure Slake1.0/ is placed at the configured slake_dir."
        )

    try:
        with open(json_path, encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to parse SLAKE JSON {json_path}: {e}") from e

    total_rows = len(rows)
    english_rows = [r for r in rows if r.get("q_lang") == "en"]
    logger.info("SLAKE %s: %d total rows, %d English", split, total_rows, len(english_rows))

    samples: list[SLAKESample] = []

    for row in english_rows:
        img_name = row.get("img_name", "")
        img_path = slake_dir / "imgs" / img_name
        cache_key = str(img_path.resolve())

        if cache_key in _IMAGE_CACHE:
            image = _IMAGE_CACHE[cache_key]
        else:
            if not img_path.exists():
                logger.warning("SLAKE image not found, skipping: %s", img_path)
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                _IMAGE_CACHE[cache_key] = image
            except Exception as e:
                logger.warning("Failed to load SLAKE image %s: %s, skipping", img_path, e)
                continue

        modality = _MODALITY_MAP.get(row.get("modality", ""), "unknown")
        answer_type = _ANSWER_TYPE_MAP.get(str(row.get("answer_type", "OPEN")).upper(), "open")

        triple_raw = row.get("triple", [])
        if isinstance(triple_raw, list):
            triple = [str(t) for t in triple_raw]
        elif isinstance(triple_raw, str):
            triple = [triple_raw] if triple_raw else []
        else:
            triple = []

        question = row.get("question", "")
        answer = str(row.get("answer", ""))

        # Skip samples with empty/whitespace-only answers (e.g. qid 1622)
        if not answer.strip():
            logger.warning("SLAKE %s: skipping qid=%s — empty answer", split, row.get("qid", "?"))
            continue

        sample = SLAKESample(
            image=image,
            question=question,
            answer=answer,
            answer_type=answer_type,
            modality=modality,
            source="slake",
            sample_id=f"slake_{split}_{row.get('qid', '')}",
            location=row.get("location", ""),
            content_type=row.get("content_type", ""),
            triple=triple,
            img_name=img_name,
        )
        samples.append(sample)

    # Deduplicate exact (img_name, question, answer) triples — keeps first occurrence
    seen: set[tuple[str, str, str]] = set()
    deduped: list[SLAKESample] = []
    for s in samples:
        key = (s.img_name, s.question.strip().lower(), s.answer.strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    if len(deduped) < len(samples):
        logger.info(
            "SLAKE %s: removed %d duplicate triples (%d → %d)",
            split, len(samples) - len(deduped), len(samples), len(deduped),
        )
        samples = deduped

    unique_images = len({s.img_name for s in samples})
    logger.info("SLAKE %s: %d samples, %d unique images", split, len(samples), unique_images)
    return samples
