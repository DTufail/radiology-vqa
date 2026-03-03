"""Prepare cached demo data from evaluation results.

Reads the calibrated agent evaluation JSON (Config 6) and extracts the six
preloaded example cases, downloads the corresponding VQA-RAD images from
HuggingFace, and writes demo_cache.json + copies images to demo/images/.

Usage (full pipeline — needs eval JSON from SageMaker):
    python scripts/prepare_demo_cache.py \\
        --eval-json data/evaluation_reports/phase6c_agent/agent_vqa_rad_test_*.json \\
        --output-dir demo/

Usage (images only — no eval JSON, uses hardcoded demo data):
    python scripts/prepare_demo_cache.py --images-only --output-dir demo/

The script requires:
    pip install datasets Pillow

It does NOT require torch, transformers, or any GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Example case definitions ──────────────────────────────────────────────────
# These are the six cases that tell the full story of the pipeline.
# Indices into the VQA-RAD *test* split (0-based).
EXAMPLE_CASES = [
    {
        "sample_id":  "vqa_rad_test_2",
        "split_index": 2,
        "story": "Confident correct answer with grounding citations — system working as intended",
    },
    {
        "sample_id":  "vqa_rad_test_22",
        "split_index": 22,
        "story": "Correct open-ended answer: system identifies imaging plane and grounds it in RadLex",
    },
    {
        "sample_id":  "vqa_rad_test_4",
        "split_index": 4,
        "story": (
            "HERO CASE: Hallucination caught — VLM said 'symmetrical to bone marrow' "
            "but ground truth is 'not seen here'. The supervisor abstained because the "
            "answer could not be grounded."
        ),
    },
    {
        "sample_id":  "vqa_rad_test_7",
        "split_index": 7,
        "story": "Low-confidence abstention — system correctly declines when VLM is uncertain",
    },
    {
        "sample_id":  "vqa_rad_test_0",
        "split_index": 0,
        "story": (
            "Honest limitation: system answered confidently but incorrectly. "
            "High calibrated confidence does not guarantee correctness — "
            "clinical review is always required."
        ),
    },
    {
        "sample_id":  "vqa_rad_test_10",
        "split_index": 10,
        "story": "Confident correct closed answer — imaging plane identification grounded in RadLex",
    },
]

EXAMPLE_IDS = {c["sample_id"] for c in EXAMPLE_CASES}


# ── Image download ─────────────────────────────────────────────────────────────

def download_images(output_image_dir: Path) -> dict[str, Path]:
    """Download example images from the VQA-RAD HuggingFace dataset.

    Returns a dict mapping sample_id → local image path.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    try:
        from PIL import Image as PILImage
    except ImportError:
        log.error("Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    output_image_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading VQA-RAD test split from HuggingFace…")
    dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
    log.info("Loaded %d test samples.", len(dataset))

    id_to_path: dict[str, Path] = {}
    for case in EXAMPLE_CASES:
        sample_id   = case["sample_id"]
        split_index = case["split_index"]
        out_path    = output_image_dir / f"{sample_id}.jpg"

        if out_path.exists():
            log.info("  %s already exists — skipping.", out_path.name)
            id_to_path[sample_id] = out_path
            continue

        if split_index >= len(dataset):
            log.warning("  Index %d out of range (dataset has %d samples).", split_index, len(dataset))
            continue

        row = dataset[split_index]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(out_path, "JPEG", quality=90)
        log.info("  Saved %s (index=%d, q=%r)", out_path.name, split_index, row["question"][:60])
        id_to_path[sample_id] = out_path

    return id_to_path


# ── Eval JSON loading ──────────────────────────────────────────────────────────

def load_eval_json(eval_json_path: Path) -> dict[str, dict]:
    """Load per-sample results from an EvaluationResult JSON.

    Returns a dict mapping sample_id → PerSampleResult dict.
    """
    log.info("Loading eval JSON: %s", eval_json_path)
    with open(eval_json_path) as f:
        data = json.load(f)

    by_id: dict[str, dict] = {}
    for sample in data.get("per_sample", []):
        by_id[sample["sample_id"]] = sample

    log.info("  Found %d per-sample results.", len(by_id))
    return by_id


# ── Merge eval data with case metadata ────────────────────────────────────────

def build_demo_entry(
    case:      dict,
    eval_data: dict[str, dict] | None,
    image_dir: Path,
    fallback:  dict | None = None,
) -> dict:
    """Build a single demo_cache.json entry.

    Prefers real eval data when available; falls back to the hardcoded
    representative data from the existing demo_cache.json.
    """
    sample_id  = case["sample_id"]
    image_file = f"images/{sample_id}.jpg"

    if eval_data and sample_id in eval_data:
        s = eval_data[sample_id]
        entry = {
            "sample_id":       sample_id,
            "question":        s["question"],
            "ground_truth":    s["ground_truth"],
            "prediction":      s["prediction"],
            "correct":         s["correct"],
            "answer_type":     s["answer_type"],
            "decision":        s.get("decision", ""),
            "confidence":      s["confidence"],
            "raw_confidence":  s["confidence"],   # EvaluationResult doesn't store raw separately
            "visual_answer":   s.get("visual_answer", ""),
            "retrieval_query": s.get("retrieval_query", ""),
            "reasoning":       s.get("reasoning", ""),
            "retry_count":     s.get("retry_count", 0),
            "image_file":      image_file,
            "story":           case["story"],
            "citations":       s.get("citations", []),
        }
        log.info("  %s — loaded from eval JSON.", sample_id)
    elif fallback:
        # Use the hardcoded representative data, just update the image_file path
        entry = dict(fallback)
        entry["image_file"] = image_file
        entry["story"]      = case["story"]
        log.info("  %s — using hardcoded fallback data.", sample_id)
    else:
        log.warning("  %s — no eval data and no fallback; skipping.", sample_id)
        return None

    return entry


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Path to the calibrated agent EvaluationResult JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo"),
        help="Output directory (default: demo/). Images go to <output-dir>/images/.",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only download images; do not overwrite demo_cache.json.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image download (only rebuild demo_cache.json from eval JSON).",
    )
    args = parser.parse_args()

    output_dir   = args.output_dir.resolve()
    image_dir    = output_dir / "images"
    cache_path   = output_dir / "demo_cache.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download images ──────────────────────────────────────────
    id_to_image_path: dict[str, Path] = {}
    if not args.no_images:
        log.info("=== Step 1: Downloading VQA-RAD images ===")
        id_to_image_path = download_images(image_dir)
    else:
        log.info("Skipping image download (--no-images).")

    if args.images_only:
        log.info("Done (--images-only). Images saved to %s", image_dir)
        return

    # ── Step 2: Load eval JSON ───────────────────────────────────────────
    eval_data: dict[str, dict] | None = None
    if args.eval_json:
        if not args.eval_json.exists():
            # Try glob expansion
            matches = list(args.eval_json.parent.glob(args.eval_json.name))
            if matches:
                args.eval_json = sorted(matches)[-1]  # newest
                log.info("Resolved glob to: %s", args.eval_json)
            else:
                log.warning("Eval JSON not found: %s — using fallback data.", args.eval_json)
        if args.eval_json.exists():
            eval_data = load_eval_json(args.eval_json)

    # ── Step 3: Load existing fallback data ─────────────────────────────
    fallback_by_id: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            existing = json.load(f)
        fallback_by_id = {e["sample_id"]: e for e in existing.get("examples", [])}
        log.info("Loaded %d fallback entries from existing demo_cache.json.", len(fallback_by_id))

    # ── Step 4: Build demo entries ───────────────────────────────────────
    log.info("=== Step 2: Building demo_cache.json ===")
    examples = []
    for case in EXAMPLE_CASES:
        entry = build_demo_entry(
            case      = case,
            eval_data = eval_data,
            image_dir = image_dir,
            fallback  = fallback_by_id.get(case["sample_id"]),
        )
        if entry is not None:
            examples.append(entry)

    # ── Step 5: Write demo_cache.json ────────────────────────────────────
    cache_data = {"examples": examples}
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    log.info("Wrote %d examples to %s", len(examples), cache_path)

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=== Summary ===")
    log.info("  Images:         %s", image_dir)
    log.info("  Demo cache:     %s", cache_path)
    log.info("  Examples:       %d / %d", len(examples), len(EXAMPLE_CASES))
    log.info("")
    log.info("To launch the demo:")
    log.info("  cd demo && python app.py")
    log.info("  # or:  DEMO_MODE=cached python demo/app.py")


if __name__ == "__main__":
    main()
