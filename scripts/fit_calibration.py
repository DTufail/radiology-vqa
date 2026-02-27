#!/usr/bin/env python3
"""Phase 6C — Fit confidence calibrators on a held-out validation set.

Procedure
---------
1. Load the fine-tuned LLaVA-Next 7B with LoRA adapter.
2. Run inference on SLAKE validation set (1,053 samples) — held-out during training.
3. Collect (raw_confidence, correct) pairs.
4. Fit both Platt scaling and isotonic regression calibrators.
5. Report ECE before/after for both methods, pick the better one.
6. Save calibrator to ``data/calibration/``.
7. Optionally evaluate calibration transfer on VQA-RAD test set.

CRITICAL: Always fit on the validation set; never on the test set.

Usage
-----
    # On SageMaker with fine-tuned adapter:
    VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \\
      python scripts/fit_calibration.py \\
        --val-dataset slake \\
        --val-split validation \\
        --test-dataset vqa_rad \\
        --test-split test \\
        --output data/calibration/

    # Without adapter (zero-shot calibration baseline):
    python scripts/fit_calibration.py \\
        --val-dataset slake --val-split validation --output data/calibration/
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fit_calibration")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit Platt/isotonic calibrators for Phase 6C.")
    p.add_argument(
        "--val-dataset",
        default="slake",
        choices=["slake", "vqa_rad", "pathvqa"],
        help="Dataset to use for calibration fitting (default: slake).",
    )
    p.add_argument(
        "--val-split",
        default="validation",
        help="Split to use for calibration fitting (default: validation).",
    )
    p.add_argument(
        "--test-dataset",
        default=None,
        choices=["slake", "vqa_rad", "pathvqa", None],
        help="Optional dataset to evaluate calibration transfer on.",
    )
    p.add_argument(
        "--test-split",
        default="test",
        help="Split to use for calibration transfer evaluation (default: test).",
    )
    p.add_argument(
        "--output",
        default="data/calibration",
        help="Output directory for calibrator JSON files (default: data/calibration/).",
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap on validation samples (default: all). Useful for quick testing.",
    )
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Cap on test samples for transfer evaluation (default: all).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: use environment / pydantic defaults).",
    )
    return p.parse_args()


# ── dataset loading ───────────────────────────────────────────────────────────

def _load_dataset(dataset_name: str, split: str):
    """Load a dataset split and return a list of VQASample objects."""
    logger.info("Loading %s / %s ...", dataset_name, split)

    if dataset_name == "slake":
        from radiology_vqa.slake_loader import load_slake
        from radiology_vqa.config import settings
        samples = load_slake(settings.slake_dir, split=split)  # English filtered internally

    elif dataset_name == "vqa_rad":
        from radiology_vqa.loader import load_vqa_rad
        samples = load_vqa_rad(split=split)

    elif dataset_name == "pathvqa":
        from radiology_vqa.loader import load_pathvqa
        samples = load_pathvqa(split=split)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info("Loaded %d samples from %s/%s", len(samples), dataset_name, split)
    return samples


# ── inference loop ────────────────────────────────────────────────────────────

def _run_inference(vlm, samples, max_samples=None, desc="inference"):
    """Run VLM inference and collect (raw_confidence, correct) pairs.

    NOTE: We bypass the calibrator during this step (calling _extract_confidence
    directly) so we always collect raw scores, regardless of the backend config.
    If the VLM was loaded without calibration, this is identical to calling
    predict() normally.
    """
    from radiology_vqa.benchmark.metrics import normalize_answer, is_match

    if max_samples is not None:
        samples = samples[:max_samples]

    confidences = []
    corrects = []
    total = len(samples)

    logger.info("Running %s on %d samples ...", desc, total)
    t0 = time.perf_counter()

    for i, sample in enumerate(samples, 1):
        try:
            pred = vlm.predict(sample.image, sample.question)
            # Use raw_confidence if calibration was applied; else use confidence.
            raw_conf = pred.raw_confidence if pred.raw_confidence is not None else pred.confidence
            correct = is_match(
                normalize_answer(pred.answer),
                normalize_answer(sample.answer),
                sample.answer_type,
            )
            confidences.append(raw_conf)
            corrects.append(correct)
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            continue

        if i % 100 == 0 or i == total:
            elapsed = time.perf_counter() - t0
            logger.info(
                "  %d/%d  elapsed=%.1fs  acc_so_far=%.3f",
                i,
                total,
                elapsed,
                sum(corrects) / len(corrects) if corrects else 0.0,
            )

    logger.info(
        "%s complete: %d samples, accuracy=%.3f, mean_conf=%.3f",
        desc,
        len(corrects),
        sum(corrects) / len(corrects) if corrects else 0.0,
        sum(confidences) / len(confidences) if confidences else 0.0,
    )
    return confidences, corrects


# ── calibration fitting ───────────────────────────────────────────────────────

def _fit_and_compare(confidences, corrects, output_dir: Path) -> tuple[str, str]:
    """Fit Platt and isotonic calibrators, compare ECE, save the better one.

    Returns (best_method, best_path).
    """
    from radiology_vqa.calibration.platt import PlattScaler
    from radiology_vqa.calibration.isotonic import IsotonicCalibrator
    from radiology_vqa.evaluation.calibration import expected_calibration_error

    output_dir.mkdir(parents=True, exist_ok=True)

    ece_raw = expected_calibration_error(confidences, corrects)
    logger.info("\n=== Calibration Results ===")
    logger.info("ECE (raw, no calibration): %.4f", ece_raw)

    # ── Platt scaling ──────────────────────────────────────────────────────
    platt = PlattScaler()
    platt_result = platt.fit(confidences, corrects)
    logger.info(
        "Platt: a=%.4f b=%.4f  ECE %.4f → %.4f",
        platt_result["a"],
        platt_result["b"],
        platt_result["ece_before"],
        platt_result["ece_after"],
    )
    platt_path = str(output_dir / "platt_scaler.json")
    platt.save(platt_path)

    # ── Isotonic regression ────────────────────────────────────────────────
    iso = IsotonicCalibrator()
    iso_result = iso.fit(confidences, corrects)
    logger.info(
        "Isotonic: ECE %.4f → %.4f",
        iso_result["ece_before"],
        iso_result["ece_after"],
    )
    iso_path = str(output_dir / "isotonic_scaler.json")
    iso.save(iso_path)

    # ── Pick best ─────────────────────────────────────────────────────────
    if platt_result["ece_after"] <= iso_result["ece_after"]:
        best_method = "platt"
        best_path = platt_path
        logger.info(
            "Best calibrator: Platt (ECE %.4f vs isotonic %.4f)",
            platt_result["ece_after"],
            iso_result["ece_after"],
        )
    else:
        best_method = "isotonic"
        best_path = iso_path
        logger.info(
            "Best calibrator: Isotonic (ECE %.4f vs Platt %.4f)",
            iso_result["ece_after"],
            platt_result["ece_after"],
        )

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "ece_raw": ece_raw,
        "platt": platt_result,
        "isotonic": iso_result,
        "best_method": best_method,
        "best_path": best_path,
    }
    summary_path = output_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    return best_method, best_path


# ── threshold sweep ───────────────────────────────────────────────────────────

def _threshold_sweep(calibrated_confs, corrects):
    """Sweep high/low confidence thresholds for supervisor re-tuning.

    Prints a table of (high_t, low_t, abstain_rate, accuracy_when_answered).
    Pick thresholds that maximise accuracy_when_answered while keeping
    abstention between 10–15%.
    """
    logger.info("\n=== Threshold Sweep (calibrated confidence) ===")
    logger.info(
        "%-8s %-8s %-14s %-24s",
        "high_t", "low_t", "abstain_rate", "accuracy_when_answered",
    )
    n = len(calibrated_confs)

    for high_t in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for low_t in [0.30, 0.35, 0.40, 0.45, 0.50]:
            if low_t >= high_t:
                continue
            # Simulate: abstain if conf < low_t; answer if conf >= low_t.
            # (high_t is used by supervisor for re-query, but in pure VLM mode
            #  it directly maps to answer vs abstain.)
            answered_idx = [i for i, c in enumerate(calibrated_confs) if c >= low_t]
            abstained = n - len(answered_idx)
            abstain_rate = abstained / n if n > 0 else 0.0
            acc_when_answered = (
                sum(corrects[i] for i in answered_idx) / len(answered_idx)
                if answered_idx
                else 0.0
            )
            logger.info(
                "%-8.2f %-8.2f %-14.3f %.3f",
                high_t, low_t, abstain_rate, acc_when_answered,
            )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("Phase 6C — Confidence Calibration Fitting")
    logger.info("=" * 60)
    logger.info("Val dataset:  %s / %s", args.val_dataset, args.val_split)
    logger.info("Output dir:   %s", output_dir)

    # ── Load VLM (without calibration — we want raw scores) ───────────────
    from radiology_vqa.config import settings
    from radiology_vqa.vlm.llava import LLaVABackend

    adapter_path = os.environ.get("VLM_ADAPTER_PATH", settings.vlm_adapter_path) or None
    logger.info("VLM adapter:  %s", adapter_path or "(none — zero-shot)")

    vlm = LLaVABackend(
        model_id=settings.vlm_model_id,
        quantize=settings.vlm_quantize,
        device=settings.vlm_device,
        max_new_tokens=settings.vlm_max_new_tokens,
        concise_mode=settings.vlm_concise_mode,
        adapter_path=adapter_path,
        calibration_method="none",   # always collect raw scores during fitting
        calibration_model_path="",
    )

    # ── Validation inference ───────────────────────────────────────────────
    val_samples = _load_dataset(args.val_dataset, args.val_split)
    val_confs, val_corrects = _run_inference(
        vlm, val_samples, max_samples=args.max_val_samples, desc="validation inference"
    )

    if not val_confs:
        logger.error("No valid predictions collected from validation set. Exiting.")
        sys.exit(1)

    # ── Fit calibrators ────────────────────────────────────────────────────
    best_method, best_path = _fit_and_compare(val_confs, val_corrects, output_dir)

    # ── Threshold sweep on calibrated val scores ───────────────────────────
    if best_method == "platt":
        from radiology_vqa.calibration.platt import PlattScaler
        calibrator = PlattScaler.load(best_path)
    else:
        from radiology_vqa.calibration.isotonic import IsotonicCalibrator
        calibrator = IsotonicCalibrator.load(best_path)

    calibrated_val_confs = [calibrator.calibrate(c) for c in val_confs]
    _threshold_sweep(calibrated_val_confs, val_corrects)

    # ── Optional: evaluate transfer on test set ────────────────────────────
    if args.test_dataset:
        logger.info("\n=== Calibration Transfer Evaluation (%s/%s) ===", args.test_dataset, args.test_split)
        test_samples = _load_dataset(args.test_dataset, args.test_split)
        test_confs, test_corrects = _run_inference(
            vlm, test_samples, max_samples=args.max_test_samples, desc="test inference"
        )

        if test_confs:
            from radiology_vqa.evaluation.calibration import expected_calibration_error
            ece_raw_test = expected_calibration_error(test_confs, test_corrects)
            calibrated_test = [calibrator.calibrate(c) for c in test_confs]
            ece_cal_test = expected_calibration_error(calibrated_test, test_corrects)
            logger.info(
                "Test ECE: raw=%.4f → calibrated=%.4f (improvement: %.4f)",
                ece_raw_test,
                ece_cal_test,
                ece_raw_test - ece_cal_test,
            )

    logger.info("\n=== Done ===")
    logger.info("Best calibrator: %s → %s", best_method, best_path)
    logger.info(
        "Next steps:\n"
        "  1. Add to configs/phase6_calibrated.yaml:\n"
        "       calibration_method: %r\n"
        "       calibration_model_path: %r\n"
        "  2. Run threshold sweep results above to tune supervisor thresholds.\n"
        "  3. Re-run evaluation with Config C (FT Agent + calibration).",
        best_method,
        best_path,
    )


if __name__ == "__main__":
    main()
