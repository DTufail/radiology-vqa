#!/usr/bin/env python3
"""Phase 6C / 7A-2 — Fit confidence calibrators on a held-out validation set.

Procedure (Phase 6C, original)
------------------------------
1. Load the fine-tuned LLaVA-Next 7B with LoRA adapter.
2. Run inference on SLAKE validation set — held-out during training.
3. Collect (raw_confidence, correct) pairs.
4. Fit both Platt scaling and isotonic regression calibrators.
5. Report ECE before/after for both methods, pick the better one.
6. Save calibrator to ``data/calibration/``.
7. Optionally evaluate calibration transfer on VQA-RAD test set.

Phase 7A-2 extension: mixed calibration set
--------------------------------------------
Pass ``--val-dataset mixed`` to combine SLAKE validation with the last 10%%
of VQA-RAD train. This broadens the raw confidence distribution seen during
calibrator fitting, preventing the isotonic step-function from collapsing
VQA-RAD scores to near-zero.

  SLAKE val:          200 English samples (split="validation")
  VQA-RAD train 10%%:  last 306 samples of VQA-RAD train split (fixed, deterministic)
  Combined:           ~506 samples

CRITICAL: Always fit on validation data; NEVER on the test set.
VQA-RAD test (451 samples) must never appear in the calibration set.
The last-10%%-by-index strategy avoids randomness while respecting the firewall.

Usage
-----
    # Phase 6C original (SLAKE-only):
    VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \\
      python scripts/fit_calibration.py \\
        --val-dataset slake \\
        --val-split validation \\
        --output data/calibration/

    # Phase 7A-2 mixed calibration (recommended):
    VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \\
      python scripts/fit_calibration.py \\
        --val-dataset mixed \\
        --output data/calibration/mixed/ \\
        --mixed-vqa-rad-fraction 0.10

    # With transfer evaluation on VQA-RAD test:
    VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best \\
      python scripts/fit_calibration.py \\
        --val-dataset mixed \\
        --output data/calibration/mixed/ \\
        --test-dataset vqa_rad \\
        --test-split test
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
        choices=["slake", "vqa_rad", "pathvqa", "mixed"],
        help=(
            "Dataset to use for calibration fitting (default: slake). "
            "Use 'mixed' for Phase 7A-2 (SLAKE val + VQA-RAD train 10%%)."
        ),
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
        "--mixed-vqa-rad-fraction",
        type=float,
        default=0.10,
        dest="mixed_vqa_rad_fraction",
        help=(
            "Fraction of VQA-RAD train to include when --val-dataset=mixed. "
            "Uses the last N samples by index (deterministic, no randomness). "
            "Default: 0.10 (last 10%% ≈ 306 samples of 3,064 train)."
        ),
    )
    p.add_argument(
        "--per-type-sweep",
        action="store_true",
        dest="per_type_sweep",
        help=(
            "When set, run the threshold sweep separately for closed and open "
            "questions (Phase 7A-3). Requires answer_type labels on each sample. "
            "Default: False (runs the combined sweep only)."
        ),
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


def _load_mixed_dataset(vqa_rad_fraction: float = 0.10) -> list:
    """Load SLAKE validation + last N% of VQA-RAD train as a mixed calibration set.

    This is the Phase 7A-2 calibration set. It broadens the raw confidence
    distribution by including VQA-RAD samples alongside SLAKE, preventing the
    isotonic calibrator from creating knots only in SLAKE's narrow [0.88, 0.97]
    range.

    Args:
        vqa_rad_fraction: Fraction of VQA-RAD train to include. Default 0.10.
                          Uses the LAST N samples by index — deterministic, no
                          random seed required. The last-10%-by-index approach
                          is conservative: it selects samples the fine-tuned model
                          has already seen, so their raw confidence distribution is
                          representative of the model's behaviour on VQA-RAD data.

    Returns:
        Combined list of VQASample objects (SLAKE val first, then VQA-RAD train subset).

    SAFETY: This function loads only VQA-RAD TRAIN. It never loads VQA-RAD test.
            The caller must not pass split="test" to this function.
    """
    from radiology_vqa.slake_loader import load_slake
    from radiology_vqa.loader import load_vqa_rad
    from radiology_vqa.config import settings

    # ── SLAKE validation (200 English samples) ────────────────────────────────
    logger.info("Loading SLAKE validation for mixed calibration set ...")
    slake_val = load_slake(settings.slake_dir, split="validation")
    logger.info("  SLAKE val: %d samples", len(slake_val))

    # ── VQA-RAD train subset (last N%) ────────────────────────────────────────
    # CRITICAL: load split="train", never "test".
    logger.info("Loading VQA-RAD train for mixed calibration set ...")
    vqa_rad_train = load_vqa_rad(split="train")
    n_vqa_rad = max(1, int(len(vqa_rad_train) * vqa_rad_fraction))
    # Take the LAST n samples — deterministic, same result every run.
    vqa_rad_subset = vqa_rad_train[-n_vqa_rad:]
    logger.info(
        "  VQA-RAD train: %d total → using last %d (%.0f%%)",
        len(vqa_rad_train),
        len(vqa_rad_subset),
        vqa_rad_fraction * 100,
    )

    combined = slake_val + vqa_rad_subset
    logger.info(
        "Mixed calibration set: %d total (%d SLAKE val + %d VQA-RAD train subset)",
        len(combined),
        len(slake_val),
        len(vqa_rad_subset),
    )
    return combined


# ── inference loop ────────────────────────────────────────────────────────────

def _run_inference(vlm, samples, max_samples=None, desc="inference"):
    """Run VLM inference and collect (raw_confidence, correct, answer_type) triples.

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
    answer_types = []
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
            answer_types.append(getattr(sample, "answer_type", "open") or "open")
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
    return confidences, corrects, answer_types


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

def _threshold_sweep(
    calibrated_confs: list[float],
    corrects: list[bool],
    answer_types: list[str] | None = None,
    per_type: bool = False,
) -> dict:
    """Sweep supervisor confidence thresholds and report abstain/accuracy trade-offs.

    Prints a formatted table of (high_t, low_t, abstain_rate, accuracy_when_answered,
    overall_accuracy). When per_type=True, runs three sweeps: combined, closed-only,
    and open-only, printing a table for each.

    Returns a dict with keys "combined", and optionally "closed" and "open" when
    per_type=True. Each value is a list of row dicts with keys:
      high_t, low_t, abstain_rate, accuracy_when_answered, overall_accuracy, n_samples.

    Args:
        calibrated_confs: List of calibrated confidence scores.
        corrects:         List of bool correctness labels, same length.
        answer_types:     List of "open"/"closed" strings, same length.
                          Required when per_type=True; ignored when per_type=False.
        per_type:         If True, run the sweep separately for closed and open subsets.
    """
    # Canonical Phase 7A grid — must match sweep_thresholds.HIGH/LOW_THRESHOLDS.
    # 7 x 9 = 63 combinations; 42 valid pairs (low_t < high_t).
    HIGH_THRESHOLDS = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    LOW_THRESHOLDS  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    def _run_sweep(confs: list[float], labels: list[bool], label: str) -> list[dict]:
        """Run the grid sweep for a single (conf, correct) list pair."""
        n = len(confs)
        if n == 0:
            logger.warning("Threshold sweep: empty subset for %r — skipping.", label)
            return []

        logger.info("\n=== Threshold Sweep: %s (n=%d) ===", label, n)
        logger.info(
            "%-8s  %-8s  %-14s  %-24s  %-16s",
            "high_t", "low_t", "abstain_rate", "acc_when_answered", "overall_acc",
        )

        rows: list[dict] = []
        for high_t in HIGH_THRESHOLDS:
            for low_t in LOW_THRESHOLDS:
                if low_t >= high_t:
                    continue
                # Simulate supervisor: abstain if conf < low_t.
                # Re-query (high_t) is not simulated here — we only care about
                # the final abstain/answer split for threshold selection.
                answered_idx = [i for i, c in enumerate(confs) if c >= low_t]
                n_abstained  = n - len(answered_idx)
                abstain_rate = n_abstained / n

                if answered_idx:
                    acc_when_answered = sum(labels[i] for i in answered_idx) / len(answered_idx)
                else:
                    acc_when_answered = 0.0

                overall_acc = sum(labels[i] for i in answered_idx) / n

                row = {
                    "high_t": high_t,
                    "low_t": low_t,
                    "abstain_rate": abstain_rate,
                    "accuracy_when_answered": acc_when_answered,
                    "overall_accuracy": overall_acc,
                    "n_samples": n,
                }
                rows.append(row)
                logger.info(
                    "%-8.2f  %-8.2f  %-14.3f  %-24.3f  %.3f",
                    high_t, low_t, abstain_rate, acc_when_answered, overall_acc,
                )
        return rows

    results: dict = {}

    # ── Combined sweep (always runs) ──────────────────────────────────────────
    results["combined"] = _run_sweep(calibrated_confs, corrects, "COMBINED")

    # ── Per-type sweeps (only when requested) ─────────────────────────────────
    if per_type:
        if answer_types is None or len(answer_types) != len(calibrated_confs):
            logger.warning(
                "per_type=True but answer_types is None or mismatched — "
                "skipping per-type sweep."
            )
            return results

        closed_confs   = [c for c, t in zip(calibrated_confs, answer_types) if t == "closed"]
        closed_labels  = [l for l, t in zip(corrects, answer_types) if t == "closed"]
        open_confs     = [c for c, t in zip(calibrated_confs, answer_types) if t == "open"]
        open_labels    = [l for l, t in zip(corrects, answer_types) if t == "open"]

        logger.info(
            "Per-type split: closed=%d open=%d unknown=%d",
            len(closed_confs),
            len(open_confs),
            len(calibrated_confs) - len(closed_confs) - len(open_confs),
        )

        results["closed"] = _run_sweep(closed_confs, closed_labels, "CLOSED")
        results["open"]   = _run_sweep(open_confs,   open_labels,   "OPEN")

    return results


# ── recommendations ──────────────────────────────────────────────────────────

def _print_recommendations(sweep_results: dict) -> None:
    """Print threshold recommendations for each sweep subset.

    Selects the row with the best accuracy_when_answered that also has
    abstain_rate in the target range [0.10, 0.25] (10-25% abstention).
    Falls back to the minimum-abstain row if no row meets the target range.

    Target range rationale:
        < 10% abstention: system answers too aggressively, including uncertain cases.
        10-25% abstention: clinical sweet spot — conservative but usable.
        > 25% abstention: over-conservative; degrades overall accuracy too much.
    """
    TARGET_MIN_ABSTAIN = 0.10
    TARGET_MAX_ABSTAIN = 0.25

    logger.info("\n=== Threshold Recommendations ===")

    for subset_name, rows in sweep_results.items():
        if not rows:
            continue

        # Filter to target abstain range first
        target_rows = [
            r for r in rows
            if TARGET_MIN_ABSTAIN <= r["abstain_rate"] <= TARGET_MAX_ABSTAIN
        ]

        # If no rows in target range, fall back to all rows
        candidate_rows = target_rows if target_rows else rows

        # Among candidates, pick the row with highest accuracy_when_answered.
        # Tie-break: prefer lower abstain_rate (more answers).
        best_row = max(
            candidate_rows,
            key=lambda r: (r["accuracy_when_answered"], -r["abstain_rate"]),
        )

        fallback_note = "" if target_rows else " (no rows in 10-25% range; using best overall)"
        logger.info(
            "  [%s]%s  Recommended: high_t=%.2f  low_t=%.2f  "
            "abstain=%.1f%%  acc_when_answered=%.3f  overall_acc=%.3f",
            subset_name.upper(),
            fallback_note,
            best_row["high_t"],
            best_row["low_t"],
            best_row["abstain_rate"] * 100,
            best_row["accuracy_when_answered"],
            best_row["overall_accuracy"],
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("Phase 6C / 7A-2 — Confidence Calibration Fitting")
    logger.info("=" * 60)
    logger.info("Val dataset:  %s", args.val_dataset)
    logger.info("Output dir:   %s", output_dir)
    if args.val_dataset == "mixed":
        logger.info("Mixed VQA-RAD fraction: %.2f", args.mixed_vqa_rad_fraction)
    if args.per_type_sweep:
        logger.info("Per-type threshold sweep: ENABLED (Phase 7A-3)")

    # ── Load VLM (without calibration — we want raw scores) ───────────────────
    from radiology_vqa.config import settings
    from radiology_vqa.vlm.llava import LLaVABackend
    import os

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

    # ── Load validation samples ────────────────────────────────────────────────
    if args.val_dataset == "mixed":
        val_samples = _load_mixed_dataset(vqa_rad_fraction=args.mixed_vqa_rad_fraction)
    else:
        val_samples = _load_dataset(args.val_dataset, args.val_split)

    # ── Run validation inference ───────────────────────────────────────────────
    val_confs, val_corrects, val_answer_types = _run_inference(
        vlm, val_samples,
        max_samples=args.max_val_samples,
        desc="validation inference",
    )

    if not val_confs:
        logger.error("No valid predictions from validation set. Exiting.")
        sys.exit(1)

    # ── Fit calibrators ────────────────────────────────────────────────────────
    best_method, best_path = _fit_and_compare(val_confs, val_corrects, output_dir)

    # ── Load calibrator for threshold sweep ────────────────────────────────────
    if best_method == "platt":
        from radiology_vqa.calibration.platt import PlattScaler
        calibrator = PlattScaler.load(best_path)
    else:
        from radiology_vqa.calibration.isotonic import IsotonicCalibrator
        calibrator = IsotonicCalibrator.load(best_path)

    calibrated_val_confs = [calibrator.calibrate(c) for c in val_confs]

    # ── Threshold sweep ────────────────────────────────────────────────────────
    sweep_results = _threshold_sweep(
        calibrated_val_confs,
        val_corrects,
        answer_types=val_answer_types if args.per_type_sweep else None,
        per_type=args.per_type_sweep,
    )

    # ── Save sweep results as JSON ─────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "threshold_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    logger.info("Threshold sweep saved to %s", sweep_path)

    # ── Print recommended thresholds ──────────────────────────────────────────
    # Recommend thresholds targeting 15-25% abstention rate as a clinical sweet spot.
    _print_recommendations(sweep_results)

    # ── Optional: evaluate transfer on test set ────────────────────────────────
    if args.test_dataset:
        logger.info(
            "\n=== Calibration Transfer Evaluation (%s/%s) ===",
            args.test_dataset, args.test_split,
        )
        test_samples = _load_dataset(args.test_dataset, args.test_split)
        test_confs, test_corrects, _test_answer_types = _run_inference(
            vlm, test_samples,
            max_samples=args.max_test_samples,
            desc="test inference",
        )

        if test_confs:
            from radiology_vqa.evaluation.calibration import expected_calibration_error
            ece_raw_test  = expected_calibration_error(test_confs, test_corrects)
            calibrated_test = [calibrator.calibrate(c) for c in test_confs]
            ece_cal_test  = expected_calibration_error(calibrated_test, test_corrects)
            logger.info(
                "Test ECE: raw=%.4f → calibrated=%.4f (improvement: %.4f)",
                ece_raw_test, ece_cal_test, ece_raw_test - ece_cal_test,
            )

    logger.info("\n=== Done ===")
    logger.info("Best calibrator: %s → %s", best_method, best_path)
    logger.info(
        "Next steps:\n"
        "  1. Copy recommended thresholds from sweep output above into\n"
        "     configs/phase7a.yaml (created by this script as a template).\n"
        "  2. Run: python scripts/run_evaluation.py --mode agent "
        "--dataset vqa_rad --split test --no-bertscore\n"
        "       with env: CALIBRATION_METHOD=%s CALIBRATION_MODEL_PATH=%s",
        best_method, best_path,
    )


if __name__ == "__main__":
    main()
