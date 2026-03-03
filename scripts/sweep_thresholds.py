#!/usr/bin/env python3
"""Phase 7A-3 — Per-question-type threshold sweep (CPU-only, no GPU required).

Reads a fit-calibration results JSON produced by fit_calibration.py
(Layout A: flat list of {confidence, correct, answer_type} records),
sweeps a (high_t, low_t) grid per subset (closed / open / combined),
and writes the best thresholds to configs/phase7a.yaml.

Layout A  — flat list of prediction records::

    [{"confidence": 0.82, "correct": true, "answer_type": "closed"}, ...]

Layout B  — pre-swept dict with a ``"combined"`` key (already processed by
fit_calibration.py --per-type-sweep).  In this case the script returns a
sentinel ``([], [], [])`` from ``_load_results`` so the caller can detect that
re-sweeping is not possible and fall back to the Layout B rows directly.

Usage
-----
    python scripts/sweep_thresholds.py \\
        --results data/calibration/mixed/threshold_sweep.json \\
        [--calibrator data/calibration/mixed/iso.json] \\
        [--calibrator-method isotonic] \\
        [--target-min-abstain 0.10] \\
        [--target-max-abstain 0.25] \\
        [--output-config configs/phase7a.yaml] \\
        [--output-sweep results/sweep.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical Phase 7A threshold grid — must match fit_calibration._threshold_sweep().
# HIGH x LOW = 7 x 9 = 63 combinations; 42 valid (low_t < high_t).
# Covers isotonic-calibrated score range [0.10, 0.80].
HIGH_THRESHOLDS = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
LOW_THRESHOLDS  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 7A-3: per-question-type threshold sweep (CPU-only).",
    )
    p.add_argument(
        "--results",
        required=True,
        metavar="PATH",
        help="Path to fit-calibration results JSON (Layout A or B).",
    )
    p.add_argument(
        "--calibrator",
        default=None,
        metavar="PATH",
        help="Optional path to a saved calibrator JSON.",
    )
    p.add_argument(
        "--calibrator-method",
        default="isotonic",
        dest="calibrator_method",
        choices=["isotonic", "platt"],
        help="Calibration method stored in --calibrator (default: isotonic).",
    )
    p.add_argument(
        "--target-min-abstain",
        type=float,
        default=0.10,
        dest="target_min_abstain",
        help="Minimum abstain rate in the target window (default: 0.10).",
    )
    p.add_argument(
        "--target-max-abstain",
        type=float,
        default=0.25,
        dest="target_max_abstain",
        help="Maximum abstain rate in the target window (default: 0.25).",
    )
    p.add_argument(
        "--output-config",
        default="configs/phase7a.yaml",
        dest="output_config",
        metavar="PATH",
        help="Where to write the phase7a YAML config (default: configs/phase7a.yaml).",
    )
    p.add_argument(
        "--output-sweep",
        default=None,
        dest="output_sweep",
        metavar="PATH",
        help="Optional path to write the sweep rows as JSON.",
    )
    return p.parse_args()


# ── private helpers ───────────────────────────────────────────────────────────


def _load_results(
    results_path: Path,
) -> tuple[list[float], list[bool], list[str]]:
    """Load a fit-calibration results JSON.

    Supports two layouts:

    * **Layout A** (flat list) — returns ``(confs, corrects, answer_types)``.
    * **Layout B** (dict with ``"combined"`` key) — returns the sentinel
      ``([], [], [])`` so the caller knows re-sweeping is not possible.

    Raises:
        FileNotFoundError: if *results_path* does not exist.
        ValueError: if a Layout A record is missing required keys.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as fh:
        data = json.load(fh)

    # Layout B detection -------------------------------------------------------
    if isinstance(data, dict) and "combined" in data:
        return [], [], []

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list or a dict with 'combined', "
            f"got {type(data).__name__}"
        )

    confs: list[float] = []
    corrects: list[bool] = []
    answer_types: list[str] = []

    for i, rec in enumerate(data):
        if "confidence" not in rec or "correct" not in rec:
            raise ValueError(
                f"Record {i} missing required keys "
                f"('confidence', 'correct'): {rec}"
            )
        confs.append(float(rec["confidence"]))
        corrects.append(bool(rec["correct"]))
        raw_type = rec.get("answer_type")
        answer_types.append(str(raw_type) if raw_type is not None else "open")

    return confs, corrects, answer_types


def _apply_calibrator(
    confs: list[float],
    calibrator_path: str | None,
    method: str,
) -> list[float]:
    """Apply a saved Platt / isotonic calibrator to raw confidence scores.

    If *calibrator_path* is ``None`` or empty the scores are returned unchanged.
    """
    if not calibrator_path:
        return confs

    from radiology_vqa.calibration import apply_calibrator, load_calibrator  # type: ignore

    calibrator = load_calibrator(Path(calibrator_path), method=method)
    return apply_calibrator(calibrator, confs)


def _run_sweep_grid(
    confs: list[float],
    corrects: list[bool],
) -> list[dict[str, Any]]:
    """Sweep a ``(high_t, low_t)`` grid and return one row per valid pair.

    A sample is *answered* when ``conf >= low_t``; otherwise it *abstains*.
    ``high_t`` is recorded for reference but not used as a hard filter —
    only ``low_t`` drives the abstention decision.

    Returns:
        List of dicts with keys:
        ``high_t``, ``low_t``, ``abstain_rate``,
        ``accuracy_when_answered``, ``overall_accuracy``, ``n_samples``.
        Returns an empty list when the input is empty.
    """
    n = len(confs)
    if n == 0:
        return []

    rows: list[dict[str, Any]] = []
    for high_t in HIGH_THRESHOLDS:
        for low_t in LOW_THRESHOLDS:
            if low_t >= high_t:
                continue

            answered_correct = 0
            answered_total = 0
            for c, ok in zip(confs, corrects):
                if c >= low_t:
                    answered_total += 1
                    if ok:
                        answered_correct += 1

            abstained = n - answered_total
            abstain_rate = abstained / n
            accuracy_when_answered = (
                answered_correct / answered_total if answered_total else 0.0
            )
            overall_accuracy = answered_correct / n

            rows.append(
                {
                    "high_t": high_t,
                    "low_t": low_t,
                    "abstain_rate": abstain_rate,
                    "accuracy_when_answered": accuracy_when_answered,
                    "overall_accuracy": overall_accuracy,
                    "n_samples": n,
                }
            )
    return rows


def _find_best_row(
    rows: list[dict[str, Any]],
    target_min: float,
    target_max: float,
) -> dict[str, Any]:
    """Pick the row that maximises ``accuracy_when_answered`` in the target range.

    Strategy:
    1. Filter rows whose ``abstain_rate`` falls within ``[target_min, target_max]``.
    2. Among those, maximise ``accuracy_when_answered``; tie-break on lower
       ``abstain_rate``.
    3. If no row falls in the target range, fall back to the full ``rows`` list.
    4. Returns an empty dict when ``rows`` is empty.
    """
    if not rows:
        return {}

    target_rows = [
        r for r in rows if target_min <= r["abstain_rate"] <= target_max
    ]
    pool = target_rows if target_rows else rows
    return max(
        pool,
        key=lambda r: (r["accuracy_when_answered"], -r["abstain_rate"]),
    )


def _write_phase7a_yaml(
    output_path: Path,
    best_closed: dict[str, Any],
    best_open: dict[str, Any],
    best_combined: dict[str, Any],
    calibration_method: str,
    calibrator_path: str,
) -> None:
    """Write the phase7a.yaml config with calibrated per-type thresholds.

    Falls back to the combined thresholds when a per-type row is missing.
    """
    # Fall back to combined when per-type dicts are empty.
    closed = best_closed if best_closed else best_combined
    opened = best_open if best_open else best_combined

    closed_high = closed.get("high_t", 0.65)
    closed_low  = closed.get("low_t",  0.20)
    open_high   = opened.get("high_t", 0.85)
    open_low    = opened.get("low_t",  0.55)

    content = f"""\
# Phase 7A configuration — auto-generated by scripts/sweep_thresholds.py.
# Do not edit manually. Re-run sweep_thresholds.py to refresh thresholds.
#
# To reproduce:
#   python scripts/sweep_thresholds.py \\
#       --results data/calibration/mixed/threshold_sweep.json \\
#       --output-config configs/phase7a.yaml
#
# To evaluate:
#   export VLM_ADAPTER_PATH=checkpoints/llava-med-qlora/best
#   export CALIBRATION_METHOD={calibration_method}
#   export CALIBRATION_MODEL_PATH={calibrator_path}
#   export SUPERVISOR_CLOSED_LOW_CONFIDENCE={closed_low:.2f}
#   export SUPERVISOR_OPEN_LOW_CONFIDENCE={open_low:.2f}
#   python scripts/run_evaluation.py \\
#       --mode agent --dataset vqa_rad --split test \\
#       --no-bertscore --config configs/phase7a.yaml

vlm:
  backend: "llava"
  model_id: "llava-hf/llava-v1.6-mistral-7b-hf"
  quantize: "4bit"
  max_new_tokens: 128
  device: "auto"
  concise_mode: true
  adapter_path: "checkpoints/llava-med-qlora/best"

data:
  index_dir: "data/indices_v2"

rag:
  retrieval_method: "hybrid"
  bm25_index_dir: "data/bm25_index"
  bm25_top_k: 20
  dense_top_k: 20
  rrf_k: 60
  retrieval_top_k: 5
  retrieval_min_score: 0.0

# Phase 7A-2: isotonic calibrator re-fitted on mixed validation set.
calibration_method: "{calibration_method}"
calibration_model_path: "{calibrator_path}"

agent:
  # Phase 7A-3: per-question-type thresholds from threshold sweep on mixed val.
  # Sweep target: 10-25% abstention, maximise accuracy_when_answered.
  supervisor_closed_high_confidence: {closed_high:.2f}
  supervisor_closed_low_confidence: {closed_low:.2f}
  supervisor_open_high_confidence: {open_high:.2f}
  supervisor_open_low_confidence: {open_low:.2f}
  supervisor_per_type_thresholds: true
  supervisor_max_retries: 1
  agreement_method: "embedding"
  supervisor_semantic_threshold: 0.87
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    logger.info("Wrote phase7a config → %s", output_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901
    # ── Ensure src/ is importable for standalone script execution ─────────────
    _src_path = Path(__file__).parent.parent / "src"
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    results_path = Path(args.results)
    confs, corrects, answer_types = _load_results(results_path)
    is_layout_b = len(confs) == 0

    if is_layout_b:
        logger.warning(
            "Layout B detected — re-sweeping is not possible. "
            "Using best row from pre-swept data."
        )
        with open(results_path) as fh:
            layout_b = json.load(fh)

        def _best_from_rows(rows: list) -> dict:
            if not rows:
                return {}
            return max(rows, key=lambda r: r.get("accuracy_when_answered", 0.0))

        best_closed   = _best_from_rows(layout_b.get("closed",   []))
        best_open     = _best_from_rows(layout_b.get("open",     []))
        best_combined = _best_from_rows(layout_b.get("combined", []))
    else:
        if args.calibrator:
            confs = _apply_calibrator(confs, args.calibrator, args.calibrator_method)

        closed_confs    = [c  for c,  t in zip(confs,    answer_types) if t == "closed"]
        closed_corrects = [ok for ok, t in zip(corrects, answer_types) if t == "closed"]
        open_confs      = [c  for c,  t in zip(confs,    answer_types) if t == "open"]
        open_corrects   = [ok for ok, t in zip(corrects, answer_types) if t == "open"]

        rows_closed   = _run_sweep_grid(closed_confs,   closed_corrects)
        rows_open     = _run_sweep_grid(open_confs,     open_corrects)
        rows_combined = _run_sweep_grid(confs,          corrects)

        best_closed   = _find_best_row(rows_closed,   args.target_min_abstain, args.target_max_abstain)
        best_open     = _find_best_row(rows_open,     args.target_min_abstain, args.target_max_abstain)
        best_combined = _find_best_row(rows_combined, args.target_min_abstain, args.target_max_abstain)

        if args.output_sweep:
            sweep_path = Path(args.output_sweep)
            sweep_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sweep_path, "w") as fh:
                json.dump(
                    {"closed": rows_closed, "open": rows_open, "combined": rows_combined},
                    fh,
                    indent=2,
                )
            logger.info("Wrote sweep rows → %s", sweep_path)

    _write_phase7a_yaml(
        Path(args.output_config),
        best_closed,
        best_open,
        best_combined,
        args.calibrator_method,
        args.calibrator or "",
    )


if __name__ == "__main__":
    main()
