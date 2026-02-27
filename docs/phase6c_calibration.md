# Phase 6C — Confidence Calibration

## Goal

The fine-tuned VLM (Phase 6A/6B) had a significant overconfidence problem: it was assigning raw confidence scores of 0.97–0.99 even on wrong answers. This made the supervisor's threshold logic meaningless — almost every prediction cleared the `HIGH_CONF=0.85` bar regardless of correctness. Phase 6C applied post-hoc calibration to map those raw scores into well-calibrated probabilities, restoring the clinical safety story: the system should know when it doesn't know.

**Target:** ECE < 0.10, AUROC improvement over uncalibrated baseline.

---

## What We Built

### Calibration Library (`src/radiology_vqa/calibration/`)

Two calibrators were implemented:

**Platt Scaling (`platt.py`)**
- Parametric: `calibrated = sigmoid(a × raw + b)`
- Fit via L-BFGS-B minimising negative log-likelihood on held-out val data
- Two learned parameters (a, b), serialised to JSON for auditability
- Output clamped to [0.01, 0.99]

**Isotonic Regression (`isotonic.py`)**
- Non-parametric: learns a monotone step function over the confidence range
- Fits via `sklearn.IsotonicRegression`, stores knot points as JSON
- Reconstructed at inference time using `numpy.interp` (no sklearn dependency at runtime)
- Output clamped to [0.01, 0.99]

### Fitting Script (`scripts/fit_calibration.py`)

The script follows a strict held-out protocol:

1. Load the fine-tuned LLaVA adapter (no calibration active — collect raw scores)
2. Run inference on SLAKE validation set (1,053 samples, held-out during QLoRA training)
3. Collect `(raw_confidence, correct)` pairs
4. Fit both Platt and isotonic; compare ECE; save the better one to `data/calibration/`
5. Run a threshold sweep on calibrated val scores to recommend new `HIGH_CONF`/`LOW_CONF` values
6. Optionally evaluate calibration transfer on VQA-RAD test set

### VLM Integration (`src/radiology_vqa/vlm/llava.py`)

`LLaVABackend.__init__` accepts `calibration_method` and `calibration_model_path`. When a calibrator is loaded, `predict()` applies it and returns both `confidence` (calibrated) and `raw_confidence` in `VLMPrediction`. Graceful degradation: missing file logs a warning and falls back to raw confidence.

### Config (`configs/phase6_calibrated.yaml`)

Extends `phase6.yaml` with:
```yaml
calibration_method: "isotonic"
calibration_model_path: "data/calibration/isotonic_scaler.json"
agent:
  supervisor_high_confidence: 0.60
  supervisor_low_confidence: 0.35
```

---

## Problems We Hit

### Platt Scaling Exploded

Platt scaling failed badly. The fine-tuned model's raw confidence distribution was very narrow — almost everything clustered around 0.93. Platt tried to compensate by learning an extreme slope: `a = 17.63`. This is a known failure mode: Platt requires a spread-out distribution to fit a meaningful sigmoid. The resulting ECE after Platt was 0.1106, barely better than raw (0.1331).

**Decision:** Reject Platt, use isotonic regression instead.

### Isotonic Regression Worked

With only 12 knot points learned on 200 SLAKE validation samples (subset used for speed), isotonic regression brought ECE from 0.1331 → 0.0047 on the validation set. Because it's non-parametric, it handled the narrow raw distribution by learning a sharp step function: scores near 0.93 get mapped to their empirical accuracy, not squeezed through a sigmoid.

### Threshold Re-tuning Required

After isotonic calibration, the confidence distribution became bimodal — scores were no longer clustered at 0.93, they spread toward 0.0 and 1.0. The original Phase 5 supervisor thresholds (`HIGH_CONF=0.85`, `LOW_CONF=0.55`) were calibrated for raw scores and were now wrong. We ran a sweep:

| high_t | low_t | abstain_rate | accuracy_when_answered |
|--------|-------|-------------|------------------------|
| 0.60   | 0.30  | 1.0%        | 80.8%                  |
| 0.60   | 0.35  | 15.5%       | 88.8%  ← sweet spot    |
| 0.60   | 0.40  | 28.3%       | 91.2%                  |
| 0.65   | 0.35  | 15.5%       | 88.8%                  |

**Decision:** Set `HIGH_CONF=0.60`, `LOW_CONF=0.35`. This gives a 15.5% abstention rate on the validation set with 88.8% accuracy when answered — a good clinical safety balance.

### Calibration Transfer Risk

Calibrators fit on SLAKE validation were applied to VQA-RAD test — a different dataset with different question distribution and difficulty. There was a risk the calibration would not transfer. In practice, ECE transferred well (confirmed in the evaluation run), though the abstention rate jumped from the expected 15.5% to 41.0% on VQA-RAD. VQA-RAD open questions are harder, so more predictions fell below the low threshold.

---

## Final Results

Evaluated on VQA-RAD test set (451 samples):

| Metric | Zero-Shot VLM | FT VLM | FT Agent (pre-cal) | FT Agent (calibrated) |
|--------|--------------|--------|--------------------|-----------------------|
| Overall accuracy | 41.5% | 50.8% | 42.1% | 35.5% |
| Closed accuracy | 61.4% | 71.7% | 58.6% | 50.6% |
| Accuracy when answered | 41.5% | 50.8% | 52.3% | **60.2%** |
| Abstention rate | 0% | 0% | 19.5% | 41.0% |
| Correct abstention rate | — | — | 55.7% | **62.7%** |
| ECE (↓) | 0.434 | 0.351 | 0.214 | **0.075** |
| AUROC | 0.769 | 0.751 | 0.761 | **0.868** |

Mean confidence on correct predictions: **0.739**
Mean confidence on wrong predictions: **0.240**

That 0.499 gap is what makes AUROC 0.868 — the calibrated confidence is a strong discriminator of correctness.

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Use SLAKE validation (not VQA-RAD) for calibration fitting | SLAKE val was held-out during QLoRA training; VQA-RAD test must stay unseen |
| Reject Platt, use isotonic | Platt learned `a=17.63` (extreme) due to narrow raw distribution. Isotonic is flexible enough to handle it |
| Re-tune thresholds after calibration | Post-isotonic scores are bimodal, not clustered at 0.93. Old thresholds were meaningless |
| Accept 41% abstention on VQA-RAD | The tradeoff is honest: the system answers fewer questions but the answers it gives are right 60.2% of the time, above the 50.8% FT VLM ceiling |
| Keep `raw_confidence` field in VLMPrediction | Allows the fitting script to always collect uncalibrated scores during the fitting loop, even if calibration is already active in the backend |

---

## Interpretation

The calibration achieved its goal. ECE dropped from 0.351 (fine-tuned, uncalibrated) to 0.075 — well under the 0.10 target. AUROC rose to 0.868, meaning the system's confidence scores now reliably predict whether the answer is correct.

The tradeoff is expected: calibration made the system more cautious. Overall accuracy fell to 35.5% because 41% of questions are now abstained on. But **when the system answers, it is right 60.2% of the time** — better than every other config. And 62.7% of abstentions were correct (the system abstained on questions it would have gotten wrong). That is the selective prediction property the Phase 4 design was built around.

The 41% abstention rate is aggressive. For Phase 6D ablation, trying `LOW_CONF=0.30` would bring abstention down to ~1% on validation, giving a point on the precision-recall curve closer to the pre-calibration agent.

---

## Artifacts

| File | Description |
|------|-------------|
| `scripts/fit_calibration.py` | Calibration fitting and threshold sweep script |
| `src/radiology_vqa/calibration/platt.py` | Platt scaling implementation |
| `src/radiology_vqa/calibration/isotonic.py` | Isotonic regression implementation |
| `src/radiology_vqa/evaluation/calibration.py` | ECE, AUROC, bin analysis metrics |
| `configs/phase6_calibrated.yaml` | Config with isotonic calibration + re-tuned thresholds |
| `data/calibration/isotonic_scaler.json` | Fitted calibrator (12 knots, SageMaker) |
| `data/calibration/calibration_summary.json` | Full comparison of both methods |
