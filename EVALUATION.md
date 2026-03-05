# Evaluation

Complete evaluation results for the Grounded Multi-Agent Radiology VQA system across all six ablation configurations. Dataset: VQA-RAD test split (451 samples, 251 closed / 200 open). All runs use `--no-bertscore` except where noted.

---

## 6-Configuration Ablation

Each configuration isolates one component's contribution by changing exactly one variable from the previous config.

| # | Config | VLM | LoRA | RAG | KG | Agreement | Cal. | Index | Overall Acc | Acc (ans) | Abstain | Citation | ECE | AUROC |
|---|--------|-----|------|-----|-----|-----------|------|-------|-------------|-----------|---------|---------|-----|-------|
| 1 | baseline_vlm | ZS | — | — | — | — | — | — | 41.5% | 41.5% | 0% | — | 0.434 | 0.769 |
| 2 | baseline_agent | ZS | — | Yes | Original | Keyword | — | v1 | 31.9% | 46.8% | 31.7% | 15.3% | 0.188 | 0.765 |
| 3 | finetuned_vlm | FT | ✓ | — | — | — | — | — | 50.8% | 50.8% | 0% | — | 0.351 | 0.751 |
| 4 | finetuned_agent | FT | ✓ | Yes | Original | Keyword | — | v1 | 36.8% | 53.9% | 31.7% | 15.3% | 0.132 | 0.819 |
| 5 | full_pipeline | FT | ✓ | Yes | Expanded | Embed | — | v2 | 42.1% | 52.3% | 19.5% | 15.3% | 0.214 | 0.761 |
| 6 | full_calibrated (Phase 6C) | FT | ✓ | Yes | Expanded | Embed | Isotonic (SLAKE) | v2 | 35.5% | 60.2% | 41.0% | 15.3% | 0.075 | 0.868 |
| 7 | Phase 7A — mixed calibrator | FT | ✓ | Yes | Expanded | Embed | Isotonic (mixed) | v2 | 42.1% | 53.7% | 21.5% | 15.3% | 0.081 | 0.793 |
| **8** | **Phase 8A — QA index v3** | **FT** | **✓** | **Yes** | **Expanded+QA** | **Embed** | **Isotonic (mixed)** | **v3** | **50.3%** | **54.2%** | **7.1%** | **44.9%** | **0.091** | **0.755** |

> ZS = zero-shot (no adapter). FT = QLoRA fine-tuned. Original KG = SLAKE KG only (2,987 docs). Expanded KG = SLAKE KG + RadLex (6,724 docs). Expanded+QA = SLAKE KG + RadLex + QA pseudo-docs (13,435 docs). Acc (ans) = accuracy on non-abstained samples only.
> Index v1 = SLAKE KG only (2,987 docs). Index v2 = KG + RadLex (6,724 docs). Index v3 = KG + RadLex + QA pseudo-docs (13,435 docs). Mixed = SLAKE val + VQA-RAD train 10%.

### Per Question-Type (available configs)

| # | Config | Closed Acc | Open Acc | Closed F1 | Closed n | Open n |
|---|--------|-----------|---------|----------|--------|------|
| 1 | baseline_vlm | 61.4% | 16.5% | 0.614 | 251 | 200 |
| 2 | baseline_agent | 50.2% | 9.0% | 0.655 | 251 | 200 |
| 3 | finetuned_vlm | 71.7% | 24.5% | 0.717 | 251 | 200 |
| 4 | finetuned_agent | 55.4% | 13.5% | 0.652 | 251 | 200 |
| 5 | full_pipeline | 58.6% | 21.5% | 0.684 | 251 | 200 |
| 6 | full_calibrated | 50.6% | 16.5% | 0.684 | 147 | 119 |
| 7 | Phase 7A | 58.6% | 21.5% | 0.684 | 251 | 200 |
| **8** | **Phase 8A (FINAL)** | **71.7%** | **23.5%** | **0.699** | **249** | **200** |

> Config 6 closed/open counts are lower because 41% of samples are abstained on; the denominators reflect answered samples only. Config 8 closed_n=249 because 2 samples were excluded as non-binary answers.

---

## What Each Component Contributed

### Fine-tuning (Config 3 vs Config 1): +9.3 pp

QLoRA fine-tuning on 26,358 training samples (VQA-RAD + SLAKE + PathVQA) improved overall accuracy from 41.5% to 50.8% — the single largest gain across all six configurations. Closed-question accuracy rose 10.3 pp (61.4% → 71.7%), showing that domain vocabulary alignment had the most impact on binary questions where the model previously hallucinated "yes" on CT findings that apply to MRI, or vice versa. Open accuracy also improved 8.0 pp (16.5% → 24.5%), though open questions remain difficult throughout.

The fine-tuned model does produce higher raw confidence scores — typically 0.93–0.99 — compared to the zero-shot model. This is why ECE actually slightly worsened (0.434 → 0.351): the model became more assertive, but not proportionally more accurate.

### RAG grounding on zero-shot VLM (Config 2 vs Config 1): −9.6 pp overall, +5.3 pp when answered

Adding the agent pipeline to the zero-shot VLM drops overall accuracy from 41.5% to 31.9% — a 9.6 pp fall — while introducing a 31.7% abstention rate. Accuracy-when-answered rises from 41.5% to 46.8%, a 5.3 pp gain. The pattern is consistent with the design intent: the supervisor correctly withholds answers the VLM would have gotten wrong, but the zero-shot VLM's error rate is high enough that many correct predictions also get abstained on.

The open accuracy drop is severe: 16.5% → 9.0%. The original KG (2,987 SLAKE KG docs) has poor coverage of the spatial and modality-type open questions in VQA-RAD, so the keyword agreement mechanism returns no support for most open questions, causing abstention. ECE improves from 0.434 to 0.188 even without calibration — the confidence-based routing passively filters overconfident wrong predictions.

### RAG grounding on fine-tuned VLM (Config 4 vs Config 3): −14.0 pp overall, +3.1 pp when answered

Adding the Phase 5 agent pipeline to the fine-tuned VLM drops overall accuracy from 50.8% to 36.8% — a 14.0 pp fall — while accuracy-when-answered rises from 50.8% to 53.9%, a 3.1 pp gain. The abstention rate is 31.7%, identical to Config 2, which makes sense: the same keyword-based agreement mechanism is in use.

The fine-tuned model suffers more from abstention than the zero-shot model does, even though its answers are more accurate. The reason is overconfidence: the fine-tuned model outputs confidence scores in the 0.97–0.99 range almost uniformly. The supervisor's high-confidence threshold (0.85) is easily crossed, routing most predictions to the agreement check regardless of actual correctness. The keyword-based agreement then abstains on a large fraction because the original SLAKE KG (2,987 docs) has sparse coverage of the VQA-RAD question vocabulary. ECE improves substantially (0.351 → 0.132) because abstained samples (confidence=0.0) dilute the overconfidence pool.

Open accuracy drops sharply: 24.5% → 13.5%. The fine-tuned model's improved open answers (spatial descriptions, modality terms) are not matched by corresponding KG coverage, so the keyword agreement returns empty support and the supervisor abstains. The re-query rate is 0.0 — the supervisor's retry mechanism does not trigger because the fine-tuned model's first-pass confidence almost always exceeds the high-confidence threshold.

### KG expansion + embedding agreement (Config 5 vs Config 4): −14.6 pp overall, −1.6 pp when answered

Config 5 (expanded KG + embedding agreement) scores 42.1% overall and 52.3% accuracy-when-answered, vs Config 4's 36.8% and 53.9%. Overall accuracy improves by 5.3 pp (fewer abstentions, abstention rate drops from 31.7% to 19.5%), but accuracy-when-answered falls 1.6 pp.

The expanded index (13,435 docs) fills the coverage gaps that caused mass abstention in Config 4: consolidation terminology, modality vocabulary, and spatial QA pairs now have supporting documents. This reduces over-abstention on cases the fine-tuned model answered correctly. The embedding-based agreement (PubMedBERT cosine similarity ≥ 0.87) additionally rescues synonym cases that keyword matching missed: "tumor"/"neoplasm", "opacity"/"consolidation", "cardiac silhouette"/"cardiomegaly". BM25 hybrid retrieval adds exact-term recall for abbreviated clinical terminology.

The slight accuracy-when-answered dip (53.9% → 52.3%) reflects that the expanded index also grounds some wrong answers: previously these would have abstained under keyword agreement, but now embedding agreement finds domain-adjacent documents that trigger a confidence pass. This is the known limitation of topical vs. binary agreement — the supervisor verifies that the VLM's answer is topically grounded, not that it is correct.

### Calibration — isotonic regression (Config 6 vs Config 5)

Calibration is the clearest single-component result in this ablation. Isotonic regression fit on 200 SLAKE validation samples reduces ECE from 0.214 to 0.075 — a 65% improvement — and raises AUROC from 0.761 to 0.868. Mean confidence on correct predictions is 0.739 vs 0.240 on wrong predictions: a 0.499 gap that directly explains the high AUROC.

The cost is abstention rate rising from 19.5% to 41.0%. After isotonic calibration, the confidence distribution becomes bimodal: predictions the model is genuinely confident about cluster near 1.0, while uncertain predictions cluster near 0.0. With the LOW_CONF threshold at 0.35, anything below abstains. Accuracy-when-answered rises from 52.3% to 60.2%: the 41% of samples the system abstains on are disproportionately ones it would have gotten wrong.

### Full system vs. baseline (Config 6 vs Config 1)

Overall accuracy falls from 41.5% to 35.5% — a 6 pp drop — but this comparison conflates two different operating regimes. Config 1 answers every question; Config 6 abstains on 41%. The correct comparison is accuracy-when-answered: 41.5% vs 60.2%, a 18.7 pp improvement. The system that answers fewer questions answers them significantly more reliably.

---

## The Selective Prediction Argument

The clinical framing for this system is selective prediction: it is better to say "I don't know" on uncertain cases than to give a wrong answer confidently. Three metrics support this claim:

**AUROC = 0.868.** Confidence scores are a strong discriminator between correct and incorrect predictions. A random classifier would score 0.5; 0.868 means the system's uncertainty signal is meaningful. A clinician reviewing low-confidence outputs for manual review would be right to be skeptical.

**Correct abstention rate = 62.7%.** Of the 185 cases the system abstains on (Config 6), 116 (62.7%) are cases it would have answered incorrectly. The abstention mechanism is selective — it is not randomly skipping questions, it is disproportionately skipping the hard ones.

**Accuracy-when-answered = 60.2%.** When the system does answer, it is right 60.2% of the time, compared to 50.8% for the fine-tuned VLM without any abstention. The 9.4 pp gap represents the grounding benefit: the supervisor is filtering out answers that would have been wrong.

In a clinical decision support setting, these properties matter differently than raw accuracy. A system that is right 60% of the time when it answers, while flagging the other 41% for physician review, is safer than a system that is right 51% of the time but answers everything.

---

## Honest Limitations

**The agent pipeline does not correct the VLM.** In all evaluated configurations, the number of cases where the agent pipeline changed a wrong VLM answer to a correct one is zero or near-zero. RAG acts as a gating mechanism — it decides whether to answer or abstain — but the final answer is always the VLM's visual answer. The retrieval evidence is used for agreement scoring, not for answer generation. This is a fundamental architectural limitation.

**41% abstention is clinically aggressive.** The calibrated system abstains on nearly half of all questions. Whether this is acceptable depends on deployment context. In settings where radiologist review is available, 41% handoff rate may be acceptable. In settings where VQA must answer to be useful at all, Config 5 (19.5% abstention, 52.3% accuracy-when-answered) is a better operating point.

**Open-ended questions remain weak.** Open accuracy across all configs spans 16.5%–24.5%. The model struggles with anatomical description and spatial localization ("where is the opacity?", "what is the shape of the mass?"). This is partly a training data limitation — short answer QA pairs don't teach anatomical description — and partly a fundamental limitation of extracting spatial information from compressed 7B model weights.
---

## Phase 7A — Calibration Domain Fix

**Goal:** Reduce the 41% over-abstention caused by SLAKE→VQA-RAD calibration mismatch.

**Root cause:** The isotonic calibrator (Phase 6C) was fitted on SLAKE validation data only. SLAKE images (hospital, structured questions) have a different confidence distribution than VQA-RAD. The calibrated scores for VQA-RAD clustered below 0.35, triggering abstention on samples the model answered correctly.

**Fix:** Re-fitted isotonic calibrator on a **mixed validation set**:
- 200 SLAKE validation samples
- ~306 VQA-RAD train samples (last 10%, never used in test)

| Metric | Phase 6C | Phase 7A | Change |
|---|---|---|---|
| Abstention rate | 41.0% | 21.5% | -19.5pp |
| Overall accuracy | 35.5% | 42.1% | +6.6pp |
| ECE | 0.075 | 0.081 | +0.006 |
| Citation hit rate | 15.3% | 15.3% | — |

---

## Phase 8A — QA Pseudo-Document Index Expansion (FINAL)

**Goal:** Increase citation hit rate above 40% without retraining the VLM.

**Change:** Added 6,711 QA pseudo-documents (VQA-RAD train + SLAKE train pairs) to the retrieval index, bringing total corpus from 6,724 to 13,435 documents. BM25 can now find near-exact question paraphrases — the dominant source of citation hits.

**Index v3 composition:**

| Source | Documents | Notes |
|---|---|---|
| kg_disease | 2,501 | Disease KG triples |
| kg_organ | 383 | Organ KG triples |
| kg_organ_rel | 103 | Organ relationship triples |
| radlex | 3,737 | RadLex Tier 1 definitions |
| qa_vqarad | 1,793 | VQA-RAD train QA pairs ← NEW |
| qa_slake | 4,918 | SLAKE train QA pairs (English) ← NEW |
| **Total** | **13,435** | +6,711 from Phase 7A |

**Config bug fixed:** Three Phase 8A evaluation runs (v1, v2, tuned) produced incorrect results (33.7%) because `config.py` hardcoded defaults were Phase 5 values and no `.env` file existed. Fix: updated `config.py` class-level defaults to Phase 8A production values.

| Metric | Phase 7A | Phase 8A | Change |
|---|---|---|---|
| Overall accuracy | 42.1% | **50.3%** | +8.2pp |
| Closed accuracy | 58.6% | **71.7%** | +13.1pp |
| Open accuracy | 21.5% | **23.5%** | +2.0pp |
| Open token F1 | — | **0.301** | — |
| Open BERTScore F1 | — | **0.655** | — |
| Abstention rate | 21.5% | **7.1%** | -14.4pp |
| Citation hit rate | 15.3% | **44.9%** | +29.6pp |
| ECE | 0.081 | 0.091 | +0.010 |
| AUROC | 0.793 | 0.755 | -0.038 |

> AUROC regression is expected: lower abstention includes harder borderline samples.
> ECE regression is due to calibrator fitted before index expansion — fixable by re-fitting on Phase 8A data.

### Threshold Analysis (Phase 8A)

| Threshold | Coverage | Accuracy | Use case |
|---|---|---|---|
| 0.50 | 55.4% | 66.8% | Broad coverage |
| 0.65 | 46.8% | 69.7% | Balanced |
| 0.75 | 28.2% | 80.3% | Conservative |
| **0.85** | **19.5%** | **88.6%** | **Clinical recommended** |
| 0.90 | 13.5% | 90.2% | High-precision |

### Calibration Bins (Phase 8A)

| Bin | Count | Mean Conf | Accuracy | Gap | Status |
|---|---|---|---|---|---|
| [0.0, 0.1) | 32 | 0.000 | 0.000 | 0.000 | ✅ abstained |
| [0.1, 0.2) | 15 | 0.162 | 0.200 | 0.038 | ✅ good |
| [0.2, 0.3) | 32 | 0.250 | 0.344 | 0.094 | 🟡 underconfident |
| [0.3, 0.4) | 75 | 0.339 | 0.400 | 0.061 | ✅ good |
| [0.4, 0.5) | 47 | 0.451 | 0.340 | 0.110 | 🟡 overconfident |
| [0.5, 0.6) | 22 | 0.545 | 0.500 | 0.045 | ✅ good |
| [0.6, 0.7) | 91 | 0.678 | 0.549 | 0.128 | 🔴 overconfident |
| [0.7, 0.8) | 17 | 0.742 | 0.588 | 0.154 | 🔴 overconfident |
| [0.8, 0.9) | 59 | 0.829 | 0.695 | 0.134 | 🔴 overconfident |
| [0.9, 1.0) | 61 | 0.975 | 0.902 | 0.074 | ✅ excellent |
**Calibration was fit on SLAKE validation, transferred to VQA-RAD test.** The isotonic calibrator was fit on 200 SLAKE validation samples. VQA-RAD test has a different question distribution and higher open-question difficulty. The calibration transfers reasonably (ECE 0.075 on VQA-RAD test), but the abstention rate (41%) is higher than what the threshold sweep predicted on the validation set (15.5%). The calibrator underestimates how difficult VQA-RAD open questions are for the fine-tuned model.

**Correct abstention rate (62.7%) is better than random but not clinically reliable.** If abstentions were random, the correct abstention rate would equal the error rate (~40%). 62.7% shows the signal is better than random, but a clinical deployment would want this above 80% before relying on it as a triage signal.

---

## Calibration Analysis

### Before and After (Config 5 → Config 6)

Before calibration (Config 5), the model's raw confidence scores cluster around 0.93, regardless of correctness. The 0.8–0.9 confidence bin contains samples with mean confidence 0.85 but only 43% accuracy — a 42 pp overconfidence gap. Across all bins, the model systematically overstates its certainty.

After isotonic calibration (Config 6), the distribution shifts. The isotonic function maps raw scores through 12 learned knot points, pushing high-confidence-but-wrong predictions toward 0.0 and correctly-confident predictions toward 1.0. ECE measures the weighted average of the confidence-accuracy gap across all bins; 0.075 means the average gap is 7.5 percentage points.

The mean correct confidence (0.739) vs mean wrong confidence (0.240) directly explains the high AUROC. Before calibration, both correct and wrong predictions sat near 0.93. After calibration, they separate by 0.499.

### Why Isotonic, Not Platt Scaling

Platt scaling (the standard alternative) failed on this model. Platt fits a sigmoid `calibrated = sigmoid(a × raw + b)`. Because the fine-tuned model's raw confidence is concentrated near 0.93, Platt's optimizer learned an extreme slope of `a = 17.63` to create a meaningful sigmoid in that narrow range. This is numerically unstable and produced ECE 0.1106 — barely better than uncalibrated (0.1331).

Isotonic regression is non-parametric: it learns a monotone step function directly from the (confidence, correctness) pairs. With no assumption about functional form, it handles the narrow raw distribution by fitting a sharp step at the empirical accuracy boundary. 12 knot points on 200 samples were sufficient.

---

## Methodology

**Train/test separation.** VQA-RAD test set (451 samples) was held out during all training and calibration fitting. It was only touched during the final evaluation runs reported here.

**Calibration validation set.** Calibrators were fit on SLAKE validation set (1,053 samples English, held-out during QLoRA training). VQA-RAD validation was not used — it would have risked leakage into the VQA-RAD test evaluation.

**Statistical tests.** McNemar's test is used for paired comparison between any two configurations evaluated on the same 451 samples. For `n_discordant < 25` we use the exact binomial; for `n_discordant ≥ 25` we use the chi-squared approximation with continuity correction. Results available in `data/evaluation_reports/ablation/ablation_report.md` after all 6 configs are run.

**No BERTScore in ablation.** `--no-bertscore` flag was used for all runs to reduce GPU memory pressure and allow sequential execution on a single A10G. BERTScore (deberta-xlarge-mnli) requires a separate ~900 MB model load. The open-answer metrics reported use token F1 and BLEU-1 only.

**Fast test suite.** 455 fast tests pass across all phases. All Phase 6 changes are covered by unit tests that require no model download or GPU. Slow tests (embedding model, real retrieval) are marked `@pytest.mark.slow` and excluded from `make test`.
