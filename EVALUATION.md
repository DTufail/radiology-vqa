# Evaluation

Complete evaluation results for the Grounded Multi-Agent Radiology VQA system across all six ablation configurations. Dataset: VQA-RAD test split (451 samples, 251 closed / 200 open). All runs use `--no-bertscore` except where noted.

---

## 6-Configuration Ablation

Each configuration isolates one component's contribution by changing exactly one variable from the previous config.

| # | Config | VLM | LoRA | RAG | KG | Agreement | Cal. | Overall Acc | Acc (ans) | Abstain | ECE | AUROC |
|---|--------|-----|------|-----|-----|-----------|------|-------------|-----------|---------|-----|-------|
| 1 | baseline_vlm | ZS | — | — | — | — | — | 41.5% | 41.5% | 0% | 0.434 | 0.769 |
| 2 | baseline_agent | ZS | — | Yes | Original | Keyword | — | 31.9% | 46.8% | 31.7% | 0.188 | 0.765 |
| 3 | finetuned_vlm | FT | ✓ | — | — | — | — | 50.8% | 50.8% | 0% | 0.351 | 0.751 |
| 4 | finetuned_agent | FT | ✓ | Yes | Original | Keyword | — | 36.8% | 53.9% | 31.7% | 0.132 | 0.819 |
| 5 | full_pipeline | FT | ✓ | Yes | Expanded | Embed | — | 42.1% | 52.3% | 19.5% | 0.214 | 0.761 |
| 6 | full_calibrated | FT | ✓ | Yes | Expanded | Embed | Isotonic | 35.5% | 60.2% | 41.0% | 0.075 | 0.868 |

> ZS = zero-shot (no adapter). FT = QLoRA fine-tuned. Original KG = SLAKE KG only (2,987 docs). Expanded KG = SLAKE KG + RadLex + QA pseudo-docs (13,435 docs). Acc (ans) = accuracy on non-abstained samples only.

### Per Question-Type (available configs)

| # | Config | Closed Acc | Open Acc | Closed F1 | Closed n | Open n |
|---|--------|-----------|---------|----------|--------|------|
| 1 | baseline_vlm | 61.4% | 16.5% | 0.614 | 251 | 200 |
| 2 | baseline_agent | 50.2% | 9.0% | 0.655 | 251 | 200 |
| 3 | finetuned_vlm | 71.7% | 24.5% | 0.717 | 251 | 200 |
| 4 | finetuned_agent | 55.4% | 13.5% | 0.652 | 251 | 200 |
| 5 | full_pipeline | 58.6% | 21.5% | 0.684 | 251 | 200 |
| 6 | full_calibrated | 50.6% | 16.5% | 0.684 | 147 | 119 |

> Config 6 closed/open counts are lower because 41% of samples are abstained on; the denominators reflect answered samples only.

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
