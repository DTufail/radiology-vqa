"""Evaluation orchestrator for running the full pipeline on datasets.

Two evaluation modes:
- "agent": runs the full multi-agent pipeline (VLM + RAG + supervisor)
- "vlm_only": runs VLM.predict() directly (baseline for comparison)

Both modes produce the same EvaluationResult schema, enabling fair comparison.
"""

import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

from radiology_vqa.evaluation.agent_metrics import (
    abstention_rate as compute_abstention_rate,
    accuracy_when_answered as compute_accuracy_when_answered,
    citation_relevance as compute_citation_relevance,
    re_query_rate_from_counts,
)
from radiology_vqa.evaluation.calibration import (
    calibration_bins as compute_calibration_bins,
    confidence_discrimination as compute_confidence_discrimination,
    expected_calibration_error as compute_ece,
    threshold_analysis as compute_threshold_analysis,
)
from radiology_vqa.evaluation.metrics import compute_all_metrics, normalize_answer
from radiology_vqa.evaluation.result import EvaluationResult, PerSampleResult

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Run evaluation of the agent pipeline or VLM-only baseline.

    Usage::

        evaluator = AgentEvaluator()

        # Full agent pipeline evaluation
        agent_result = evaluator.evaluate(dataset="vqa_rad", split="test", mode="agent")

        # VLM-only baseline
        vlm_result = evaluator.evaluate(dataset="vqa_rad", split="test", mode="vlm_only")

        agent_result.save(Path("data/evaluation_reports/agent_vqa_rad_test.json"))
    """

    def __init__(self, config=None):
        """Initialize. Does NOT load models yet — that happens on first evaluate().

        Lazy loading because:
        1. The evaluator might be created just to load saved results.
        2. VLM takes ~90 seconds to load — don't pay this if not needed.
        3. Different modes need different resources.
        """
        if config is None:
            from radiology_vqa.config import settings

            config = settings
        self._config = config
        self._agent_runner = None  # lazy: created on first agent evaluate
        self._vlm = None  # lazy: created on first vlm_only evaluate

    def evaluate(
        self,
        dataset: str = "vqa_rad",
        split: str = "test",
        mode: str = "agent",
        max_samples: int | None = None,
        compute_bertscore: bool = True,
        save_intermediate: bool = True,
        resume_from: Path | None = None,
    ) -> EvaluationResult:
        """Run full evaluation on a dataset.

        Steps:
        1. Load dataset samples (sliced to max_samples if given).
        2. If resume_from provided, skip already-evaluated sample_ids.
        3. For each sample: run agent or VLM-only, record all metadata.
        4. Every 50 samples: save intermediate results (crash recovery).
        5. Log progress every 25 samples.
        6. Compute all metrics using Phase 5A pure functions.
        7. Return an EvaluationResult.

        Error handling per sample:
        - If a single sample fails, log a warning, record prediction="" correct=False,
          and continue.  The evaluation must complete.
        """
        t_start = time.perf_counter()

        samples = self._load_dataset(dataset, split, max_samples)
        n = len(samples)
        logger.info("[%s] Evaluating %s %s (%d samples)…", mode, dataset.upper(), split, n)
        print(f"[{mode}] Evaluating {dataset.upper()} {split} ({n} samples)…")

        per_sample_results: list[PerSampleResult] = []
        completed_ids: set[str] = set()

        intermediate_path = (
            self._config.eval_output_dir / f"intermediate_{mode}_{dataset}_{split}.json"
        )

        if resume_from is not None and resume_from.exists():
            loaded = self._load_intermediate(resume_from)
            per_sample_results = [PerSampleResult(**d) for d in loaded]
            completed_ids = {r.sample_id for r in per_sample_results}
            logger.info("[%s] Resuming from %d completed samples", mode, len(per_sample_results))
            print(f"[{mode}] Resuming from {len(per_sample_results)} completed samples")

        for sample in samples:
            if sample.sample_id in completed_ids:
                continue

            t_sample = time.perf_counter()
            try:
                if mode == "agent":
                    run_info = self._run_agent(sample.image, sample.question, sample.answer_type)
                else:
                    run_info = self._run_vlm_only(sample.image, sample.question)
            except Exception as exc:
                logger.warning(
                    "[%s] Sample %s failed: %s — recording as incorrect",
                    mode,
                    sample.sample_id,
                    exc,
                )
                run_info = {
                    "prediction": "",
                    "confidence": 0.0,
                    "decision": "",
                    "citations": [],
                    "reasoning": f"Error: {exc}",
                    "retrieval_query": "",
                    "visual_answer": "",
                    "retry_count": 0,
                    "latency_seconds": time.perf_counter() - t_sample,
                }

            correct = normalize_answer(run_info["prediction"]) == normalize_answer(sample.answer)
            per_sample_results.append(
                PerSampleResult(
                    sample_id=sample.sample_id,
                    question=sample.question,
                    ground_truth=sample.answer,
                    prediction=run_info["prediction"],
                    correct=correct,
                    answer_type=sample.answer_type,
                    confidence=run_info["confidence"],
                    latency_seconds=run_info.get("latency_seconds", 0.0),
                    decision=run_info.get("decision", ""),
                    citations=run_info.get("citations", []),
                    reasoning=run_info.get("reasoning", ""),
                    retrieval_query=run_info.get("retrieval_query", ""),
                    visual_answer=run_info.get("visual_answer", ""),
                    retry_count=run_info.get("retry_count", 0),
                )
            )

            done = len(per_sample_results)

            if done % 25 == 0 or done == n:
                correct_so_far = sum(r.correct for r in per_sample_results)
                abstained = sum(1 for r in per_sample_results if r.decision == "abstain")
                mean_lat = sum(r.latency_seconds for r in per_sample_results) / done
                eta_min = mean_lat * (n - done) / 60
                abstain_str = f" | {abstained} abstained" if mode == "agent" else ""
                msg = (
                    f"[{mode}] {done:4d}/{n} ({100*done/n:.1f}%) | "
                    f"{correct_so_far} correct{abstain_str} | "
                    f"~{mean_lat:.1f}s/sample | ETA: {eta_min:.0f}min"
                )
                logger.info(msg)
                print(msg)

            if save_intermediate and done % 50 == 0:
                self._save_intermediate(per_sample_results, intermediate_path)
                print(f"[{mode}]  ** Intermediate results saved ({done} samples) **")

        # ── Compute aggregate metrics ────────────────────────────────
        logger.info("[%s] Computing metrics…", mode)
        print(f"[{mode}] Computing metrics…")

        predictions = [r.prediction for r in per_sample_results]
        ground_truths = [r.ground_truth for r in per_sample_results]
        answer_types = [r.answer_type for r in per_sample_results]
        confidences = [r.confidence for r in per_sample_results]
        correct_bools = [r.correct for r in per_sample_results]
        decisions = [r.decision for r in per_sample_results]
        citations_list = [r.citations for r in per_sample_results]
        retry_counts = [r.retry_count for r in per_sample_results]

        if compute_bertscore:
            print(f"[{mode}] Computing BERTScore (this may take a minute)…")

        metrics = compute_all_metrics(
            predictions, ground_truths, answer_types, compute_bertscore=compute_bertscore
        )

        abs_rate = compute_abstention_rate(decisions) if mode == "agent" else 0.0
        acc_answered = (
            compute_accuracy_when_answered(predictions, ground_truths, decisions)
            if mode == "agent"
            else metrics["overall_accuracy"]
        )
        rq_rate = re_query_rate_from_counts(retry_counts) if mode == "agent" else 0.0
        cit_rel = (
            compute_citation_relevance(citations_list, ground_truths)
            if mode == "agent"
            else {"citation_hit_rate": 0.0}
        )

        ece_val = compute_ece(confidences, correct_bools)
        disc = compute_confidence_discrimination(confidences, correct_bools)
        cal_bins = compute_calibration_bins(confidences, correct_bools)
        thresh = compute_threshold_analysis(confidences, correct_bools)

        latencies = [r.latency_seconds for r in per_sample_results]
        total_sec = time.perf_counter() - t_start
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        median_lat = statistics.median(latencies) if latencies else 0.0

        correct_total = sum(correct_bools)
        abstained_total = sum(1 for d in decisions if d == "abstain")
        acc_pct = correct_total / len(per_sample_results) if per_sample_results else 0.0
        abstain_str = (
            f" | {abstained_total} abstained ({100*abstained_total/n:.1f}%)"
            if mode == "agent"
            else ""
        )
        print(
            f"[{mode}] Complete: {len(per_sample_results)}/{n} | "
            f"{correct_total} correct ({100*acc_pct:.1f}%){abstain_str}"
        )

        result = EvaluationResult(
            model_name=self._config.vlm_model_id,
            dataset=dataset,
            split=split,
            total_samples=len(per_sample_results),
            evaluation_mode=mode,
            timestamp=datetime.now().isoformat(),
            config_snapshot={
                "vlm_model_id": self._config.vlm_model_id,
                "vlm_backend": self._config.vlm_backend,
                "supervisor_high_confidence": self._config.supervisor_high_confidence,
                "supervisor_low_confidence": self._config.supervisor_low_confidence,
                "supervisor_max_retries": self._config.supervisor_max_retries,
            },
            overall_accuracy=metrics["overall_accuracy"],
            closed_accuracy=metrics["closed_accuracy"],
            open_accuracy=metrics["open_accuracy"],
            closed_precision=metrics["closed_precision"],
            closed_recall=metrics["closed_recall"],
            closed_f1=metrics["closed_f1"],
            closed_confusion=metrics["closed_confusion"],
            closed_count=metrics["closed_count"],
            open_token_f1=metrics["open_token_f1"],
            open_bleu_1=metrics["open_bleu_1"],
            open_bertscore_f1=metrics["open_bertscore_f1"],
            open_bertscore_precision=metrics["open_bertscore_precision"],
            open_bertscore_recall=metrics["open_bertscore_recall"],
            open_count=metrics["open_count"],
            abstention_rate=abs_rate,
            accuracy_when_answered=acc_answered,
            re_query_rate=rq_rate,
            citation_relevance_hit_rate=cit_rel.get("citation_hit_rate", 0.0),
            ece=ece_val,
            mean_correct_confidence=disc["mean_correct_confidence"],
            mean_wrong_confidence=disc["mean_wrong_confidence"],
            confidence_auroc=disc["auroc"],
            calibration_bins=cal_bins,
            threshold_analysis=thresh,
            total_seconds=total_sec,
            mean_latency_seconds=mean_lat,
            median_latency_seconds=median_lat,
            per_sample=per_sample_results,
        )

        logger.info(
            "[%s] Done. Overall: %.1f%% | ECE: %.3f | AUROC: %.3f",
            mode,
            100 * result.overall_accuracy,
            result.ece,
            result.confidence_auroc,
        )
        return result

    def _run_agent(self, image, question: str, answer_type: str = "") -> dict:
        """Run agent pipeline on one sample.

        Lazily initializes AgentRunner on first call.
        Returns dict with: prediction, confidence, decision, citations,
        reasoning, retrieval_query, visual_answer, retry_count, latency_seconds.
        """
        if self._agent_runner is None:
            from radiology_vqa.graph.runner import AgentRunner

            logger.info("Loading AgentRunner (VLM + Retriever)…")
            self._agent_runner = AgentRunner(self._config)

        t0 = time.perf_counter()
        output = self._agent_runner.run_query(image, question, answer_type=answer_type)
        latency = time.perf_counter() - t0

        return {
            "prediction": output.answer,
            "confidence": output.confidence,
            "decision": output.decision,
            "citations": output.citations,
            "reasoning": output.reasoning,
            "retrieval_query": output.retrieval_query,
            "visual_answer": output.visual_answer,
            "retry_count": 0,  # SystemOutput doesn't expose retry_count
            "latency_seconds": latency,
        }

    def _run_vlm_only(self, image, question: str) -> dict:
        """Run VLM directly on one sample (no agent pipeline).

        Lazily initializes VLM backend on first call.
        Returns dict with: prediction, confidence, latency_seconds.
        decision="" citations=[] for VLM-only mode.
        """
        if self._vlm is None:
            from radiology_vqa.vlm.factory import create_vlm_backend

            logger.info("Loading VLM backend: %s", self._config.vlm_backend)
            self._vlm = create_vlm_backend(self._config)

        t0 = time.perf_counter()
        pred = self._vlm.predict(image, question)
        latency = time.perf_counter() - t0

        return {
            "prediction": pred.answer,
            "confidence": pred.confidence,
            "decision": "answer",
            "citations": [],
            "reasoning": "",
            "retrieval_query": "",
            "visual_answer": pred.answer,
            "retry_count": 0,
            "latency_seconds": latency,
        }

    def _load_dataset(self, dataset: str, split: str, max_samples: int | None) -> list:
        """Load dataset samples. Returns list of VQASample."""
        if dataset == "vqa_rad":
            from radiology_vqa.loader import load_vqa_rad

            samples = load_vqa_rad(split)
        elif dataset == "pathvqa":
            from radiology_vqa.loader import load_pathvqa

            samples = load_pathvqa(split)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset!r}. Supported: 'vqa_rad', 'pathvqa'."
            )

        if max_samples is not None:
            samples = samples[:max_samples]
        return samples

    def _save_intermediate(
        self,
        per_sample: list[PerSampleResult],
        path: Path,
    ) -> None:
        """Save intermediate per-sample results to JSON for crash recovery."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in per_sample]
        path.write_text(json.dumps(data, indent=2))
        logger.debug("Intermediate results saved: %d samples → %s", len(data), path)

    def _load_intermediate(self, path: Path) -> list[dict]:
        """Load intermediate per-sample dicts for resume."""
        return json.loads(path.read_text())
