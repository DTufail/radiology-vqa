"""Benchmark runner: execute VLM inference over a dataset split and compute metrics."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from radiology_vqa.benchmark.metrics import compute_metrics, is_match, normalize_answer
from radiology_vqa.schema import VQASample
from radiology_vqa.vlm.interface import VLMInterface

logger = logging.getLogger(__name__)


class BenchmarkResult(BaseModel):
    """Persistent benchmark output. Saved as JSON after each run."""

    model_name: str
    dataset: str
    split: str
    total_samples: int
    metrics: dict
    per_sample: list[dict]
    runtime: dict
    timestamp: str
    config: dict


class BenchmarkRunner:
    """Run a VLM backend over a dataset split and compute metrics.

    The runner is agnostic to the VLM backend — it depends only on
    :class:`VLMInterface`. Pass any conforming backend (including
    ``MockVLMBackend`` in tests).
    """

    def __init__(self, vlm: VLMInterface) -> None:
        self._vlm = vlm

    def run(
        self,
        samples: list[VQASample],
        dataset_name: str,
        split: str,
        max_samples: int | None = None,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        """Run VLM inference over samples and collect predictions.

        Args:
            samples: Dataset split to evaluate.
            dataset_name: Human-readable name (e.g. "vqa_rad").
            split: Split name (e.g. "test", "validate").
            max_samples: If set, evaluate only the first N samples.
                Useful for quick sanity checks.
            batch_size: Number of samples per forward pass. batch_size > 1
                requires the backend to support true batching (e.g. BLIP-2).
                LLaVA-Med should use batch_size=1 (sequential).

        Returns:
            :class:`BenchmarkResult` with per-sample records and aggregate metrics.
        """
        if max_samples is not None:
            samples = samples[:max_samples]

        logger.info(
            "Starting benchmark: model=%s dataset=%s split=%s n=%d batch_size=%d",
            self._vlm.model_name,
            dataset_name,
            split,
            len(samples),
            batch_size,
        )

        per_sample: list[dict] = []
        total_latency = 0.0

        for chunk_start in range(0, len(samples), batch_size):
            chunk = samples[chunk_start : chunk_start + batch_size]

            if batch_size == 1:
                predictions = [self._vlm.predict(chunk[0].image, chunk[0].question)]
            else:
                predictions = self._vlm.predict_batch(
                    [(s.image, s.question) for s in chunk]
                )

            for sample, prediction in zip(chunk, predictions):
                correct = is_match(prediction.answer, sample.answer, sample.answer_type)
                total_latency += prediction.latency_seconds
                per_sample.append(
                    {
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "ground_truth": sample.answer,
                        "predicted_answer": normalize_answer(prediction.answer),
                        "answer_type": sample.answer_type,
                        "correct": correct,
                        "confidence": prediction.confidence,
                        "latency_seconds": prediction.latency_seconds,
                    }
                )

            done = len(per_sample)
            if done % 50 < batch_size or done == len(samples):
                acc_so_far = sum(s["correct"] for s in per_sample) / done
                logger.info(
                    "  [%d/%d] running accuracy=%.3f mean_latency=%.2fs",
                    done,
                    len(samples),
                    acc_so_far,
                    total_latency / done,
                )

        metrics = compute_metrics(per_sample)
        n = len(per_sample)
        mean_latency = total_latency / n if n > 0 else 0.0

        runtime = {
            "total_seconds": round(total_latency, 2),
            "mean_latency_seconds": round(mean_latency, 3),
            "median_latency_seconds": round(
                sorted(s["latency_seconds"] for s in per_sample)[n // 2] if n > 0 else 0.0, 3
            ),
            "samples_per_second": round(n / total_latency, 2) if total_latency > 0 else 0.0,
        }

        logger.info(
            "Benchmark complete: overall_acc=%.3f closed_acc=%.3f open_acc=%.3f",
            metrics["overall_accuracy"],
            metrics["closed_accuracy"],
            metrics["open_accuracy"],
        )

        return BenchmarkResult(
            model_name=self._vlm.model_name,
            dataset=dataset_name,
            split=split,
            total_samples=n,
            metrics=metrics,
            per_sample=per_sample,
            runtime=runtime,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config={"model_name": self._vlm.model_name, "batch_size": batch_size},
        )

    def save_result(self, result: BenchmarkResult, output_dir: Path) -> Path:
        """Save benchmark result as indented JSON.

        Filename format: ``{model}_{dataset}_{split}_{timestamp}.json``
        Images are never included in the output.

        Args:
            result: Completed benchmark result.
            output_dir: Directory to write the JSON file.

        Returns:
            Path to the written file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a filesystem-safe timestamp
        ts = result.timestamp.replace(":", "-").replace("+", "").split(".")[0]
        safe_model = result.model_name.replace("/", "_")
        filename = f"{safe_model}_{result.dataset}_{result.split}_{ts}.json"
        path = output_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info("Saved benchmark result: %s", path)
        return path
