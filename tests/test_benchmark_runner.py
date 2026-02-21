"""Tests for BenchmarkRunner with MockVLMBackend (no model downloads)."""

import json

from radiology_vqa.benchmark.runner import BenchmarkResult, BenchmarkRunner


def test_runner_produces_result(mock_vlm_backend, sample_vqa_samples_for_benchmark):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")

    assert isinstance(result, BenchmarkResult)
    assert result.model_name == "mock"
    assert result.dataset == "vqa_rad"
    assert result.split == "test"
    assert result.total_samples == len(sample_vqa_samples_for_benchmark)


def test_runner_per_sample_count(mock_vlm_backend, sample_vqa_samples_for_benchmark):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")

    assert len(result.per_sample) == len(sample_vqa_samples_for_benchmark)


def test_runner_per_sample_required_fields(mock_vlm_backend, sample_vqa_samples_for_benchmark):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")

    required_keys = {
        "sample_id",
        "question",
        "ground_truth",
        "predicted_answer",
        "answer_type",
        "correct",
        "confidence",
        "latency_seconds",
    }
    for record in result.per_sample:
        assert required_keys.issubset(record.keys()), f"Missing keys: {required_keys - record.keys()}"


def test_runner_max_samples(mock_vlm_backend, sample_vqa_samples_for_benchmark):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test", max_samples=3)

    assert result.total_samples == 3
    assert len(result.per_sample) == 3


def test_runner_metrics_keys(mock_vlm_backend, sample_vqa_samples_for_benchmark):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")

    required_metric_keys = {
        "overall_accuracy",
        "closed_accuracy",
        "open_accuracy",
        "total",
        "total_closed",
        "total_open",
    }
    assert required_metric_keys.issubset(result.metrics.keys())


def test_runner_mock_accuracy(mock_vlm_backend, sample_vqa_samples_for_benchmark):
    """MockVLMBackend always returns 'yes'.
    5 closed samples have answer='yes'  → closed_accuracy = 1.0
    5 open samples have answer='pneumonia' → open_accuracy = 0.0
    overall = 0.5
    """
    import pytest

    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")

    assert result.metrics["closed_accuracy"] == pytest.approx(1.0)
    assert result.metrics["open_accuracy"] == pytest.approx(0.0)
    assert result.metrics["overall_accuracy"] == pytest.approx(0.5)


def test_runner_runtime_keys(mock_vlm_backend, sample_vqa_samples_for_benchmark):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")

    assert "total_seconds" in result.runtime
    assert "mean_latency_seconds" in result.runtime
    assert "samples_per_second" in result.runtime


def test_save_result_writes_valid_json(
    mock_vlm_backend, sample_vqa_samples_for_benchmark, tmp_path
):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")
    path = runner.save_result(result, tmp_path)

    assert path.exists()
    assert path.suffix == ".json"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "mock"
    assert data["total_samples"] == len(sample_vqa_samples_for_benchmark)
    assert "metrics" in data
    assert "per_sample" in data
    assert "runtime" in data
    assert "timestamp" in data


def test_save_result_filename_contains_model_and_dataset(
    mock_vlm_backend, sample_vqa_samples_for_benchmark, tmp_path
):
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")
    path = runner.save_result(result, tmp_path)

    assert "mock" in path.name
    assert "vqa_rad" in path.name
    assert "test" in path.name


def test_save_result_creates_output_dir(
    mock_vlm_backend, sample_vqa_samples_for_benchmark, tmp_path
):
    """save_result must create the output directory if it doesn't exist."""
    runner = BenchmarkRunner(mock_vlm_backend)
    result = runner.run(sample_vqa_samples_for_benchmark, "vqa_rad", "test")
    new_dir = tmp_path / "nested" / "benchmarks"
    path = runner.save_result(result, new_dir)

    assert new_dir.exists()
    assert path.exists()
