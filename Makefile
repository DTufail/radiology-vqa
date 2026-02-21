.PHONY: install test test-slow lint format download-data validate-data \
        build-index build-index-kg test-retrieval \
        benchmark benchmark-quick benchmark-compare quick-inference

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short -m "not slow"

test-slow:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

download-data:
	python scripts/download_datasets.py

validate-data:
	python scripts/validate_data.py

build-index:
	python scripts/build_index.py

build-index-kg:
	python scripts/build_index.py --source kg

test-retrieval:
	python scripts/test_retrieval.py

benchmark:
	python scripts/run_benchmark.py

benchmark-quick:
	python scripts/run_benchmark.py --max-samples 50

benchmark-compare:
	python scripts/run_benchmark.py --compare

quick-inference:
	python scripts/quick_inference.py --dataset vqa_rad --index 0
