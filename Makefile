.PHONY: install test test-slow lint format download-data validate-data \
        build-index build-index-kg test-retrieval \
        benchmark benchmark-quick benchmark-compare quick-inference \
        run-agent run-agent-batch \
        eval-quick eval-full eval-agent eval-vlm report

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

run-agent:
	python scripts/run_agent.py --dataset vqa_rad --index 0

run-agent-batch:
	python scripts/run_agent.py --dataset vqa_rad --range 0 10

eval-quick:
	python scripts/run_evaluation.py --mode compare --max-samples 20

eval-full:
	python scripts/run_evaluation.py --mode compare

eval-agent:
	python scripts/run_evaluation.py --mode agent

eval-vlm:
	python scripts/run_evaluation.py --mode vlm_only

report:
	python scripts/generate_report.py \
		--agent data/evaluation_reports/agent_result.json \
		--baseline data/evaluation_reports/baseline_result.json
