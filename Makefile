.PHONY: install test lint format demo benchmark determinism clean all

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

demo:
	python examples/run_demo.py

benchmark:
	python cli.py benchmark

determinism:
	python cli.py benchmark --determinism

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

# 一键收口：格式 + 静态 + 测试 + 基准
all: format lint test benchmark
