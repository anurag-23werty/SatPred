.DEFAULT_GOAL := help
CONFIG ?= configs/default.yaml

.PHONY: help install lint format typecheck test test-data coverage check download verify prepare report clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install git hooks
	uv sync
	uv run pre-commit install

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Static type check (mypy --strict)
	uv run mypy

test:  ## Unit tests (no dataset needed)
	uv run pytest -m "not data"

test-data:  ## Integration tests against the real dataset
	uv run pytest -m data

coverage:  ## Unit tests with coverage report
	uv run pytest -m "not data" --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs

download:  ## Download and verify the NASA SMAP/MSL dataset
	uv run sathealth download --config $(CONFIG)

verify:  ## Re-verify checksums of the downloaded dataset
	uv run sathealth verify --config $(CONFIG)

prepare:  ## Validate, split, scale and store the dataset
	uv run sathealth prepare --config $(CONFIG)

report:  ## Regenerate docs/DATA_CARD.md from the dataset
	uv run sathealth report --config $(CONFIG) --out docs/DATA_CARD.md

clean:  ## Remove caches (keeps data/)
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
