PY ?= python
PKG := kgllm_pharm

.PHONY: help install dev lint format type test smoke docker clean

help:
	@echo "targets: install dev lint format type test smoke docker clean"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	black . && isort .

type:
	mypy --strict src/$(PKG)

test:
	pytest -q

smoke:
	$(PY) -m $(PKG).gateway.counter train --config diagrams/experiment/_smoke.yaml

docker:
	docker build -t $(PKG):latest .

clean:
	rm -rf build dist *.egg-info src/*.egg-info .mypy_cache .ruff_cache .pytest_cache runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
