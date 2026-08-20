VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: setup lint fmt test seed run clean

setup:
	uv venv --python 3.14
	uv pip install --python $(PY) -e ".[dev]"

lock:
	uv pip compile pyproject.toml --extra dev --python-version 3.14 -o requirements.txt

lint:
	$(VENV)/bin/ruff check .

fmt:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

test:
	$(PY) -m pytest

seed:
	$(PY) -m invoice_agent.db --seed

run:
	$(VENV)/bin/invoice-agent $(ARGS)

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ invoices.db invoices.db-wal invoices.db-shm
