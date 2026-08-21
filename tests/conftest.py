"""Shared fixtures. Nobody writing a component test should be standing up a database.

Factories take keyword overrides, so a test states only the field it cares about:

    inv = make_extraction(total_amount=Decimal("100000"), vendor_name="Fraudster LLC")
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from invoice_agent.config import Settings
from invoice_agent.db import seed_db
from invoice_agent.models import ExtractedInvoice, InvoiceRun, LineItem, SourceDocument
from invoice_agent.obs import JsonLogger
from invoice_agent.repository import SqliteInvoiceRepository

SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "seed"
INVOICE_DIR = Path(__file__).resolve().parents[1] / "data" / "invoices"


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """A seeded database, one per test."""
    path = tmp_path / "test.db"
    seed_db(path, SEED_DIR)
    return str(path)


@pytest.fixture
def repo(db_path: str) -> SqliteInvoiceRepository:
    return SqliteInvoiceRepository(db_path)


@pytest.fixture
def settings(db_path: str) -> Settings:
    return Settings(db_path=db_path, seed_dir=str(SEED_DIR))


@pytest.fixture
def invoice_dir() -> Path:
    """The 20 sample invoices."""
    return INVOICE_DIR


@pytest.fixture
def make_source() -> Callable[..., SourceDocument]:
    def _make(text: str = "INVOICE\nTotal: $100.00", **overrides) -> SourceDocument:
        defaults = dict(
            run_id=uuid4(),
            source_path="data/invoices/invoice_1001.txt",
            source_filename="invoice_1001.txt",
            source_format="txt",
            raw_text=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            text_extraction_ok=True,
            loaded_at=datetime.now(UTC),
        )
        return SourceDocument(**{**defaults, **overrides})

    return _make


@pytest.fixture
def make_extraction() -> Callable[..., ExtractedInvoice]:
    def _make(**overrides) -> ExtractedInvoice:
        defaults = dict(
            invoice_number="INV-1001",
            vendor_name="Widgets Inc.",
            currency="USD",
            total_amount="5000.00",
            subtotal="5000.00",
            issue_date="2026-01-15",
            due_date="2026-02-01",
            line_items=[
                LineItem(raw_item_name="WidgetA", quantity=10, unit_price="250.00"),
                LineItem(raw_item_name="WidgetB", quantity=5, unit_price="500.00"),
            ],
        )
        return ExtractedInvoice(**{**defaults, **overrides})

    return _make


@pytest.fixture
def make_run(make_source, make_extraction) -> Callable[..., InvoiceRun]:
    def _make(run_id: UUID | None = None, **overrides) -> InvoiceRun:
        defaults = dict(
            run_id=run_id or uuid4(),
            status="running",
            current_stage="extract",
            source=make_source(),
            extraction=make_extraction(),
            started_at=datetime.now(UTC),
        )
        return InvoiceRun(**{**defaults, **overrides})

    return _make


@pytest.fixture
def log_stream() -> StringIO:
    """Captured JSON-lines output. Parse with `json.loads` per line."""
    return StringIO()


@pytest.fixture
def logger(log_stream: StringIO) -> JsonLogger:
    return JsonLogger(log_stream)
