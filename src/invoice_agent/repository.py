"""SQLite-backed InvoiceRepository.

The only way one component reads another component's output (invariant 10).
Validation needs to know whether a prior version of an invoice was paid; it
reads the run store rather than importing the payment module.

Safe to share across threads: every operation opens its own connection.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from invoice_agent.db import connect, init_db
from invoice_agent.models import Identification, InventoryItem, InvoiceRun, Vendor


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteInvoiceRepository:
    """Implements the InvoiceRepository protocol from models.py."""

    def __init__(self, db_path: str | Path, *, initialize: bool = False) -> None:
        self.db_path = str(db_path)
        if initialize:
            init_db(self.db_path)

    # -- run store ---------------------------------------------------------

    def save(self, run: InvoiceRun) -> None:
        """Upsert by run_id. Called after every node, so the UI can watch a run
        progress and a crash leaves the last completed stage on disk."""
        extraction = run.extraction
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, status, current_stage, content_sha256,
                                  invoice_number, vendor, payload, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status         = excluded.status,
                    current_stage  = excluded.current_stage,
                    content_sha256 = excluded.content_sha256,
                    invoice_number = excluded.invoice_number,
                    vendor         = excluded.vendor,
                    payload        = excluded.payload,
                    updated_at     = excluded.updated_at
                """,
                (
                    str(run.run_id),
                    run.status,
                    run.current_stage,
                    run.source.content_sha256,
                    extraction.invoice_number if extraction else None,
                    extraction.vendor_name if extraction else None,
                    run.model_dump_json(),
                    run.started_at.isoformat(),
                    _now(),
                ),
            )

    def find_by_content_hash(self, sha256: str) -> InvoiceRun | None:
        """Byte-identical duplicate check. Free, and runs before any LLM call."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM runs WHERE content_sha256 = ? ORDER BY started_at LIMIT 1",
                (sha256,),
            ).fetchone()
        return InvoiceRun.model_validate_json(row["payload"]) if row else None

    def find_by_invoice_number(self, number: str, vendor: str) -> list[InvoiceRun]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT payload FROM runs WHERE invoice_number = ? AND vendor = ?"
                " ORDER BY started_at",
                (number, vendor),
            ).fetchall()
        return [InvoiceRun.model_validate_json(r["payload"]) for r in rows]

    # -- identity ----------------------------------------------------------

    def identify(
        self, number: str, vendor: str, fingerprint: str, run_id: UUID
    ) -> Identification:
        """Atomically claim an (invoice_number, vendor) identity.

        Correctness lives in the UNIQUE constraint, not in scheduling -- so
        invoices process in parallel and SQLite picks the winner. The loser
        never waits; it only needs to know a holder exists and whether the
        content agrees.
        """
        with connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO invoice_identities"
                    " (invoice_number, vendor, fingerprint, run_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (number, vendor, fingerprint, str(run_id), _now()),
                )
            except sqlite3.IntegrityError:
                holder = conn.execute(
                    "SELECT run_id, fingerprint FROM invoice_identities"
                    " WHERE invoice_number = ? AND vendor = ?",
                    (number, vendor),
                ).fetchone()
                return Identification(
                    acquired=False,
                    holder_run_id=UUID(holder["run_id"]),
                    holder_fingerprint=holder["fingerprint"],
                )
        return Identification(
            acquired=True, holder_run_id=run_id, holder_fingerprint=fingerprint
        )

    # -- reference data ----------------------------------------------------

    def load_inventory(self) -> dict[str, InventoryItem]:
        """Keyed by canonical item name -- what canonicalization resolves to."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT item, stock, unit_price, category FROM inventory"
            ).fetchall()
        return {
            r["item"]: InventoryItem(
                item=r["item"],
                stock=r["stock"],
                unit_price=Decimal(r["unit_price"]) if r["unit_price"] is not None else None,
                category=r["category"],
            )
            for r in rows
        }

    def find_vendor(self, name: str) -> Vendor | None:
        """Absent vendor returns None -- the caller emits UNKNOWN_VENDOR. Never
        invent history; the approval agent reasons over whatever this returns."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT name, first_seen, invoice_count, status FROM vendors WHERE name = ?",
                (name,),
            ).fetchone()
        return Vendor(**dict(row)) if row else None
