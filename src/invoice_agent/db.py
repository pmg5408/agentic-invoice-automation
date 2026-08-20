"""SQLite schema, connection handling, and the seed loader.

Run ``python -m invoice_agent.db --seed`` (or ``make seed``) to build the
database from ``data/seed/``.

Money is stored as TEXT, never REAL. A REAL column silently turns Decimal into
float, which is the bug invariant 2 exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from invoice_agent.config import Settings, get_settings
from invoice_agent.models import InventoryItem, Vendor

SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory (
    item        TEXT PRIMARY KEY,
    stock       INTEGER NOT NULL,
    unit_price  TEXT,
    category    TEXT
);

CREATE TABLE IF NOT EXISTS vendors (
    name           TEXT PRIMARY KEY,
    first_seen     TEXT    NOT NULL,
    invoice_count  INTEGER NOT NULL DEFAULT 0,
    status         TEXT    NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    current_stage   TEXT NOT NULL,
    content_sha256  TEXT,
    invoice_number  TEXT,
    vendor          TEXT,
    payload         TEXT NOT NULL,   -- serialized InvoiceRun
    started_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_content  ON runs (content_sha256);
CREATE INDEX IF NOT EXISTS idx_runs_identity ON runs (invoice_number, vendor);

-- One row per (invoice_number, vendor). The UNIQUE constraint is where
-- duplicate correctness lives, so the pipeline never has to serialize.
CREATE TABLE IF NOT EXISTS invoice_identities (
    invoice_number  TEXT NOT NULL,
    vendor          TEXT NOT NULL,
    fingerprint     TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (invoice_number, vendor)
);
"""


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection with the pragmas this system depends on.

    A fresh connection per operation, so repositories are safe to share across
    threads -- sqlite3 connections are not.
    """
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Without WAL, concurrent runs hit "database is locked".
    conn.execute("PRAGMA journal_mode=WAL")
    # Wait rather than failing instantly when another writer holds the lock.
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def load_seed_files(seed_dir: str | Path) -> tuple[list[InventoryItem], list[Vendor]]:
    """Parse and validate the seed JSON without touching the database."""
    seed = Path(seed_dir)
    inventory = [InventoryItem(**r) for r in json.loads((seed / "inventory.json").read_text())]
    vendors = [Vendor(**r) for r in json.loads((seed / "vendors.json").read_text())]
    return inventory, vendors


def seed_db(db_path: str | Path, seed_dir: str | Path) -> tuple[int, int]:
    """Load reference data. Idempotent -- re-running replaces the catalogue."""
    inventory, vendors = load_seed_files(seed_dir)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT OR REPLACE INTO inventory (item, stock, unit_price, category)"
            " VALUES (?, ?, ?, ?)",
            [
                (i.item, i.stock, None if i.unit_price is None else str(i.unit_price), i.category)
                for i in inventory
            ],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO vendors (name, first_seen, invoice_count, status)"
            " VALUES (?, ?, ?, ?)",
            [(v.name, v.first_seen.isoformat(), v.invoice_count, v.status) for v in vendors],
        )
    return len(inventory), len(vendors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize and seed the invoice database.")
    parser.add_argument("--seed", action="store_true", help="load data/seed/ reference data")
    parser.add_argument("--db", default=None, help="database path (default: from config)")
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    db_path = args.db or settings.db_path

    if args.seed:
        items, vendors = seed_db(db_path, settings.seed_dir)
        print(f"seeded {db_path}: {items} inventory items, {vendors} vendors")
    else:
        init_db(db_path)
        print(f"initialized {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
