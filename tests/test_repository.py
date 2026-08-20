"""Repository behaviour, with the concurrency guarantee front and centre."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from uuid import uuid4

from invoice_agent.models import InventoryItem, Vendor
from invoice_agent.repository import SqliteInvoiceRepository


class TestRunStore:
    def test_save_then_find_by_content_hash(self, repo, make_run):
        run = make_run()
        repo.save(run)
        found = repo.find_by_content_hash(run.source.content_sha256)
        assert found is not None
        assert found.run_id == run.run_id

    def test_unknown_hash_returns_none(self, repo):
        assert repo.find_by_content_hash("0" * 64) is None

    def test_save_is_idempotent_on_run_id(self, repo, make_run):
        run = make_run()
        repo.save(run)
        repo.save(run.model_copy(update={"current_stage": "validate"}))
        runs = repo.find_by_invoice_number("INV-1001", "Widgets Inc.")
        assert len(runs) == 1
        assert runs[0].current_stage == "validate"

    def test_find_by_invoice_number_scopes_to_vendor(self, repo, make_run, make_extraction):
        repo.save(make_run())
        repo.save(make_run(extraction=make_extraction(vendor_name="Gadgets Co.")))
        assert len(repo.find_by_invoice_number("INV-1001", "Widgets Inc.")) == 1
        assert len(repo.find_by_invoice_number("INV-1001", "Gadgets Co.")) == 1

    def test_decimal_survives_the_round_trip(self, repo, make_run, make_extraction):
        run = make_run(extraction=make_extraction(total_amount=Decimal("22562.80")))
        repo.save(run)
        restored = repo.find_by_content_hash(run.source.content_sha256)
        assert restored.extraction.total_amount == Decimal("22562.80")


class TestIdentify:
    def test_first_caller_acquires(self, repo):
        result = repo.identify("INV-1004", "Precision Parts Ltd.", "fp-a", uuid4())
        assert result.acquired is True

    def test_second_caller_loses_and_sees_the_holder(self, repo):
        winner = uuid4()
        repo.identify("INV-1004", "Precision Parts Ltd.", "fp-a", winner)
        loser = repo.identify("INV-1004", "Precision Parts Ltd.", "fp-b", uuid4())
        assert loser.acquired is False
        assert loser.holder_run_id == winner
        # Different fingerprint -> caller raises DUPLICATE_INVOICE rather than skipping.
        assert loser.holder_fingerprint == "fp-a"

    def test_matching_fingerprint_is_visible_to_the_loser(self, repo):
        repo.identify("INV-1004", "Precision Parts Ltd.", "same", uuid4())
        loser = repo.identify("INV-1004", "Precision Parts Ltd.", "same", uuid4())
        assert loser.acquired is False
        assert loser.holder_fingerprint == "same"

    def test_different_vendors_do_not_collide(self, repo):
        assert repo.identify("INV-1001", "Widgets Inc.", "fp", uuid4()).acquired
        assert repo.identify("INV-1001", "Gadgets Co.", "fp", uuid4()).acquired

    def test_concurrent_callers_exactly_one_wins(self, db_path):
        """The race from architecture.md: two runs of INV-1004 in flight at once.

        Correctness lives in the UNIQUE constraint, not in scheduling, so both
        threads run the full path and SQLite picks the winner.
        """
        attempts = 12
        barrier = threading.Barrier(attempts)

        def attempt(_: int):
            repo = SqliteInvoiceRepository(db_path)
            barrier.wait()
            return repo.identify("INV-1004", "Precision Parts Ltd.", "fp", uuid4())

        with ThreadPoolExecutor(max_workers=attempts) as pool:
            results = list(pool.map(attempt, range(attempts)))

        winners = [r for r in results if r.acquired]
        assert len(winners) == 1
        # Every loser agrees on who holds it.
        holders = {r.holder_run_id for r in results}
        assert holders == {winners[0].holder_run_id}


class TestReferenceData:
    def test_inventory_loads_with_decimal_prices(self, repo):
        inventory = repo.load_inventory()
        assert set(inventory) == {"WidgetA", "WidgetB", "GadgetX", "FakeItem"}
        assert inventory["WidgetA"] == InventoryItem(
            item="WidgetA", stock=15, unit_price=Decimal("250.00"), category="widgets"
        )

    def test_zero_stock_item_is_present_not_missing(self, repo):
        # FakeItem must exist with stock 0 -- that is ZERO_STOCK_ITEM,
        # which is a hard_block, not ITEM_NOT_FOUND.
        assert repo.load_inventory()["FakeItem"].stock == 0

    def test_known_vendor_returns_history(self, repo):
        vendor = repo.find_vendor("Widgets Inc.")
        assert isinstance(vendor, Vendor)
        assert vendor.invoice_count > 0
        assert vendor.status == "active"

    def test_watchlist_vendor_is_flagged(self, repo):
        # INV-1012: "formerly FastShip Ltd." -- recent rename, near-zero history.
        vendor = repo.find_vendor("QuickShip Distributers")
        assert vendor.status == "watchlist"

    def test_suspicious_vendors_are_absent_by_design(self, repo):
        # Drives UNKNOWN_VENDOR on exactly INV-1003 and INV-1008.
        assert repo.find_vendor("Fraudster LLC") is None
        assert repo.find_vendor("NoProd Industries") is None
