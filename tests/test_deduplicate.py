"""Deduplicate behaviour: identity claiming and the three loss outcomes.

No LLM involved, so these set up the "holder" side directly through the repo
rather than running the pipeline twice -- per the component brief, nothing
calls repo.save() yet, so find_by_invoice_number only sees what a test saves.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from invoice_agent.models import (
    ExtractedInvoice,
    FindingCode,
    PaymentResult,
    ValidationReport,
)
from invoice_agent.nodes.deduplicate import make_deduplicate


class TestFirstCallerAcquires:
    def test_returns_a_clean_claim_with_no_finding(self, deps, make_run):
        run = make_run()
        result = make_deduplicate(deps)(run)

        assert set(result) == {"identity"}
        identity = result["identity"]
        assert identity.acquired is True
        assert identity.holder_run_id == run.run_id
        assert identity.finding is None
        assert identity.fingerprint == identity.holder_fingerprint

    def test_identity_row_is_actually_written(self, deps, repo, make_run):
        run = make_run()
        make_deduplicate(deps)(run)

        probe = repo.identify(
            run.extraction.invoice_number, run.extraction.vendor_name, "irrelevant", uuid4()
        )
        assert probe.acquired is False
        assert probe.holder_run_id == run.run_id


class TestExactDuplicate:
    def test_same_content_emits_an_info_finding(self, deps, repo, make_run, make_extraction):
        holder = make_run()
        make_deduplicate(deps)(holder)  # first caller claims the identity

        loser = make_run(extraction=make_extraction())  # identical content
        result = make_deduplicate(deps)(loser)

        identity = result["identity"]
        assert identity.acquired is False
        finding = identity.finding
        assert finding is not None
        assert finding.code == FindingCode.EXACT_DUPLICATE
        assert finding.severity == "info"
        assert finding.repairable is False
        assert str(holder.run_id) in finding.message

    def test_does_not_touch_validations(self, deps, repo, make_run, make_extraction):
        make_deduplicate(deps)(make_run())
        earlier = ValidationReport(
            pass_number=1, effective_invoice=ExtractedInvoice(), validated_at=datetime.now(UTC)
        )
        loser = make_run(extraction=make_extraction(), validations=[earlier])

        result = make_deduplicate(deps)(loser)

        # Deduplicate isn't a validation pass -- its finding lives on
        # run.identity, not appended to run.validations.
        assert "validations" not in result


class TestConflictingDuplicate:
    def test_different_content_unpaid_holder_is_duplicate_invoice(
        self, deps, repo, make_run, make_extraction
    ):
        holder = make_run()
        make_deduplicate(deps)(holder)

        loser = make_run(extraction=make_extraction(total_amount=Decimal("999999.00")))
        result = make_deduplicate(deps)(loser)

        finding = result["identity"].finding
        assert finding.code == FindingCode.DUPLICATE_INVOICE
        assert finding.severity == "blocking"

    def test_different_content_paid_holder_is_duplicate_of_paid(
        self, deps, repo, make_run, make_extraction
    ):
        holder = make_run(payment=PaymentResult(idempotency_key="k", status="paid"))
        make_deduplicate(deps)(holder)
        repo.save(holder)  # only saved runs are visible to find_by_invoice_number

        loser = make_run(extraction=make_extraction(total_amount=Decimal("999999.00")))
        result = make_deduplicate(deps)(loser)

        finding = result["identity"].finding
        assert finding.code == FindingCode.DUPLICATE_OF_PAID_INVOICE

    def test_checks_every_run_under_the_identity_not_just_the_holder(
        self, deps, repo, make_run, make_extraction
    ):
        # A run that lost identify() isn't guaranteed to be blocked from ever
        # reaching pay (triage's hard_block list only names duplicate-of-paid,
        # not a plain duplicate) -- so a non-holder run could be the one
        # that's actually paid. The check has to see that, not just the
        # current holder's row.
        holder = make_run()
        make_deduplicate(deps)(holder)  # holder wins identify()
        repo.save(holder)  # holder itself is unpaid

        already_paid = make_run(extraction=make_extraction(total_amount=Decimal("42.00")))
        repo.save(
            already_paid.model_copy(
                update={"payment": PaymentResult(idempotency_key="k", status="paid")}
            )
        )

        loser = make_run(extraction=make_extraction(total_amount=Decimal("999999.00")))
        result = make_deduplicate(deps)(loser)

        assert result["identity"].finding.code == FindingCode.DUPLICATE_OF_PAID_INVOICE

    def test_holder_not_yet_saved_defaults_to_unpaid(self, deps, repo, make_run, make_extraction):
        # A race still in progress: the holder claimed the identity but
        # hasn't reached repo.save() yet.
        holder = make_run()
        make_deduplicate(deps)(holder)

        loser = make_run(extraction=make_extraction(total_amount=Decimal("1.00")))
        result = make_deduplicate(deps)(loser)

        assert result["identity"].finding.code == FindingCode.DUPLICATE_INVOICE


class TestIdentityFallback:
    def test_missing_vendor_still_collides_on_identity(
        self, deps, repo, make_run, make_extraction
    ):
        # None would fall through to SQLite NULL, and NULL != NULL under
        # UNIQUE -- two vendorless invoices would never collide.
        holder = make_run(extraction=make_extraction(vendor_name=None))
        make_deduplicate(deps)(holder)

        loser = make_run(extraction=make_extraction(vendor_name=None))
        result = make_deduplicate(deps)(loser)

        assert result["identity"].finding.code == FindingCode.EXACT_DUPLICATE
