"""Node: deduplicate.

Brief : docs/components.md section 3 -- Deduplicate
LLM   : No
In    : run.extraction
Out   : identity row; on loss, a DUPLICATE_* or EXACT_DUPLICATE finding

Build the fingerprint (hash of line items + quantities + total), call
deps.repo.identify(), and handle the loss.

Equal fingerprint -> EXACT_DUPLICATE (info, no human -- decision 19 has
triage skip recommend/critique when it sees this). Different fingerprint ->
DUPLICATE_INVOICE, or DUPLICATE_OF_PAID_INVOICE when this identity has
already resulted in a payment -- checked across every run under the
identity, not just the one that currently holds it (see _identity_already_paid).

The loser never waits for the winner. It only needs to know a holder exists
and whether the content agrees.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import (
    ExtractedInvoice,
    Finding,
    FindingCode,
    InvoiceRun,
    InvoiceScope,
    ValidationReport,
)


def _fingerprint(extraction: ExtractedInvoice) -> str:
    """Hash of what actually determines whether two extractions agree: the
    line items, their quantities, and the total. Not invoice_number/vendor --
    those are the identity being claimed, not the content being compared."""
    payload = {
        "line_items": [
            {
                "raw_item_name": li.raw_item_name,
                "quantity": li.quantity,
                "unit_price": str(li.unit_price) if li.unit_price is not None else None,
            }
            for li in extraction.line_items
        ],
        "total_amount": (
            str(extraction.total_amount) if extraction.total_amount is not None else None
        ),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _identity_fields(extraction: ExtractedInvoice) -> tuple[str, str]:
    # SQLite's UNIQUE treats every NULL as distinct from every other NULL, so
    # two vendorless or numberless invoices would never collide on identity
    # if we passed None through. Empty string collides correctly instead.
    return extraction.invoice_number or "", extraction.vendor_name or ""


def _identity_already_paid(deps: Deps, number: str, vendor: str) -> bool:
    """Has *any* run under this identity already been paid -- not just the
    current holder. Whoever won identify() is only the first to claim the
    identity, not a guarantee that every other run under it is blocked from
    ever reaching pay; nothing in triage's rules forces DUPLICATE_INVOICE
    (as opposed to DUPLICATE_OF_PAID_INVOICE) to hard_block, so a run that
    lost the identity race could still end up approved through review. What
    we actually need to know is whether money has already moved for this
    identity, from anyone."""
    return any(
        candidate.payment is not None and candidate.payment.status == "paid"
        for candidate in deps.repo.find_by_invoice_number(number, vendor)
    )


def _exact_duplicate_finding(holder_run_id: UUID) -> Finding:
    # No expected/actual: a fingerprint match/mismatch has no meaningful
    # "how different" to reason about, so there's nothing structured to add
    # beyond what the code and message already say.
    return Finding(
        code=FindingCode.EXACT_DUPLICATE,
        scope=InvoiceScope(),
        message=f"Same content as run {holder_run_id}; safe to treat as a duplicate.",
    )


def _conflicting_duplicate_finding(holder_run_id: UUID, already_paid: bool) -> Finding:
    code = FindingCode.DUPLICATE_OF_PAID_INVOICE if already_paid else FindingCode.DUPLICATE_INVOICE
    paid_note = " That run has already been paid." if already_paid else ""
    return Finding(
        code=code,
        scope=InvoiceScope(),
        message=(
            f"Same invoice identity as run {holder_run_id} but different content.{paid_note}"
        ),
    )


def make_deduplicate(deps: Deps) -> NodeFn:
    """Build the deduplicate node. Closes over Deps so the node itself stays testable."""

    def deduplicate(run: InvoiceRun) -> dict:
        assert run.extraction is not None, "deduplicate runs after extract"
        extraction = run.extraction

        log = deps.logger.bind(run.run_id)
        with log.stage("deduplicate") as fields:
            number, vendor = _identity_fields(extraction)
            fingerprint = _fingerprint(extraction)

            claim = deps.repo.identify(number, vendor, fingerprint, run.run_id)
            fields["acquired"] = claim.acquired

            finding: Finding | None = None
            if not claim.acquired:
                if claim.holder_fingerprint == fingerprint:
                    fields["outcome"] = "exact_duplicate"
                    finding = _exact_duplicate_finding(claim.holder_run_id)
                else:
                    already_paid = _identity_already_paid(deps, number, vendor)
                    fields["outcome"] = (
                        "duplicate_of_paid" if already_paid else "duplicate_invoice"
                    )
                    finding = _conflicting_duplicate_finding(claim.holder_run_id, already_paid)

        if finding is None:
            return {}

        report = ValidationReport(
            pass_number=1,
            findings=[finding],
            validated_at=datetime.now(UTC),
        )
        return {"validations": run.validations + [report]}

    return deduplicate
