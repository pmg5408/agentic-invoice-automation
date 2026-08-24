"""Node: validate.

Brief : docs/components.md section 4 -- Validator
LLM   : No
In    : run.extraction, run.repair, deps.inventory(), deps.repo
Out   : ValidationReport

Two phases: canonicalize, then check. Deterministic -- an LLM must never be
the component that decides whether 22 > 15.

Canonicalize: exact match, then fuzzy (rapidfuzz) against settings
fuzzy_accept_cutoff / fuzzy_margin. Record every resolution including
successes -- a fuzzy match is a decision worth showing.

Aggregation is not optional. INV-1013 lists WidgetA on three rows
(15 + 5 + 2 = 22) against stock of 15. Every row passes individually; the
invoice must not. Populate aggregated_quantities and check against that.

Do not price-check against the catalogue. INV-1013 has legitimate volume
discounts and INV-1014 is EUR. Total-vs-line-sum is the reliable check.

Validation observes and reports; triage interprets. status is a derived severity
roll-up, not a decision.

Duplicates belong to other components. deduplicate owns the identity findings
and writes them to run.identity; pay rechecks before money moves. A run that
wins identify() can never have a paid sibling -- pay sits far behind
deduplicate on every path -- so a check here would be vacuous for the winner
and a second copy of deduplicate's for the loser.

On pass 2 the extraction is read through run.repair's patches. Repair never
rewrites the extraction (invariant 4), so without that overlay pass 2 would be
byte-identical to pass 1 and the repair loop would do nothing. The patched
view is a local: run.extraction on disk stays exactly what the model said, and
the patches themselves record what changed.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from rapidfuzz import fuzz, process

from invoice_agent.config import Settings
from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import (
    ExtractedInvoice,
    Finding,
    FindingCode,
    InventoryItem,
    InvoiceRun,
    InvoiceScope,
    ItemResolution,
    LineItem,
    LineScope,
    RepairAttempt,
    StageMetrics,
    ValidationReport,
)

# Fields an invoice cannot be paid without. Dates are handled separately: the
# extractor collapses "absent" and "unreadable" into the same None, so
# validation cannot tell them apart and UNPARSEABLE_DATE covers both.
REQUIRED_FIELDS = ("invoice_number", "vendor_name", "total_amount")

DATE_FIELDS = ("issue_date", "due_date")


# --------------------------------------------------------------------------
# The repair overlay
# --------------------------------------------------------------------------

_LINE_PATH = re.compile(r"^line_items\[(\d+)\]\.(\w+)$")


def _as_str(value: str | None) -> str | None:
    return value if value and value.strip() else None


def _as_int(value: str | None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_decimal(value: str | None) -> Decimal | None:
    try:
        return Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError):
        return None


def _as_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# Every patch value is a string an LLM wrote, so each target names the coercion
# that has to succeed before it is applied. A patch that will not coerce is
# dropped and its finding stands -- a malformed patch must never be able to
# make validation pass.
_INVOICE_TARGETS = {
    "invoice_number": _as_str,
    "vendor_name": _as_str,
    "currency": _as_str,
    "total_amount": _as_decimal,
    "subtotal": _as_decimal,
    "issue_date": _as_date,
    "due_date": _as_date,
}

_LINE_TARGETS = {
    "quantity": _as_int,
    "unit_price": _as_decimal,
    "line_total": _as_decimal,
}


@dataclass(frozen=True)
class _EffectiveView:
    """The invoice as it reads after repair. ``invoice`` is a throwaway copy;
    ``forced_items`` carries the one patch kind the extraction cannot hold --
    canonical_item is a field of ItemResolution, which is validate's own
    output, so it lands as a resolution override rather than a copied field."""

    invoice: ExtractedInvoice
    forced_items: dict[int, tuple[str, float]]


def _effective_view(
    extraction: ExtractedInvoice,
    repair: RepairAttempt | None,
    inventory: dict[str, InventoryItem],
) -> _EffectiveView:
    if repair is None or not repair.patches:
        return _EffectiveView(extraction, {})

    invoice_updates: dict[str, Any] = {}
    line_updates: dict[int, dict[str, Any]] = {}
    forced: dict[int, tuple[str, float]] = {}

    for patch in repair.patches:
        match = _LINE_PATH.match(patch.field_path)
        if match is None:
            coerce = _INVOICE_TARGETS.get(patch.field_path)
            value = coerce(patch.new_value) if coerce else None
            if value is not None:
                invoice_updates[patch.field_path] = value
            continue

        index, field = int(match[1]), match[2]
        if not 0 <= index < len(extraction.line_items):
            continue
        if field == "canonical_item":
            # An item the catalogue does not have is not a resolution.
            if patch.new_value in inventory:
                forced[index] = (patch.new_value, patch.confidence)
            continue
        coerce = _LINE_TARGETS.get(field)
        value = coerce(patch.new_value) if coerce else None
        if value is not None:
            line_updates.setdefault(index, {})[field] = value

    if line_updates:
        invoice_updates["line_items"] = [
            item.model_copy(update=line_updates[i]) if i in line_updates else item
            for i, item in enumerate(extraction.line_items)
        ]

    invoice = extraction.model_copy(update=invoice_updates) if invoice_updates else extraction
    return _EffectiveView(invoice, forced)


# --------------------------------------------------------------------------
# Phase 1 -- canonicalize
# --------------------------------------------------------------------------


def _rank(name: str, inventory: dict[str, InventoryItem]) -> list[tuple[str, float]]:
    """Every catalogue item scored against this name, best first.

    WRatio rather than plain ratio: it is the only scorer that reads
    "WidgetA (rush order)" as a near-match for WidgetA while still scoring
    "WidgetC" as the coin flip between WidgetA and WidgetB that it is.
    """
    if not name:
        return []
    ranked = process.extract(
        name, list(inventory), scorer=fuzz.WRatio, processor=str.lower, limit=None
    )
    return [(choice, score) for choice, score, _ in ranked]


def _resolve(
    index: int,
    raw_item_name: str,
    inventory: dict[str, InventoryItem],
    folded: dict[str, str],
    settings: Settings,
    forced: dict[int, tuple[str, float]],
) -> ItemResolution:
    """Map one raw item name to a catalogue key.

    Three gates, all from settings. An accepted fuzzy match must clear
    fuzzy_accept_cutoff *and* beat the runner-up by fuzzy_margin. The margin is
    the load-bearing half: "WidgetC" scores 85.7 against both WidgetA and
    WidgetB, and a coin flip is exactly what must not be applied silently.
    """
    name = raw_item_name.strip()
    base = {"line_index": index, "raw_item_name": raw_item_name}

    canonical = folded.get(name.casefold())
    if canonical is not None:
        return ItemResolution(
            **base,
            canonical_item=canonical,
            method="exact",
            confidence=1.0,
            candidates_considered=[canonical],
        )

    ranked = _rank(name, inventory)
    # What repair gets offered, and what the record shows it chose from.
    candidates = [item for item, score in ranked if score >= settings.fuzzy_candidate_cutoff][
        : settings.fuzzy_top_k
    ]

    if index in forced:
        item, confidence = forced[index]
        return ItemResolution(
            **base,
            canonical_item=item,
            method="llm",
            confidence=confidence,
            candidates_considered=candidates,
        )

    best_score = ranked[0][1] if ranked else 0.0
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    clear_winner = (
        best_score >= settings.fuzzy_accept_cutoff
        and best_score - runner_up >= settings.fuzzy_margin
    )
    return ItemResolution(
        **base,
        canonical_item=ranked[0][0] if clear_winner else None,
        method="fuzzy" if clear_winner else "unresolved",
        confidence=best_score / 100,
        candidates_considered=candidates,
    )


def _aggregate(
    line_items: list[LineItem], resolutions: list[ItemResolution]
) -> tuple[dict[str, int], dict[str, int]]:
    """Quantity and line count per catalogue item. Stock is checked per *item*,
    not per line -- INV-1013 spreads 22 WidgetA across three rows that each
    pass on their own.

    Quantities are summed as written, negatives included. Clamping them would
    be editorialising; a negative quantity has its own blocking finding, and if
    repair corrects it pass 2 aggregates honestly.
    """
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for resolution, item in zip(resolutions, line_items, strict=True):
        canonical = resolution.canonical_item
        if canonical is None:
            continue
        totals[canonical] = totals.get(canonical, 0) + (item.quantity or 0)
        counts[canonical] = counts.get(canonical, 0) + 1
    return totals, counts


# --------------------------------------------------------------------------
# Phase 2 -- check
# --------------------------------------------------------------------------


def _line_findings(
    line_items: list[LineItem],
    resolutions: list[ItemResolution],
    inventory: dict[str, InventoryItem],
) -> list[Finding]:
    findings: list[Finding] = []
    for resolution, item in zip(resolutions, line_items, strict=True):
        index = resolution.line_index
        if resolution.canonical_item is None:
            findings.append(
                Finding(
                    code=FindingCode.ITEM_NOT_FOUND,
                    scope=LineScope(line_index=index, field="canonical_item"),
                    expected="a catalogue item",
                    actual=resolution.raw_item_name or "(no item name)",
                    message=(
                        f"Line {index}: {resolution.raw_item_name!r} did not match any "
                        f"catalogue item."
                    ),
                )
            )
        elif inventory[resolution.canonical_item].stock == 0:
            # Specific beats general: an out-of-stock item also "exceeds
            # stock", but ZERO_STOCK_ITEM is the one triage hard-blocks on,
            # and two findings for one fact is noise the agent reasons past.
            findings.append(
                Finding(
                    code=FindingCode.ZERO_STOCK_ITEM,
                    scope=LineScope(line_index=index, field="canonical_item"),
                    expected="stock available",
                    actual="0 in stock",
                    message=f"Line {index}: {resolution.canonical_item} is out of stock.",
                )
            )

        if item.quantity is None:
            findings.append(
                Finding(
                    code=FindingCode.MISSING_REQUIRED_FIELD,
                    scope=LineScope(line_index=index, field="quantity"),
                    expected="a quantity",
                    actual="null",
                    message=f"Line {index}: no quantity was read.",
                )
            )
        elif item.quantity < 0:
            findings.append(
                Finding(
                    code=FindingCode.NEGATIVE_QUANTITY,
                    scope=LineScope(line_index=index, field="quantity"),
                    expected="a quantity of 0 or more",
                    actual=str(item.quantity),
                    message=f"Line {index}: quantity is negative ({item.quantity}).",
                )
            )
    return findings


def _stock_findings(
    totals: dict[str, int], counts: dict[str, int], inventory: dict[str, InventoryItem]
) -> list[Finding]:
    findings: list[Finding] = []
    for item, quantity in totals.items():
        stock = inventory[item].stock
        if stock > 0 and quantity > stock:
            findings.append(
                Finding(
                    code=FindingCode.QTY_EXCEEDS_STOCK,
                    # Invoice scope, not line: no single row is wrong. INV-1013's
                    # WidgetA rows are 15, 5 and 2 against stock of 15, and
                    # blaming whichever row tipped it over is order-dependent.
                    scope=InvoiceScope(),
                    expected=f"{stock} in stock",
                    actual=f"{quantity} requested across {counts[item]} line(s)",
                    message=f"{item}: {quantity} requested against {stock} in stock.",
                )
            )
    return findings


def _line_sum(line_items: list[LineItem]) -> Decimal | None:
    """Sum of the line amounts, or None when any line cannot be priced.

    A partial sum would fire TOTAL_MISMATCH on arithmetic nobody can check, so
    an unpriceable line skips the check entirely.
    """
    if not line_items:
        return None
    total = Decimal("0")
    for item in line_items:
        if item.line_total is not None:
            total += item.line_total
        elif item.quantity is not None and item.unit_price is not None:
            total += Decimal(item.quantity) * item.unit_price
        else:
            return None
    return total


def _total_finding(invoice: ExtractedInvoice, tolerance: Decimal) -> Finding | None:
    """Line sum against subtotal, falling back to total_amount.

    Subtotal first because ExtractedInvoice has no tax field: comparing the
    line sum to a taxed total would flag INV-1005, 1013, 1014 and 1016, all of
    which are arithmetically fine.
    """
    line_sum = _line_sum(invoice.line_items)
    if line_sum is None:
        return None
    against = "subtotal" if invoice.subtotal is not None else "total_amount"
    reference = getattr(invoice, against)
    if reference is None or abs(line_sum - reference) <= tolerance:
        return None
    return Finding(
        code=FindingCode.TOTAL_MISMATCH,
        scope=InvoiceScope(field=against),
        expected=f"{reference} ({against})",
        actual=f"{line_sum} (sum of line amounts)",
        message=f"Line amounts sum to {line_sum}, but {against} is {reference}.",
    )


def _invoice_findings(
    invoice: ExtractedInvoice, deps: Deps, today: date
) -> list[Finding]:
    settings = deps.settings
    findings: list[Finding] = []

    for field in REQUIRED_FIELDS:
        if getattr(invoice, field) is None:
            findings.append(
                Finding(
                    code=FindingCode.MISSING_REQUIRED_FIELD,
                    scope=InvoiceScope(field=field),
                    expected=f"{field} present",
                    actual="null",
                    message=f"Required field {field} was not read from the document.",
                )
            )

    for field in DATE_FIELDS:
        if getattr(invoice, field) is None:
            findings.append(
                Finding(
                    code=FindingCode.UNPARSEABLE_DATE,
                    scope=InvoiceScope(field=field),
                    expected="a calendar date",
                    actual="null",
                    message=f"No usable {field} was read from the document.",
                )
            )

    total = _total_finding(invoice, settings.total_mismatch_tolerance)
    if total is not None:
        findings.append(total)

    if invoice.due_date is not None and invoice.due_date < today:
        findings.append(
            Finding(
                code=FindingCode.PAST_DUE_DATE,
                scope=InvoiceScope(field="due_date"),
                expected=f"a due date on or after {today.isoformat()}",
                actual=invoice.due_date.isoformat(),
                message=f"Due date {invoice.due_date.isoformat()} has already passed.",
            )
        )

    # A null vendor is already MISSING_REQUIRED_FIELD; reporting it unknown as
    # well says the same thing twice.
    if invoice.vendor_name is not None and deps.repo.find_vendor(invoice.vendor_name) is None:
        findings.append(
            Finding(
                code=FindingCode.UNKNOWN_VENDOR,
                scope=InvoiceScope(field="vendor_name"),
                expected="a vendor in the ledger",
                actual=invoice.vendor_name,
                message=f"{invoice.vendor_name!r} has no history in the vendor ledger.",
            )
        )

    if invoice.currency is not None and invoice.currency != settings.catalogue_currency:
        # Flagged, never price-checked: INV-1014 is EUR at prices that are
        # correct in EUR.
        findings.append(
            Finding(
                code=FindingCode.CURRENCY_MISMATCH,
                scope=InvoiceScope(field="currency"),
                expected=settings.catalogue_currency,
                actual=invoice.currency,
                message=(
                    f"Invoice is in {invoice.currency}; the catalogue is priced in "
                    f"{settings.catalogue_currency}."
                ),
            )
        )

    return findings


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------


def make_validate(deps: Deps) -> NodeFn:
    """Build the validate node. Closes over Deps so the node itself stays testable."""

    settings = deps.settings

    def validate(run: InvoiceRun) -> dict:
        assert run.extraction is not None, "validate runs after extract"

        log = deps.logger.bind(run.run_id)
        started = time.perf_counter()
        with log.stage("validate") as fields:
            inventory = deps.inventory()
            folded = {key.casefold(): key for key in inventory}
            view = _effective_view(run.extraction, run.repair, inventory)
            invoice = view.invoice

            resolutions = [
                _resolve(index, item.raw_item_name, inventory, folded, settings, view.forced_items)
                for index, item in enumerate(invoice.line_items)
            ]
            totals, counts = _aggregate(invoice.line_items, resolutions)

            findings = [
                *_line_findings(invoice.line_items, resolutions, inventory),
                *_stock_findings(totals, counts, inventory),
                *_invoice_findings(invoice, deps, datetime.now(UTC).date()),
            ]

            # 1 pre-repair, 2 post-repair -- driven by rounds spent, not by how
            # many reports happen to be on the run.
            pass_number = 1 if run.repair is None else run.repair.round + 1
            report = ValidationReport(
                pass_number=pass_number,
                resolutions=resolutions,
                findings=findings,
                aggregated_quantities=totals,
                validated_at=datetime.now(UTC),
            )

            fields["pass_number"] = pass_number
            fields["status"] = report.status
            fields["findings"] = len(findings)
            fields["unresolved"] = sum(1 for r in resolutions if r.canonical_item is None)

        metrics = StageMetrics(latency_ms=int((time.perf_counter() - started) * 1000))
        return {
            "validations": run.validations + [report],
            "stage_metrics": {**run.stage_metrics, "validate": metrics},
        }

    return validate
