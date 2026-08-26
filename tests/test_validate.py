"""Validator behaviour: canonicalization, the checks, and the repair overlay.

No LLM anywhere -- deps.llm is swapped for a provider that raises, so a node
that reached for it would fail loudly rather than silently pass.

Everything in the fixtures is dated 2026, so PAST_DUE_DATE (info) fires on any
invoice that does not opt into FUTURE explicitly. Tests assert on the codes
they care about rather than on the whole set.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from invoice_agent.llm.client import LLMClient
from invoice_agent.llm.stub import RaisingProvider
from invoice_agent.models import (
    ExtractedInvoice,
    FieldPatch,
    FindingCode,
    LineItem,
    RepairAttempt,
    RepairOutput,
    ValidationReport,
)
from invoice_agent.nodes.validate import make_validate

FUTURE = date(2099, 1, 1)


# -- helpers ---------------------------------------------------------------


@pytest.fixture
def validate_deps(deps):
    """Deps whose LLM client raises on any call. Validation is deterministic."""
    return replace(deps, llm=LLMClient(RaisingProvider(), max_retries=0))


def line(name: str, quantity: int | None = 1, unit_price: str | None = "250.00", **kw) -> LineItem:
    return LineItem(
        raw_item_name=name,
        quantity=quantity,
        unit_price=Decimal(unit_price) if unit_price is not None else None,
        **kw,
    )


def run_validate(deps, run) -> ValidationReport:
    return make_validate(deps)(run)["validations"][-1]


def codes(report: ValidationReport) -> list[FindingCode]:
    return [f.code for f in report.findings]


def only(report: ValidationReport, code: FindingCode):
    matches = [f for f in report.findings if f.code == code]
    assert len(matches) == 1, f"expected exactly one {code}, got {len(matches)}"
    return matches[0]


def patch(field_path: str, new_value: str | None, *, confidence: float = 0.9) -> FieldPatch:
    return FieldPatch(
        field_path=field_path, old_value=None, new_value=new_value,
        reason="test", confidence=confidence,
    )


def attempt(*patches: FieldPatch, round_: int = 1) -> RepairAttempt:
    return RepairAttempt(round=round_, output=RepairOutput(patches=list(patches)))


# -- canonicalization ------------------------------------------------------


class TestCanonicalization:
    """Three gates: exact, then fuzzy above the cutoff with a clear margin."""

    def test_exact_name_resolves_without_fuzzy_matching(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(line_items=[line("WidgetA")]))
        resolution = run_validate(validate_deps, run).resolutions[0]
        assert (resolution.canonical_item, resolution.method) == ("WidgetA", "exact")
        assert resolution.confidence == 1.0

    def test_exact_match_ignores_case_and_surrounding_space(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(line_items=[line("  widgeta ")]))
        resolution = run_validate(validate_deps, run).resolutions[0]
        assert (resolution.canonical_item, resolution.method) == ("WidgetA", "exact")
        # raw_item_name is immutable for the life of the run (invariant 4).
        assert resolution.raw_item_name == "  widgeta "

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("Widget A", "WidgetA"), ("Gadget X", "GadgetX"), ("Widgt A", "WidgetA")],
    )
    def test_a_clear_fuzzy_winner_is_accepted(
        self, validate_deps, make_run, make_extraction, raw, expected
    ):
        run = make_run(extraction=make_extraction(line_items=[line(raw)]))
        resolution = run_validate(validate_deps, run).resolutions[0]
        assert (resolution.canonical_item, resolution.method) == (expected, "fuzzy")

    def test_an_ambiguous_name_is_left_unresolved(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1016: WidgetC scores 85.7 against both WidgetA and WidgetB. A coin
        # flip is exactly what must not be applied silently.
        run = make_run(extraction=make_extraction(line_items=[line("WidgetC")]))
        resolution = run_validate(validate_deps, run).resolutions[0]
        assert resolution.canonical_item is None
        assert resolution.method == "unresolved"

    def test_a_near_match_without_margin_is_left_unresolved(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1005: "WidgetA (rush order)" clears the accept cutoff at 90 but
        # only beats WidgetB by 6.9. Whether "(rush order)" marks a different
        # SKU is a judgment call, and repair is where judgment calls go.
        run = make_run(extraction=make_extraction(line_items=[line("WidgetA (rush order)")]))
        resolution = run_validate(validate_deps, run).resolutions[0]
        assert resolution.canonical_item is None
        assert "WidgetA" in resolution.candidates_considered

    def test_nothing_close_resolves_to_nothing_and_offers_nothing(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(line_items=[line("Sprocket")]))
        resolution = run_validate(validate_deps, run).resolutions[0]
        assert resolution.canonical_item is None
        assert resolution.candidates_considered == []

    def test_candidates_are_capped_at_the_configured_top_k(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(line_items=[line("WidgetC")]))
        resolution = run_validate(validate_deps, run).resolutions[0]
        assert len(resolution.candidates_considered) <= validate_deps.settings.fuzzy_top_k
        assert resolution.candidates_considered[:2] == ["WidgetA", "WidgetB"]

    def test_every_line_is_recorded_including_the_successes(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA"), line("Widget A"), line("WidgetC")]
            )
        )
        report = run_validate(validate_deps, run)
        assert [r.method for r in report.resolutions] == ["exact", "fuzzy", "unresolved"]
        assert [r.line_index for r in report.resolutions] == [0, 1, 2]

    def test_the_margin_comes_from_settings(self, validate_deps, make_run, make_extraction):
        loose = replace(
            validate_deps,
            settings=validate_deps.settings.model_copy(update={"fuzzy_margin": 0.0}),
        )
        run = make_run(extraction=make_extraction(line_items=[line("WidgetA (rush order)")]))
        assert run_validate(loose, run).resolutions[0].canonical_item == "WidgetA"

    def test_the_accept_cutoff_comes_from_settings(
        self, validate_deps, make_run, make_extraction
    ):
        strict = replace(
            validate_deps,
            settings=validate_deps.settings.model_copy(update={"fuzzy_accept_cutoff": 99.0}),
        )
        run = make_run(extraction=make_extraction(line_items=[line("Widget A")]))
        assert run_validate(strict, run).resolutions[0].canonical_item is None


# -- aggregation and stock -------------------------------------------------


class TestAggregationAndStock:
    """Stock is checked per item, not per line."""

    def test_quantities_are_summed_per_item_across_lines(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1013's WidgetA rows.
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA", 15), line("WidgetB", 2), line("WidgetA", 5),
                            line("WidgetA", 2)]
            )
        )
        assert run_validate(validate_deps, run).aggregated_quantities == {
            "WidgetA": 22, "WidgetB": 2
        }

    def test_the_invoice_fails_although_every_line_passes_alone(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA", 15), line("WidgetA", 5), line("WidgetA", 2)]
            )
        )
        finding = only(run_validate(validate_deps, run), FindingCode.QTY_EXCEEDS_STOCK)
        # Invoice scope: no single row is wrong, so there is no line to blame.
        assert finding.scope.kind == "invoice"
        assert finding.expected == "15 in stock"
        assert finding.actual == "22 requested across 3 line(s)"

    def test_within_stock_is_not_flagged(self, validate_deps, make_run, make_extraction):
        run = make_run(extraction=make_extraction(line_items=[line("WidgetA", 15)]))
        assert FindingCode.QTY_EXCEEDS_STOCK not in codes(run_validate(validate_deps, run))

    def test_unresolved_lines_are_not_aggregated(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(line_items=[line("WidgetC", 99)]))
        assert run_validate(validate_deps, run).aggregated_quantities == {}

    def test_negative_quantities_are_summed_as_written(
        self, validate_deps, make_run, make_extraction
    ):
        # No clamping: NEGATIVE_QUANTITY reports the problem, and if repair
        # corrects it pass 2 aggregates honestly.
        run = make_run(
            extraction=make_extraction(line_items=[line("WidgetA", -5), line("WidgetA", 3)])
        )
        assert run_validate(validate_deps, run).aggregated_quantities == {"WidgetA": -2}

    def test_an_out_of_stock_item_reports_one_code_not_two(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1003: FakeItem has stock 0. ZERO_STOCK_ITEM is the specific code
        # and the one triage hard-blocks on; QTY_EXCEEDS_STOCK would say the
        # same thing less usefully.
        run = make_run(extraction=make_extraction(line_items=[line("FakeItem", 100)]))
        report = run_validate(validate_deps, run)
        assert FindingCode.ZERO_STOCK_ITEM in codes(report)
        assert FindingCode.QTY_EXCEEDS_STOCK not in codes(report)


# -- line-level checks -----------------------------------------------------


class TestLineChecks:
    def test_an_unresolved_item_is_item_not_found(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(line_items=[line("WidgetC")]))
        finding = only(run_validate(validate_deps, run), FindingCode.ITEM_NOT_FOUND)
        assert finding.scope.kind == "line"
        assert finding.scope.line_index == 0
        assert finding.repairable is True

    def test_a_negative_quantity_is_blocking_and_repairable(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1009.
        run = make_run(extraction=make_extraction(line_items=[line("WidgetA", -5)]))
        finding = only(run_validate(validate_deps, run), FindingCode.NEGATIVE_QUANTITY)
        assert finding.scope.field == "quantity"
        assert finding.severity == "blocking"
        assert finding.repairable is True

    def test_a_line_without_a_quantity_is_a_missing_required_field(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(line_items=[line("WidgetA", None)]))
        findings = [
            f for f in run_validate(validate_deps, run).findings
            if f.code == FindingCode.MISSING_REQUIRED_FIELD and f.scope.kind == "line"
        ]
        assert [f.scope.field for f in findings] == ["quantity"]


# -- invoice-level checks --------------------------------------------------


class TestInvoiceChecks:
    @pytest.mark.parametrize("field", ["invoice_number", "vendor_name", "total_amount"])
    def test_a_missing_required_field_is_reported(
        self, validate_deps, make_run, make_extraction, field
    ):
        run = make_run(extraction=make_extraction(**{field: None}))
        findings = [
            f for f in run_validate(validate_deps, run).findings
            if f.code == FindingCode.MISSING_REQUIRED_FIELD and f.scope.kind == "invoice"
        ]
        assert [f.scope.field for f in findings] == [field]

    @pytest.mark.parametrize("field", ["issue_date", "due_date"])
    def test_a_null_date_is_unparseable_not_missing(
        self, validate_deps, make_run, make_extraction, field
    ):
        # INV-1003's due date is "yesterday". The extractor collapses
        # unreadable and absent into the same None, so one code covers both --
        # and it is repairable, which is the response either case wants.
        run = make_run(extraction=make_extraction(**{field: None}))
        finding = only(run_validate(validate_deps, run), FindingCode.UNPARSEABLE_DATE)
        assert finding.scope.field == field
        assert finding.repairable is True

    def test_line_amounts_that_disagree_with_the_subtotal_are_flagged(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA", 2, "250.00")], subtotal=Decimal("999.00")
            )
        )
        finding = only(run_validate(validate_deps, run), FindingCode.TOTAL_MISMATCH)
        assert finding.expected == "999.00 (subtotal)"
        assert finding.actual == "500.00 (sum of line amounts)"

    def test_tax_between_the_subtotal_and_the_total_is_not_a_mismatch(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1005: lines sum to the subtotal, and the total carries 5% tax.
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA", 2, "250.00")],
                subtotal=Decimal("500.00"),
                total_amount=Decimal("525.00"),
            )
        )
        assert FindingCode.TOTAL_MISMATCH not in codes(run_validate(validate_deps, run))

    def test_without_a_subtotal_the_total_is_the_reference(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA", 2, "250.00")],
                subtotal=None,
                total_amount=Decimal("900.00"),
            )
        )
        finding = only(run_validate(validate_deps, run), FindingCode.TOTAL_MISMATCH)
        assert finding.scope.field == "total_amount"

    def test_a_line_that_cannot_be_priced_skips_the_arithmetic_check(
        self, validate_deps, make_run, make_extraction
    ):
        # A partial sum would flag arithmetic nobody can check.
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA", 2, "250.00"), line("WidgetB", None, None)],
                subtotal=Decimal("500.00"),
            )
        )
        assert FindingCode.TOTAL_MISMATCH not in codes(run_validate(validate_deps, run))

    def test_a_line_total_beats_quantity_times_price(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1013 has volume discounts; the stated amount is the truth.
        run = make_run(
            extraction=make_extraction(
                line_items=[line("WidgetA", 5, "240.00", line_total=Decimal("1200.00"))],
                subtotal=Decimal("1200.00"),
            )
        )
        assert FindingCode.TOTAL_MISMATCH not in codes(run_validate(validate_deps, run))

    def test_a_past_due_date_is_information_not_a_problem(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(due_date=date(2020, 1, 1)))
        assert only(run_validate(validate_deps, run), FindingCode.PAST_DUE_DATE).severity == "info"

    def test_a_future_due_date_is_not_flagged(self, validate_deps, make_run, make_extraction):
        run = make_run(extraction=make_extraction(due_date=FUTURE))
        assert FindingCode.PAST_DUE_DATE not in codes(run_validate(validate_deps, run))

    def test_a_vendor_with_no_ledger_history_is_flagged(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1003's "Fraudster LLC".
        run = make_run(extraction=make_extraction(vendor_name="Fraudster LLC"))
        finding = only(run_validate(validate_deps, run), FindingCode.UNKNOWN_VENDOR)
        assert finding.actual == "Fraudster LLC"

    def test_a_known_vendor_is_not_flagged(self, validate_deps, make_run, make_extraction):
        run = make_run(extraction=make_extraction(vendor_name="Widgets Inc."))
        assert FindingCode.UNKNOWN_VENDOR not in codes(run_validate(validate_deps, run))

    def test_a_null_vendor_is_reported_missing_but_not_also_unknown(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1009's vendor is "" in the source, so the extractor returns None.
        run = make_run(extraction=make_extraction(vendor_name=None))
        assert FindingCode.UNKNOWN_VENDOR not in codes(run_validate(validate_deps, run))

    def test_a_foreign_currency_is_flagged_but_never_price_checked(
        self, validate_deps, make_run, make_extraction
    ):
        # INV-1014 is EUR at prices that are correct in EUR.
        run = make_run(
            extraction=make_extraction(
                currency="EUR",
                line_items=[line("WidgetA", 4, "225.00")],
                subtotal=Decimal("900.00"),
            )
        )
        report = run_validate(validate_deps, run)
        assert only(report, FindingCode.CURRENCY_MISMATCH).severity == "warning"
        assert FindingCode.TOTAL_MISMATCH not in codes(report)

    def test_the_catalogue_currency_is_not_flagged(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(currency="USD"))
        assert FindingCode.CURRENCY_MISMATCH not in codes(run_validate(validate_deps, run))


class TestStatusRollup:
    def test_an_invoice_with_nothing_wrong_is_clean(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(extraction=make_extraction(due_date=FUTURE))
        report = run_validate(validate_deps, run)
        assert report.findings == []
        assert report.status == "clean"

    def test_a_blocking_finding_blocks(self, validate_deps, make_run, make_extraction):
        run = make_run(
            extraction=make_extraction(due_date=FUTURE, line_items=[line("WidgetC")])
        )
        assert run_validate(validate_deps, run).status == "blocked"


# -- duplicates belong elsewhere -------------------------------------------


class TestDuplicatesAreNotValidatesJob:
    def test_a_paid_sibling_under_the_same_identity_produces_no_finding(
        self, validate_deps, repo, make_run, make_extraction
    ):
        from invoice_agent.models import PaymentResult

        paid = make_run(payment=PaymentResult(idempotency_key="k", status="paid"))
        repo.save(paid)

        run = make_run(extraction=make_extraction(due_date=FUTURE))
        report = run_validate(validate_deps, run)
        assert report.findings == []


# -- the repair overlay ----------------------------------------------------


class TestRepairOverlay:
    """Pass 2 reads the extraction through repair's patches. Without this the
    second report would be byte-identical to the first."""

    def test_a_canonical_item_patch_resolves_the_line(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(
                due_date=FUTURE, line_items=[line("WidgetA (rush order)", 4)]
            ),
            repair=attempt(patch("line_items[0].canonical_item", "WidgetA", confidence=0.82)),
        )
        report = run_validate(validate_deps, run)
        resolution = report.resolutions[0]
        assert (resolution.canonical_item, resolution.method) == ("WidgetA", "llm")
        assert resolution.confidence == 0.82
        assert FindingCode.ITEM_NOT_FOUND not in codes(report)

    def test_a_patch_naming_an_item_the_catalogue_lacks_is_ignored(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(line_items=[line("WidgetC")]),
            repair=attempt(patch("line_items[0].canonical_item", "WidgetZ")),
        )
        report = run_validate(validate_deps, run)
        assert report.resolutions[0].canonical_item is None
        assert FindingCode.ITEM_NOT_FOUND in codes(report)

    def test_a_quantity_patch_changes_the_aggregate(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(due_date=FUTURE, line_items=[line("WidgetA", -5)]),
            repair=attempt(patch("line_items[0].quantity", "5")),
        )
        report = run_validate(validate_deps, run)
        assert report.aggregated_quantities == {"WidgetA": 5}
        assert FindingCode.NEGATIVE_QUANTITY not in codes(report)

    def test_a_scalar_patch_clears_its_finding(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(due_date=None),
            repair=attempt(patch("due_date", "2099-02-27")),
        )
        assert FindingCode.UNPARSEABLE_DATE not in codes(run_validate(validate_deps, run))

    @pytest.mark.parametrize(
        ("field_path", "new_value"),
        [
            ("line_items[0].quantity", "five"),
            ("due_date", "not-a-date"),
            ("total_amount", "lots"),
            ("line_items[9].quantity", "5"),
            ("vendor.name.first", "Widgets Inc."),
            ("due_date", None),
        ],
    )
    def test_a_patch_that_will_not_coerce_is_dropped(
        self, validate_deps, make_run, make_extraction, field_path, new_value
    ):
        """A malformed patch must never be able to make validation pass."""
        run = make_run(
            extraction=make_extraction(due_date=None, line_items=[line("WidgetA", -5)]),
            repair=attempt(patch(field_path, new_value)),
        )
        found = codes(run_validate(validate_deps, run))
        assert FindingCode.NEGATIVE_QUANTITY in found
        assert FindingCode.UNPARSEABLE_DATE in found

    def test_the_stored_extraction_is_never_touched(
        self, validate_deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(due_date=None, line_items=[line("WidgetA", -5)]),
            repair=attempt(
                patch("due_date", "2099-02-27"), patch("line_items[0].quantity", "5")
            ),
        )
        make_validate(validate_deps)(run)
        assert run.extraction.due_date is None
        assert run.extraction.line_items[0].quantity == -5


class TestEffectiveInvoice:
    """run.validations[-1] is the complete picture for a pass: the invoice
    downstream acts on, what is wrong with it, and which values an LLM wrote.
    No consumer should need a fallback to run.extraction."""

    def test_pass_one_carries_the_extraction_verbatim(
        self, validate_deps, make_run, make_extraction
    ):
        extraction = make_extraction()
        report = run_validate(validate_deps, make_run(extraction=extraction))
        assert report.effective_invoice == extraction
        assert report.applied_patches == []

    def test_a_patched_amount_is_readable_from_the_report(
        self, validate_deps, make_run, make_extraction
    ):
        # The case this field exists for: repair fills a missing total_amount,
        # pass 2 clears the finding, and pay must be able to read the amount
        # it is about to move -- run.extraction still says None forever.
        run = make_run(
            extraction=make_extraction(total_amount=None),
            repair=attempt(patch("total_amount", "1250.00")),
        )
        report = run_validate(validate_deps, run)
        assert report.effective_invoice.total_amount == Decimal("1250.00")
        assert run.extraction.total_amount is None

    def test_applied_patches_records_what_took_effect_and_only_that(
        self, validate_deps, make_run, make_extraction
    ):
        # One good patch, one the coercion rejects. The report is where a
        # reader tells them apart; run.repair holds both, looking identical.
        good = patch("due_date", "2099-02-27")
        rejected = patch("line_items[0].quantity", "five")
        run = make_run(
            extraction=make_extraction(due_date=None, line_items=[line("WidgetA", -5)]),
            repair=attempt(good, rejected),
        )
        report = run_validate(validate_deps, run)
        assert report.applied_patches == [good]
        assert report.effective_invoice.due_date == date(2099, 2, 27)

    def test_a_canonical_item_patch_counts_as_applied(
        self, validate_deps, make_run, make_extraction
    ):
        # It changes a resolution rather than the invoice, but it took effect
        # and downstream suspicion should cover it.
        applied = patch("line_items[0].canonical_item", "WidgetA")
        run = make_run(
            extraction=make_extraction(line_items=[line("WidgetA (rush order)", 4)]),
            repair=attempt(applied),
        )
        report = run_validate(validate_deps, run)
        assert report.applied_patches == [applied]
        # The invoice itself is untouched by this patch kind.
        assert report.effective_invoice.line_items[0].raw_item_name == "WidgetA (rush order)"


class TestPassNumbering:
    def test_the_first_pass_is_pass_one(self, validate_deps, make_run):
        assert run_validate(validate_deps, make_run()).pass_number == 1

    def test_the_pass_after_a_repair_round_is_pass_two(self, validate_deps, make_run):
        # Driven by rounds spent, not by how many reports are on the run.
        run = make_run(repair=attempt(round_=1))
        assert run_validate(validate_deps, run).pass_number == 2


class TestStateUpdate:
    def test_earlier_validations_are_kept_not_replaced(
        self, validate_deps, make_run, make_extraction
    ):
        earlier = ValidationReport(
            pass_number=1,
            effective_invoice=ExtractedInvoice(),
            validated_at=date.today().isoformat(),
        )
        run = make_run(validations=[earlier])
        result = make_validate(validate_deps)(run)
        assert result["validations"][0] is earlier
        assert len(result["validations"]) == 2

    def test_a_deterministic_stage_records_no_metrics(self, validate_deps, make_run):
        """Nothing to record: no model, no cost, and log.stage() already timed
        it. load and deduplicate do the same."""
        assert set(make_validate(validate_deps)(make_run())) == {"validations"}
