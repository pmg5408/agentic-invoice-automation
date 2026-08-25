"""The field_path grammar. repair writes these strings, validate reads them.

The round-trip tests are the reason this module exists: they fail if either
half of the grammar changes without the other.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from invoice_agent.models import InvoiceScope, LineItem, LineScope
from invoice_agent.nodes import patch_paths
from invoice_agent.nodes.validate import make_validate

# Every scope shape validate emits with a field attached.
PATCHABLE_SCOPES = [
    InvoiceScope(field="invoice_number"),
    InvoiceScope(field="vendor_name"),
    InvoiceScope(field="total_amount"),
    InvoiceScope(field="subtotal"),
    InvoiceScope(field="issue_date"),
    InvoiceScope(field="due_date"),
    InvoiceScope(field="currency"),
    LineScope(line_index=0, field="canonical_item"),
    LineScope(line_index=2, field="quantity"),
]


class TestBuild:
    def test_a_line_field_becomes_an_indexed_path(self):
        scope = LineScope(line_index=2, field="canonical_item")
        assert patch_paths.build(scope) == "line_items[2].canonical_item"

    def test_an_invoice_field_is_its_own_name(self):
        assert patch_paths.build(InvoiceScope(field="due_date")) == "due_date"

    def test_a_whole_invoice_scope_names_no_field(self):
        # QTY_EXCEEDS_STOCK and the duplicate codes point at the invoice, not
        # at a field -- there is nothing for the model to patch.
        assert patch_paths.build(InvoiceScope()) is None

    def test_a_line_scope_without_a_field_names_no_field(self):
        assert patch_paths.build(LineScope(line_index=0)) is None


class TestParse:
    def test_an_indexed_path_yields_the_index_and_the_field(self):
        assert patch_paths.parse("line_items[2].quantity") == (2, "quantity")

    def test_a_bare_name_yields_no_index(self):
        assert patch_paths.parse("due_date") == (None, "due_date")

    def test_the_index_is_an_int_not_a_string(self):
        assert patch_paths.parse("line_items[0].quantity").line_index == 0

    @pytest.mark.parametrize(
        "garbage",
        [
            "vendor.name.first",
            "line_items[].quantity",
            "line_items[x].quantity",
            "line_items[0]",
            "line_items[0].",
            "",
            "two words",
        ],
    )
    def test_anything_outside_the_grammar_is_rejected(self, garbage):
        """A path nobody could have produced is a patch aimed at nothing."""
        assert patch_paths.parse(garbage) is None


class TestRoundTrip:
    @pytest.mark.parametrize("scope", PATCHABLE_SCOPES, ids=lambda s: str(s))
    def test_what_repair_writes_validate_can_read(self, scope):
        parsed = patch_paths.parse(patch_paths.build(scope))
        assert parsed.field == scope.field
        assert parsed.line_index == (scope.line_index if scope.kind == "line" else None)

    def test_every_scope_a_real_validation_produces_round_trips(
        self, deps, make_run, make_extraction
    ):
        """Catches a new finding shape whose scope the grammar cannot express,
        which would otherwise only surface as a repair that patches nothing."""
        run = make_run(
            extraction=make_extraction(
                invoice_number=None,
                vendor_name="Fraudster LLC",
                currency="EUR",
                issue_date=None,
                due_date=date(2020, 1, 1),
                subtotal=Decimal("999.00"),
                line_items=[
                    LineItem(raw_item_name="WidgetC", quantity=-5, unit_price=Decimal("1.00")),
                    LineItem(raw_item_name="FakeItem", quantity=None),
                    LineItem(raw_item_name="WidgetA", quantity=99, unit_price=Decimal("1.00")),
                ],
            )
        )
        report = make_validate(deps)(run)["validations"][-1]
        assert len(report.findings) > 8, "expected a broad spread of findings"

        for finding in report.findings:
            path = patch_paths.build(finding.scope)
            if path is None:
                assert finding.scope.field is None
                continue
            assert patch_paths.parse(path) is not None, f"{path!r} does not parse back"
