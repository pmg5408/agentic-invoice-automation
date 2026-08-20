"""Guards on the hand-written parts of models.py.

Not a round-trip test over every model -- that tests Pydantic, not us. These
three things are our own code and each fails silently when wrong.
"""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from invoice_agent.models import (
    ApprovalDraft,
    Critique,
    ExtractedInvoice,
    Finding,
    FindingCode,
    InvoiceScope,
    LineItem,
    LineScope,
    RepairOutput,
    Scope,
    ValidationReport,
)

# Mirrors the table in docs/contracts.md. If contracts.md changes, this fails first.
EXPECTED_CODES = {
    "ITEM_NOT_FOUND": ("blocking", True),
    "QTY_EXCEEDS_STOCK": ("blocking", False),
    "ZERO_STOCK_ITEM": ("blocking", False),
    "NEGATIVE_QUANTITY": ("blocking", True),
    "MISSING_REQUIRED_FIELD": ("blocking", True),
    "UNPARSEABLE_DATE": ("warning", True),
    "TOTAL_MISMATCH": ("warning", False),
    "DUPLICATE_INVOICE": ("blocking", False),
    "DUPLICATE_OF_PAID_INVOICE": ("blocking", False),
    "PAST_DUE_DATE": ("info", False),
    "UNKNOWN_VENDOR": ("warning", False),
    "CURRENCY_MISMATCH": ("warning", False),
}


class TestFindingCode:
    """Seven of these twelve share a (severity, repairable) pair. Declared as a
    plain Enum with tuple values, Python aliases them and five members survive.
    """

    def test_all_twelve_members_exist_and_are_distinct(self):
        assert len(list(FindingCode)) == 12
        assert len({id(m) for m in FindingCode}) == 12

    def test_no_member_is_an_alias(self):
        # __members__ includes aliases; iteration does not. Equal length => none.
        assert len(FindingCode.__members__) == len(list(FindingCode))

    def test_value_is_the_member_name(self):
        for m in FindingCode:
            assert m.value == m.name

    @pytest.mark.parametrize(("name", "meta"), EXPECTED_CODES.items())
    def test_severity_and_repairability_match_contracts(self, name, meta):
        severity, repairable = meta
        member = FindingCode[name]
        assert member.severity == severity
        assert member.repairable is repairable

    def test_serializes_as_name_not_tuple(self):
        class M(BaseModel):
            code: FindingCode

        assert M(code=FindingCode.ZERO_STOCK_ITEM).model_dump_json() == '{"code":"ZERO_STOCK_ITEM"}'

    def test_round_trips_to_the_same_member(self):
        class M(BaseModel):
            code: FindingCode

        assert M.model_validate_json('{"code":"DUPLICATE_OF_PAID_INVOICE"}').code is (
            FindingCode.DUPLICATE_OF_PAID_INVOICE
        )


class TestDecimalMoney:
    """Money is Decimal, never float (invariant 2)."""

    def test_json_keeps_full_precision_as_string(self):
        item = LineItem(raw_item_name="WidgetA", quantity=3, unit_price=Decimal("3500.05"))
        assert '"3500.05"' in item.model_dump_json()

    def test_round_trip_does_not_go_through_float(self):
        original = Decimal("22562.80")
        inv = ExtractedInvoice(total_amount=original)
        restored = ExtractedInvoice.model_validate_json(inv.model_dump_json())
        assert restored.total_amount == original
        assert isinstance(restored.total_amount, Decimal)


class TestScopeUnion:
    def test_line_scope_round_trips_with_its_index(self):
        class M(BaseModel):
            scope: Scope

        restored = M.model_validate_json(
            '{"scope":{"kind":"line","line_index":2,"field":"quantity"}}'
        )
        assert isinstance(restored.scope, LineScope)
        assert restored.scope.line_index == 2

    def test_invoice_scope_round_trips(self):
        class M(BaseModel):
            scope: Scope

        restored = M.model_validate_json('{"scope":{"kind":"invoice","field":"total_amount"}}')
        assert isinstance(restored.scope, InvoiceScope)

    def test_unknown_kind_is_rejected(self):
        class M(BaseModel):
            scope: Scope

        with pytest.raises(ValidationError):
            M.model_validate({"scope": {"kind": "nonsense"}})


class TestArtifactsAreFrozen:
    """Stage outputs are append-only; raw_item_name is immutable (invariants 4, 6)."""

    def test_line_item_cannot_be_mutated(self):
        item = LineItem(raw_item_name="WidgetA (rush order)", quantity=4)
        with pytest.raises(ValidationError):
            item.raw_item_name = "WidgetA"

    def test_finding_cannot_be_mutated(self):
        f = _finding()
        with pytest.raises(ValidationError):
            f.message = "never mind"


def _finding(code: FindingCode = FindingCode.QTY_EXCEEDS_STOCK) -> Finding:
    return Finding(
        code=code,
        scope=LineScope(line_index=0),
        message="22 requested across 3 lines",
    )


class TestFindingDerivesFromCode:
    """severity and repairable have one home -- the code. A Finding never
    carries its own, so the two can never disagree."""

    def test_severity_comes_from_the_code(self):
        assert _finding().severity == "blocking"

    def test_repairable_comes_from_the_code(self):
        assert _finding().repairable is False

    def test_neither_can_be_supplied_at_construction(self):
        f = Finding(
            code=FindingCode.PAST_DUE_DATE,
            scope=InvoiceScope(),
            message="due 2025-01-01",
            severity="blocking",  # ignored: not a field
            repairable=True,
        )
        assert (f.severity, f.repairable) == ("info", False)

    def test_both_still_serialize(self):
        dumped = _finding().model_dump()
        assert dumped["severity"] == "blocking"
        assert dumped["repairable"] is False


class TestLLMSchemasAreLLMAuthoredOnly:
    """These four classes are handed to guided_json verbatim. A field the caller
    already knows -- which model ran, which prompt version, what time it is --
    becomes an instruction to the model to make one up. Provenance belongs on
    StageMetrics, which the client fills from the request.
    """

    @pytest.mark.parametrize(
        "payload", [ExtractedInvoice, RepairOutput, ApprovalDraft, Critique]
    )
    def test_no_caller_known_fields(self, payload):
        leaked = [
            name
            for name in payload.model_fields
            if name in {"model", "prompt_version"} or name.endswith("_at")
        ]
        assert leaked == [], f"{payload.__name__} would ask the LLM to invent {leaked}"


class TestValidationReportStatus:
    """Derived, so a report can never claim clean while carrying blocking findings."""

    def _report(self, *findings):
        return ValidationReport(
            pass_number=1, findings=list(findings), validated_at=datetime(2026, 1, 1)
        )

    def test_no_findings_is_clean(self):
        assert self._report().status == "clean"

    def test_warnings_only_is_flagged(self):
        assert self._report(_finding(FindingCode.UNKNOWN_VENDOR)).status == "flagged"

    def test_any_blocking_finding_is_blocked(self):
        report = self._report(
            _finding(FindingCode.UNKNOWN_VENDOR), _finding(FindingCode.ZERO_STOCK_ITEM)
        )
        assert report.status == "blocked"

    def test_status_cannot_be_supplied(self):
        report = ValidationReport(
            pass_number=1,
            findings=[_finding(FindingCode.ZERO_STOCK_ITEM)],
            validated_at=datetime(2026, 1, 1),
            status="clean",  # ignored: not a field
        )
        assert report.status == "blocked"

    def test_pass_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            ValidationReport(pass_number=0, validated_at=datetime(2026, 1, 1))
