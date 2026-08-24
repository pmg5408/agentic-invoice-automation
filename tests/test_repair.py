"""Repair behaviour: one round, a closed allowlist, and the freedom to decline.

Uses stub_client, so the suite runs with no API key. A model instance is
serialized for you; a raw string drives the parse/validate/retry path.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from invoice_agent.llm.client import LLMClient
from invoice_agent.llm.stub import RaisingProvider, StubProvider
from invoice_agent.models import (
    FieldPatch,
    Finding,
    FindingCode,
    InvoiceScope,
    ItemResolution,
    LineItem,
    LineScope,
    RepairAttempt,
    RepairOutput,
    StageMetrics,
    ValidationReport,
)
from invoice_agent.nodes.repair import PROMPT_VERSION, make_repair
from invoice_agent.nodes.validate import make_validate

RUSH = "WidgetA (rush order)"


# -- helpers ---------------------------------------------------------------


def item_not_found(line_index: int, raw: str) -> Finding:
    return Finding(
        code=FindingCode.ITEM_NOT_FOUND,
        scope=LineScope(line_index=line_index, field="canonical_item"),
        expected="a catalogue item",
        actual=raw,
        message=f"Line {line_index}: {raw!r} did not match any catalogue item.",
    )


def unparseable_due_date() -> Finding:
    return Finding(
        code=FindingCode.UNPARSEABLE_DATE,
        scope=InvoiceScope(field="due_date"),
        message="No usable due_date was read from the document.",
    )


def qty_exceeds_stock() -> Finding:
    return Finding(
        code=FindingCode.QTY_EXCEEDS_STOCK,
        scope=InvoiceScope(),
        expected="15 in stock",
        actual="22 requested across 3 line(s)",
        message="WidgetA: 22 requested against 15 in stock.",
    )


def report(*findings: Finding, resolutions: list[ItemResolution] | None = None):
    return ValidationReport(
        pass_number=1,
        findings=list(findings),
        resolutions=resolutions or [],
        validated_at=datetime.now(UTC),
    )


def resolution(line_index: int, raw: str, *candidates: str) -> ItemResolution:
    return ItemResolution(
        line_index=line_index,
        raw_item_name=raw,
        canonical_item=None,
        method="unresolved",
        confidence=0.85,
        candidates_considered=list(candidates),
    )


def patch(field_path: str, new_value: str | None, old_value: str | None, **kw) -> FieldPatch:
    return FieldPatch(
        field_path=field_path,
        old_value=old_value,
        new_value=new_value,
        reason=kw.pop("reason", "the document says so"),
        confidence=kw.pop("confidence", 0.9),
    )


def with_llm(deps, *responses):
    provider = StubProvider(list(responses))
    return replace(deps, llm=LLMClient(provider, max_retries=1, backoff_base_s=0.0)), provider


@pytest.fixture
def rush_run(make_run, make_extraction):
    """INV-1005's shape: one line the deterministic layer could not resolve."""
    return make_run(
        extraction=make_extraction(
            line_items=[LineItem(raw_item_name=RUSH, quantity=4, unit_price=Decimal("250.00"))]
        ),
        validations=[
            report(
                item_not_found(0, RUSH),
                resolutions=[resolution(0, RUSH, "WidgetA", "WidgetB", "GadgetX")],
            )
        ],
    )


# -- the two outcomes that matter ------------------------------------------


class TestRepairSucceeds:
    def test_a_chosen_candidate_becomes_a_patch(self, deps, rush_run):
        node_deps, _ = with_llm(
            deps,
            RepairOutput(
                patches=[patch("line_items[0].canonical_item", "WidgetA", RUSH)],
                unrepaired=[],
            ),
        )
        attempt = make_repair(node_deps)(rush_run)["repair"]
        assert [p.new_value for p in attempt.patches] == ["WidgetA"]
        assert attempt.unrepaired == []
        assert attempt.round == 1
        assert attempt.triggered_by == [FindingCode.ITEM_NOT_FOUND]

    def test_validate_reads_the_patch_on_the_next_pass(self, deps, rush_run):
        """The loop end to end: repair patches, validate pass 2 resolves."""
        node_deps, _ = with_llm(
            deps,
            RepairOutput(
                patches=[
                    patch("line_items[0].canonical_item", "WidgetA", RUSH, confidence=0.88)
                ]
            ),
        )
        repaired = rush_run.model_copy(update=make_repair(node_deps)(rush_run))
        second = make_validate(deps)(repaired)["validations"][-1]

        assert second.pass_number == 2
        assert second.resolutions[0].canonical_item == "WidgetA"
        assert second.resolutions[0].method == "llm"
        assert FindingCode.ITEM_NOT_FOUND not in [f.code for f in second.findings]


class TestRepairDeclines:
    def test_an_empty_patch_list_is_a_valid_outcome(self, deps, make_run, make_extraction):
        # INV-1016: WidgetC at $350 sits between WidgetA (250) and WidgetB
        # (500) and matches neither. Declining is the correct answer.
        run = make_run(
            extraction=make_extraction(
                line_items=[
                    LineItem(raw_item_name="WidgetC", quantity=3, unit_price=Decimal("350.00"))
                ]
            ),
            validations=[
                report(
                    item_not_found(0, "WidgetC"),
                    resolutions=[resolution(0, "WidgetC", "WidgetA", "WidgetB")],
                )
            ],
        )
        node_deps, _ = with_llm(
            deps, RepairOutput(patches=[], unrepaired=[FindingCode.ITEM_NOT_FOUND])
        )
        attempt = make_repair(node_deps)(run)["repair"]
        assert attempt.patches == []
        assert attempt.unrepaired == [FindingCode.ITEM_NOT_FOUND]

    def test_the_finding_survives_into_the_second_pass(self, deps, make_run, make_extraction):
        run = make_run(
            extraction=make_extraction(
                line_items=[LineItem(raw_item_name="WidgetC", quantity=3)]
            ),
            validations=[report(item_not_found(0, "WidgetC"))],
        )
        node_deps, _ = with_llm(deps, RepairOutput(patches=[]))
        repaired = run.model_copy(update=make_repair(node_deps)(run))
        second = make_validate(deps)(repaired)["validations"][-1]
        assert FindingCode.ITEM_NOT_FOUND in [f.code for f in second.findings]


# -- what the model is allowed to see and to touch -------------------------


class TestNarrowing:
    def test_only_the_offered_candidates_are_shown(self, deps, rush_run):
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        make_repair(node_deps)(rush_run)
        case = provider.requests[0].segments[-1].text
        assert "choose from: WidgetA, WidgetB, GadgetX" in case
        assert "you may patch: line_items[0].canonical_item" in case

    def test_the_source_document_is_the_authority(self, deps, rush_run):
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        make_repair(node_deps)(rush_run)
        case = provider.requests[0].segments[-1].text
        assert rush_run.source.raw_text in case

    def test_a_non_repairable_finding_is_never_shown(self, deps, make_run, make_extraction):
        """Invariant 7. The model must never learn QTY_EXCEEDS_STOCK fired, or
        it will helpfully reread 20 as 2."""
        run = make_run(
            extraction=make_extraction(line_items=[LineItem(raw_item_name=RUSH, quantity=4)]),
            validations=[report(item_not_found(0, RUSH), qty_exceeds_stock())],
        )
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        attempt = make_repair(node_deps)(run)["repair"]
        case = provider.requests[0].segments[-1].text
        assert "QTY_EXCEEDS_STOCK" not in case
        assert FindingCode.QTY_EXCEEDS_STOCK not in attempt.triggered_by

    def test_the_instructions_are_a_cacheable_prefix(self, deps, rush_run):
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        make_repair(node_deps)(rush_run)
        request = provider.requests[0]
        assert request.cacheable_prefix_len == 1
        assert request.prompt_version == PROMPT_VERSION

    def test_the_schema_asks_only_for_what_the_model_authors(self, deps, rush_run):
        """RepairOutput, not RepairAttempt -- round and triggered_by are the
        caller's, and a schema that named them would order the model to invent
        them."""
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        make_repair(node_deps)(rush_run)
        properties = provider.schemas[0]["properties"]
        assert set(properties) == {"patches", "unrepaired"}


class TestScreening:
    def test_a_patch_outside_the_allowlist_is_dropped(self, deps, rush_run):
        # Nothing flagged total_amount. Editing it is the model fixing
        # something nobody questioned.
        node_deps, _ = with_llm(
            deps,
            RepairOutput(patches=[patch("total_amount", "999.00", "5000.00")]),
        )
        attempt = make_repair(node_deps)(rush_run)["repair"]
        assert attempt.patches == []
        assert attempt.unrepaired == [FindingCode.ITEM_NOT_FOUND]

    def test_a_patch_whose_old_value_disagrees_is_dropped(self, deps, rush_run):
        # Says it is replacing WidgetB; line 0 actually reads the rush order.
        node_deps, _ = with_llm(
            deps,
            RepairOutput(
                patches=[patch("line_items[0].canonical_item", "WidgetA", "WidgetB")]
            ),
        )
        attempt = make_repair(node_deps)(rush_run)["repair"]
        assert attempt.patches == []
        assert attempt.unrepaired == [FindingCode.ITEM_NOT_FOUND]

    def test_unrepaired_is_derived_not_taken_on_trust(self, deps, rush_run):
        """The model claims it fixed nothing while handing over a good patch.
        The record follows the patches, not the claim."""
        node_deps, _ = with_llm(
            deps,
            RepairOutput(
                patches=[patch("line_items[0].canonical_item", "WidgetA", RUSH)],
                unrepaired=[FindingCode.ITEM_NOT_FOUND],
            ),
        )
        attempt = make_repair(node_deps)(rush_run)["repair"]
        assert len(attempt.patches) == 1
        assert attempt.unrepaired == []

    def test_a_null_old_value_matches_a_field_that_was_null(
        self, deps, make_run, make_extraction
    ):
        run = make_run(
            extraction=make_extraction(due_date=None),
            validations=[report(unparseable_due_date())],
        )
        node_deps, _ = with_llm(
            deps, RepairOutput(patches=[patch("due_date", "2026-02-27", None)])
        )
        attempt = make_repair(node_deps)(run)["repair"]
        assert [p.field_path for p in attempt.patches] == ["due_date"]

    def test_one_finding_repaired_and_one_declined(self, deps, make_run, make_extraction):
        run = make_run(
            extraction=make_extraction(
                due_date=None, line_items=[LineItem(raw_item_name="WidgetC", quantity=3)]
            ),
            validations=[
                report(
                    item_not_found(0, "WidgetC"),
                    unparseable_due_date(),
                    resolutions=[resolution(0, "WidgetC", "WidgetA", "WidgetB")],
                )
            ],
        )
        node_deps, _ = with_llm(
            deps, RepairOutput(patches=[patch("due_date", "2026-02-27", None)])
        )
        attempt = make_repair(node_deps)(run)["repair"]
        assert [p.field_path for p in attempt.patches] == ["due_date"]
        assert attempt.unrepaired == [FindingCode.ITEM_NOT_FOUND]


# -- when it does not run, and when the model does not answer ---------------


class TestGuards:
    def test_a_clean_report_is_left_alone(self, deps, make_run):
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        assert make_repair(node_deps)(make_run(validations=[report()])) == {}
        assert provider.call_count == 0

    def test_a_report_with_only_non_repairable_findings_is_left_alone(self, deps, make_run):
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        run = make_run(validations=[report(qty_exceeds_stock())])
        assert make_repair(node_deps)(run) == {}
        assert provider.call_count == 0

    def test_the_round_cap_is_not_spent_twice(self, deps, rush_run):
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        spent = rush_run.model_copy(update={"repair": RepairAttempt(round=1)})
        assert make_repair(node_deps)(spent) == {}
        assert provider.call_count == 0

    def test_the_cap_comes_from_settings(self, deps, rush_run):
        node_deps, provider = with_llm(deps, RepairOutput(patches=[]))
        two_rounds = replace(
            node_deps, settings=node_deps.settings.model_copy(update={"repair_round_cap": 2})
        )
        spent = rush_run.model_copy(update={"repair": RepairAttempt(round=1)})
        assert make_repair(two_rounds)(spent)["repair"].round == 2
        assert provider.call_count == 1


class TestModelFailure:
    def test_an_unreachable_provider_leaves_everything_unrepaired(self, deps, rush_run):
        node_deps = replace(deps, llm=LLMClient(RaisingProvider(), max_retries=0))
        result = make_repair(node_deps)(rush_run)
        attempt = result["repair"]
        assert attempt.patches == []
        assert attempt.unrepaired == [FindingCode.ITEM_NOT_FOUND]
        # Not terminal: the findings survive and triage routes to a human.
        assert "error" not in result

    def test_output_that_never_validates_leaves_everything_unrepaired(self, deps, rush_run):
        node_deps, provider = with_llm(deps, '{"patches": ')
        attempt = make_repair(node_deps)(rush_run)["repair"]
        assert attempt.unrepaired == [FindingCode.ITEM_NOT_FOUND]
        assert provider.call_count == 2  # one retry, per settings

    def test_a_malformed_first_answer_is_retried(self, deps, rush_run):
        node_deps, provider = with_llm(
            deps,
            '{"patches": ',
            RepairOutput(patches=[patch("line_items[0].canonical_item", "WidgetA", RUSH)]),
        )
        attempt = make_repair(node_deps)(rush_run)["repair"]
        assert len(attempt.patches) == 1
        assert provider.call_count == 2


class TestStateUpdate:
    def test_stage_metrics_are_merged_not_replaced(self, deps, rush_run):
        node_deps, _ = with_llm(deps, RepairOutput(patches=[]))
        run = rush_run.model_copy(
            update={"stage_metrics": {"extract": StageMetrics(latency_ms=42)}}
        )
        metrics = make_repair(node_deps)(run)["stage_metrics"]
        assert set(metrics) == {"extract", "repair"}
        assert metrics["extract"].latency_ms == 42

    def test_provenance_comes_off_the_request(self, deps, rush_run):
        node_deps, _ = with_llm(deps, RepairOutput(patches=[]))
        metrics = make_repair(node_deps)(rush_run)["stage_metrics"]["repair"]
        assert metrics.model == deps.settings.stage("repair").model
        assert metrics.prompt_version == PROMPT_VERSION

    def test_the_extraction_is_never_rewritten(self, deps, rush_run):
        node_deps, _ = with_llm(
            deps,
            RepairOutput(patches=[patch("line_items[0].canonical_item", "WidgetA", RUSH)]),
        )
        result = make_repair(node_deps)(rush_run)
        assert "extraction" not in result
        assert rush_run.extraction.line_items[0].raw_item_name == RUSH
