"""The graph shape and the two routing decisions from architecture.md.

Node bodies are not implemented yet; this pins the wiring around them so a
component session can trust the edges it is plugging into.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from invoice_agent.deps import Deps
from invoice_agent.graph import build_graph, make_route_after_triage, make_route_after_validate
from invoice_agent.models import (
    Finding,
    FindingCode,
    InvoiceScope,
    LineScope,
    PolicyGate,
    RepairAttempt,
    ValidationReport,
)

NODES = {
    "load", "extract", "deduplicate", "validate", "repair",
    "triage", "recommend", "critique", "decide", "pay",
}

# Nodes with a real implementation. Shrinks as each component lands, so the
# honesty check below stays meaningful for whatever is still a stub.
IMPLEMENTED = {"load", "extract", "deduplicate", "validate"}
STILL_STUBBED = NODES - IMPLEMENTED


def report(*findings: Finding, pass_number: int = 1) -> ValidationReport:
    return ValidationReport(
        pass_number=pass_number,
        findings=list(findings),
        validated_at=datetime.now(UTC),
    )


def finding(code: FindingCode) -> Finding:
    return Finding(
        code=code,
        scope=LineScope(line_index=0) if code.repairable else InvoiceScope(),
        message=str(code),
    )


def attempt(round_: int = 1) -> RepairAttempt:
    return RepairAttempt(round=round_)


class TestGraphShape:
    def test_all_ten_nodes_are_registered(self, deps: Deps):
        rendered = build_graph(deps).get_graph().draw_mermaid()
        for node in NODES:
            assert f"{node}(" in rendered or f"{node}({node})" in rendered

    def test_repair_loops_back_to_validate(self, deps: Deps):
        assert "repair --> validate" in build_graph(deps).get_graph().draw_mermaid()

    def test_pipeline_before_triage_is_unconditional(self, deps: Deps):
        rendered = build_graph(deps).get_graph().draw_mermaid()
        for edge in ("load --> extract", "extract --> deduplicate", "deduplicate --> validate"):
            assert edge in rendered

    def test_review_band_runs_recommend_then_critique_then_decide(self, deps: Deps):
        rendered = build_graph(deps).get_graph().draw_mermaid()
        assert "recommend --> critique" in rendered
        assert "critique --> decide" in rendered

    def test_every_path_ends_at_pay(self, deps: Deps):
        rendered = build_graph(deps).get_graph().draw_mermaid()
        assert "decide --> pay" in rendered
        assert "pay --> __end__" in rendered


class TestRouteAfterValidate:
    """The else-branch covers three situations; architecture.md spells them out."""

    def test_clean_invoice_goes_to_triage(self, deps, make_run):
        run = make_run(validations=[report()])
        assert make_route_after_validate(deps)(run) == "triage"

    def test_non_repairable_failure_goes_to_triage(self, deps, make_run):
        # INV-1002: QTY_EXCEEDS_STOCK is legible and true. Retrying it invites
        # the model to reread 20 as 2.
        run = make_run(validations=[report(finding(FindingCode.QTY_EXCEEDS_STOCK))])
        assert make_route_after_validate(deps)(run) == "triage"

    def test_repairable_finding_goes_to_repair(self, deps, make_run):
        # INV-1016: ITEM_NOT_FOUND on WidgetC.
        run = make_run(validations=[report(finding(FindingCode.ITEM_NOT_FOUND))])
        assert make_route_after_validate(deps)(run) == "repair"

    def test_repairable_finding_after_repair_goes_to_triage(self, deps, make_run):
        # Round cap spent: a repair that declined must not loop.
        run = make_run(
            validations=[report(finding(FindingCode.ITEM_NOT_FOUND))],
            repair=attempt(round_=1),
        )
        assert make_route_after_validate(deps)(run) == "triage"

    def test_round_cap_comes_from_config(self, deps, make_run):
        two_rounds = replace(
            deps, settings=deps.settings.model_copy(update={"repair_round_cap": 2})
        )
        run = make_run(
            validations=[report(finding(FindingCode.ITEM_NOT_FOUND))],
            repair=attempt(round_=1),
        )
        assert make_route_after_validate(two_rounds)(run) == "repair"

    def test_no_validation_yet_does_not_crash(self, deps, make_run):
        assert make_route_after_validate(deps)(make_run()) == "triage"

    @pytest.mark.parametrize(
        "code", [c for c in FindingCode if not c.repairable]
    )
    def test_no_non_repairable_code_ever_reaches_repair(self, deps, make_run, code):
        """Invariant 7: FindingCode.repairable is the only thing gating repair."""
        run = make_run(validations=[report(finding(code))])
        assert make_route_after_validate(deps)(run) == "triage"


class TestRouteAfterTriage:
    def test_review_band_goes_to_the_agents(self, deps, make_run):
        run = make_run(policy=PolicyGate(band="review", evaluated_at=datetime.now(UTC)))
        assert make_route_after_triage(deps)(run) == "recommend"

    def test_auto_approve_skips_the_agents(self, deps, make_run):
        run = make_run(policy=PolicyGate(band="auto_approve", evaluated_at=datetime.now(UTC)))
        assert make_route_after_triage(deps)(run) == "decide"

    def test_hard_block_skips_the_agents(self, deps, make_run):
        run = make_run(policy=PolicyGate(band="hard_block", evaluated_at=datetime.now(UTC)))
        assert make_route_after_triage(deps)(run) == "decide"


class TestStubsAreHonest:
    def test_every_node_raises_not_implemented(self, deps, make_run):
        """Nothing silently returns None while the pipeline is half-built."""
        import importlib

        for name in sorted(STILL_STUBBED):
            module = importlib.import_module(f"invoice_agent.nodes.{name}")
            node = getattr(module, f"make_{name}")(deps)
            with pytest.raises(NotImplementedError, match=name):
                node(make_run())
