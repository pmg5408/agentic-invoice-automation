"""The pipeline, wired. Transcribed from docs/architecture.md.

    load -> extract -> deduplicate -> validate -> triage -+-> decide -> pay
                                        ^  |              |  (auto_approve / hard_block)
                                        |  v              |
                                       repair             +-> recommend -> critique -> decide -> pay
                                    (max 1 round)                         (review band)

triage runs once on every invoice and then branches. Everything before it is
unconditional; only the branch after it is optional.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from invoice_agent.deps import Deps
from invoice_agent.models import InvoiceRun
from invoice_agent.nodes.critique import make_critique
from invoice_agent.nodes.decide import make_decide
from invoice_agent.nodes.deduplicate import make_deduplicate
from invoice_agent.nodes.extract import make_extract
from invoice_agent.nodes.load import make_load
from invoice_agent.nodes.pay import make_pay
from invoice_agent.nodes.recommend import make_recommend
from invoice_agent.nodes.repair import make_repair
from invoice_agent.nodes.triage import make_triage
from invoice_agent.nodes.validate import make_validate


def make_route_after_validate(deps: Deps):
    """Repair, or move on.

    The round cap is what stops a retry loop from eventually manufacturing the
    success it is looking for. A failing invoice never stalls and never loops:
    it gets at most one repair attempt, then goes to triage regardless of
    whether the repair succeeded.
    """

    cap = deps.settings.repair_round_cap

    def route_after_validate(run: InvoiceRun) -> Literal["repair", "triage"]:
        if not run.validations:
            return "triage"
        rounds_used = 0 if run.repair is None else run.repair.round
        if run.validations[-1].repairable and rounds_used < cap:
            return "repair"
        return "triage"

    return route_after_validate


def make_route_after_triage(deps: Deps):
    """Only the review band pays for the approval agents."""

    def route_after_triage(run: InvoiceRun) -> Literal["recommend", "decide"]:
        if run.policy is not None and run.policy.band == "review":
            return "recommend"
        return "decide"

    return route_after_triage


def build_graph(deps: Deps):
    """Compile the pipeline. One invoice = one run of this graph."""
    graph = StateGraph(InvoiceRun)

    graph.add_node("load", make_load(deps))
    graph.add_node("extract", make_extract(deps))
    graph.add_node("deduplicate", make_deduplicate(deps))
    graph.add_node("validate", make_validate(deps))
    graph.add_node("repair", make_repair(deps))
    graph.add_node("triage", make_triage(deps))
    graph.add_node("recommend", make_recommend(deps))
    graph.add_node("critique", make_critique(deps))
    graph.add_node("decide", make_decide(deps))
    graph.add_node("pay", make_pay(deps))

    graph.add_edge(START, "load")
    graph.add_edge("load", "extract")
    graph.add_edge("extract", "deduplicate")
    graph.add_edge("deduplicate", "validate")

    graph.add_conditional_edges(
        "validate",
        make_route_after_validate(deps),
        {"repair": "repair", "triage": "triage"},
    )
    graph.add_edge("repair", "validate")

    graph.add_conditional_edges(
        "triage",
        make_route_after_triage(deps),
        {"recommend": "recommend", "decide": "decide"},
    )
    graph.add_edge("recommend", "critique")
    graph.add_edge("critique", "decide")

    graph.add_edge("decide", "pay")
    graph.add_edge("pay", END)

    return graph.compile()
