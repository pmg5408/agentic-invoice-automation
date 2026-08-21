"""Node: repair.

Brief : docs/components.md section 5 -- Repair
LLM   : Yes -- one call, one round
In    : repairable findings, resolutions with candidates, run.extraction
Out   : RepairAttempt

Fires only when the latest report has repairable findings and the round cap
is not spent. The router already guarantees this.

Narrow deterministically before calling the model: pass the top-k fuzzy
candidates plus the original line, never the whole catalogue.

The model must be able to decline. patches=[] with the codes listed in
unrepaired is a valid, correct outcome. INV-1016's WidgetC at $350 sits
between WidgetA (250) and WidgetB (500) and matches neither -- forcing a
match there is the failure, not the empty result.

Patches target specific fields. Never rewrite the extraction. Control then
returns to validate, which produces a second report; both persist.

Provenance is not yours to set. deps.llm records model and prompt_version
on StageMetrics from the request; the schema you pass to complete() must
contain only fields the model itself authors.

The model returns RepairOutput (patches + unrepaired). You wrap it in a
RepairAttempt with round, triggered_by and candidates_offered -- those are
yours, not the model's.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_repair(deps: Deps) -> NodeFn:
    """Build the repair node. Closes over Deps so the node itself stays testable."""

    def repair(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("repair"):
            raise NotImplementedError(
                "repair is not implemented -- see docs/components.md section 5"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"repair": attempt}

    return repair
