"""Node: triage.

Brief : docs/components.md section 6 -- Triage
LLM   : No
In    : run.extraction, latest ValidationReport
Out   : PolicyGate

Rules from config, evaluated in order. Every rule that fires produces a
RuleHit.

  hard_block   zero-stock item, duplicate of a paid invoice,
               text_extraction_ok=False
  auto_approve under threshold, no blocking findings, known vendor, sane dates
  review       everything else

Runs on every invoice. This node is what keeps most invoices away from the
LLM entirely.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_triage(deps: Deps) -> NodeFn:
    """Build the triage node. Closes over Deps so the node itself stays testable."""

    def triage(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("triage"):
            raise NotImplementedError(
                "triage is not implemented -- see docs/components.md section 6"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"policy": gate}

    return triage
