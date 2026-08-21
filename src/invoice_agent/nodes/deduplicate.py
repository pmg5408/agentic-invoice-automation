"""Node: deduplicate.

Brief : docs/components.md section 3 -- Deduplicate
LLM   : No
In    : run.extraction
Out   : identity row; on loss, a DUPLICATE_* finding

Build the fingerprint (hash of line items + quantities + total), call
deps.repo.identify(), and handle the loss.

Equal fingerprint -> skipped_duplicate, no human. Different -> DUPLICATE_INVOICE,
or DUPLICATE_OF_PAID_INVOICE when the holder is already paid.

The loser never waits for the winner. It only needs to know a holder exists
and whether the content agrees.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_deduplicate(deps: Deps) -> NodeFn:
    """Build the deduplicate node. Closes over Deps so the node itself stays testable."""

    def deduplicate(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("deduplicate"):
            raise NotImplementedError(
                "deduplicate is not implemented -- see docs/components.md section 3"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"validations": run.validations + [report]}  # only when a finding is raised

    return deduplicate
