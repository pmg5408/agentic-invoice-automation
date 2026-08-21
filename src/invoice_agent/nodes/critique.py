"""Node: critique.

Brief : docs/components.md section 7 -- Critic
LLM   : Yes -- review band only
In    : the same evidence as recommend, plus the draft
Out   : Critique

Separate prompt, separate persona. Its job: what was missed, is the rationale
supported by the evidence, is a fraud signal being waved away, would this
survive an audit.

Given only the draft it could critique writing, not reasoning -- so it gets
the full evidence.

One round. Persist both artifacts even when the critic concurs.

At least one invoice must actually flip, or the critique is theatre. INV-1003
is the candidate: Fraudster LLC, a zero-stock item, $100K, an unparseable due
date, and pressure language. A recommender that fixates on the stock finding
and misses the fraud pattern is exactly what this should catch.

Provenance is not yours to set. deps.llm records model and prompt_version
on StageMetrics from the request; the schema you pass to complete() must
contain only fields the model itself authors.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_critique(deps: Deps) -> NodeFn:
    """Build the critique node. Closes over Deps so the node itself stays testable."""

    def critique(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("critique"):
            raise NotImplementedError(
                "critique is not implemented -- see docs/components.md section 7"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"critique": critique}

    return critique
