"""Node: recommend.

Brief : docs/components.md section 7 -- Recommender
LLM   : Yes -- review band only
In    : extraction, validation, triage, vendor history from deps.repo.find_vendor()
Out   : ApprovalDraft

Structured output: recommendation, rationale, confidence, concerns, cited
evidence. Its inputs are facts (expected / actual from findings), not prose
summaries.

Vendor history must come from the vendors table via deps.repo.find_vendor().
If the prompt references history that is not in the input, the model will
invent it. An absent vendor is UNKNOWN_VENDOR, not a blank slate.

Advisory only. decide computes the outcome.

Provenance is not yours to set. deps.llm records model and prompt_version
on StageMetrics from the request; the schema you pass to complete() must
contain only fields the model itself authors.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_recommend(deps: Deps) -> NodeFn:
    """Build the recommend node. Closes over Deps so the node itself stays testable."""

    def recommend(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("recommend"):
            raise NotImplementedError(
                "recommend is not implemented -- see docs/components.md section 7"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"draft": draft}

    return recommend
