"""Node: decide.

Brief : docs/components.md section 8 -- Decide
LLM   : No
In    : run.policy, the draft and critique, latest ValidationReport
Out   : ApprovalDecision

Compute the outcome. A hard rule beats the agent; record overridden_by_rule.
Never copy the model's recommendation verbatim into decision.

The LLM advises; it never decides to spend money.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_decide(deps: Deps) -> NodeFn:
    """Build the decide node. Closes over Deps so the node itself stays testable."""

    def decide(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("decide"):
            raise NotImplementedError(
                "decide is not implemented -- see docs/components.md section 8"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"decision": decision}

    return decide
