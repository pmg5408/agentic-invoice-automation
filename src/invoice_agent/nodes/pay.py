"""Node: pay.

Brief : docs/components.md section 8 -- Pay
LLM   : No
In    : run.decision and the idempotency key only
Out   : PaymentResult

Check the idempotency key first, write a payment-intent record, call
mock_payment, write the result. If the process dies between the call and the
write, a re-run must not pay twice.

idempotency_key = sha256(content_sha256 + invoice_number + vendor).

If this node needs to know why a decision was made, something upstream is
wrong.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_pay(deps: Deps) -> NodeFn:
    """Build the pay node. Closes over Deps so the node itself stays testable."""

    def pay(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("pay"):
            raise NotImplementedError(
                "pay is not implemented -- see docs/components.md section 8"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"payment": result, "status": "completed", "finished_at": ...}

    return pay
