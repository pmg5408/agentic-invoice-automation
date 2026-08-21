"""Node: validate.

Brief : docs/components.md section 4 -- Validator
LLM   : No
In    : run.extraction, deps.inventory(), deps.repo
Out   : ValidationReport

Two phases: canonicalize, then check. Deterministic -- an LLM must never be
the component that decides whether 22 > 15.

Canonicalize: exact match, then fuzzy (rapidfuzz) against settings
fuzzy_accept_cutoff / fuzzy_margin. Record every resolution including
successes -- a fuzzy match is a decision worth showing.

Aggregation is not optional. INV-1013 lists WidgetA on three rows
(15 + 5 + 2 = 22) against stock of 15. Every row passes individually; the
invoice must not. Populate aggregated_quantities and check against that.

Do not price-check against the catalogue. INV-1013 has legitimate volume
discounts and INV-1014 is EUR. Total-vs-line-sum is the reliable check.

Validation observes and reports; triage interprets. status is a derived severity
roll-up, not a decision.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_validate(deps: Deps) -> NodeFn:
    """Build the validate node. Closes over Deps so the node itself stays testable."""

    def validate(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("validate"):
            raise NotImplementedError(
                "validate is not implemented -- see docs/components.md section 4"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"validations": run.validations + [report]}

    return validate
