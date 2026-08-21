"""Node: extract.

Brief : docs/components.md section 2 -- Extractor
LLM   : Yes -- one call
In    : run.source.raw_text
Out   : ExtractedInvoice

Structured output against the Pydantic schema, via deps.llm.complete().

Prompt layout matters for caching: stable content first (instructions,
schema, few-shot) marked cacheable=True, invoice text last and uncacheable.
No timestamps or run ids in the stable prefix.

Absent field -> None plus a missing_fields entry. Never guess: a hallucinated
amount is the worst bug this system can have. Empty string is not a value
(INV-1009 vendor is ""). Unparseable date -> None (INV-1003 due date is
"yesterday"). Normalize invoice numbers: 1002 and INV 1012 -> INV-1002,
INV-1012. Keep raw_item_name verbatim. Capture revision_marker and currency.

LLMError from deps.llm means terminal needs_human_review -- the client has
already retried.

Provenance is not yours to set. deps.llm records model and prompt_version
on StageMetrics from the request; the schema you pass to complete() must
contain only fields the model itself authors.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_extract(deps: Deps) -> NodeFn:
    """Build the extract node. Closes over Deps so the node itself stays testable."""

    def extract(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("extract"):
            raise NotImplementedError(
                "extract is not implemented -- see docs/components.md section 2"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"extraction": extracted}

    return extract
