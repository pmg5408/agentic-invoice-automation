"""Node: load.

Brief : docs/components.md section 1 -- Loader
LLM   : No
In    : run.source.source_path (the graph is seeded with a stub SourceDocument)
Out   : SourceDocument

Convert txt / json / csv / xml / pdf to text. Preserve structure: pretty-print
JSON, keep CSV tabular, pdfplumber.extract_text() for PDF.

The loader does not know what an invoice is. It interprets nothing.

invoice_1006.csv is key-value with repeated item/quantity/unit_price keys --
csv.DictReader will clobber them. invoice_1007.csv and 1015.csv are
row-per-line-item with a trailing totals block. A PDF with no text layer sets
text_extraction_ok=False, which routes straight to needs_human_review.
"""

from __future__ import annotations

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun


def make_load(deps: Deps) -> NodeFn:
    """Build the load node. Closes over Deps so the node itself stays testable."""

    def load(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("load"):
            raise NotImplementedError(
                "load is not implemented -- see docs/components.md section 1"
            )
        # Return a partial state update. Stage outputs are append-only, so lists
        # are rebuilt rather than mutated:
        #     return {"source": source_document}

    return load
