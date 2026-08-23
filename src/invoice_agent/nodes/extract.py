"""Node: extract.

Brief : docs/components.md section 2 -- Extractor
LLM   : Yes -- one call
In    : run.source.raw_text
Out   : ExtractedInvoice

Structured output against the Pydantic schema, via deps.llm.complete().

Prompt layout matters for caching: stable content first (instructions,
schema, few-shot) marked cacheable=True, invoice text last and uncacheable.
No timestamps or run ids in the stable prefix.

LLMError from deps.llm means terminal needs_human_review -- the client has
already retried.

Provenance is not yours to set. deps.llm records model and prompt_version
on StageMetrics from the request; the schema you pass to complete() must
contain only fields the model itself authors.
"""

from __future__ import annotations

from typing import Any

from invoice_agent.config import StageLLMConfig
from invoice_agent.deps import Deps, NodeFn
from invoice_agent.llm.client import LLMError, LLMRequest, PromptSegment
from invoice_agent.models import ExtractedInvoice, InvoiceRun, StageMetrics
from invoice_agent.nodes.prompts import EXTRACT_V1

PROMPT_VERSION = "extract-v1"

# Every top-level field ExtractedInvoice can report as absent. Used to fill
# missing_fields when we short-circuit the model entirely (no text to read,
# or the call never came back) -- at that point everything is missing.
_ALL_FIELDS = [
    "invoice_number",
    "vendor_name",
    "currency",
    "total_amount",
    "subtotal",
    "issue_date",
    "due_date",
    "line_items",
    "revision_marker",
]


def _empty_extraction(note: str) -> ExtractedInvoice:
    """The "nothing to extract" record -- used when the model never ran at
    all, not when it ran and found an empty invoice. Absent field -> None
    still holds; we're just the one asserting it instead of the model."""
    return ExtractedInvoice(missing_fields=list(_ALL_FIELDS), extraction_notes=[note])


def _build_request(run: InvoiceRun, stage_cfg: StageLLMConfig) -> LLMRequest:
    return LLMRequest(
        stage="extract",
        prompt_version=PROMPT_VERSION,
        model=stage_cfg.model,
        temperature=stage_cfg.temperature,
        max_tokens=stage_cfg.max_tokens,
        segments=[
            PromptSegment(text=EXTRACT_V1, role="system", cacheable=True),
            PromptSegment(
                text=(
                    f"Invoice document (source format: {run.source.source_format}):\n\n"
                    f"{run.source.raw_text}"
                ),
                role="user",
                cacheable=False,
            ),
        ],
    )


def make_extract(deps: Deps) -> NodeFn:
    """Build the extract node. Closes over Deps so the node itself stays testable."""

    stage_cfg = deps.settings.stage("extract")

    def extract(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("extract") as fields:
            error: str | None = None

            if not run.source.text_extraction_ok:
                # Nothing to read -- calling the model would just burn a
                # request on empty or garbage text.
                extraction = _empty_extraction(
                    "source text was not extractable; no extraction attempted"
                )
                metrics = StageMetrics()
            else:
                request = _build_request(run, stage_cfg)
                try:
                    response = deps.llm.complete(request, ExtractedInvoice)
                    extraction = response.parsed
                    metrics = response.metrics
                except LLMError as exc:
                    extraction = _empty_extraction(
                        "LLM extraction failed after retries; see run error for details"
                    )
                    metrics = StageMetrics(model=stage_cfg.model, prompt_version=PROMPT_VERSION)
                    error = str(exc)

            fields["invoice_number"] = extraction.invoice_number
            fields["missing_fields"] = len(extraction.missing_fields)

        update: dict[str, Any] = {
            "extraction": extraction,
            "stage_metrics": {**run.stage_metrics, "extract": metrics},
        }
        if error is not None:
            update["error"] = error
        return update

    return extract
