"""Node: repair.

Brief : docs/components.md section 5 -- Repair
LLM   : Yes -- one call, one round
In    : repairable findings, resolutions with candidates, run.extraction
Out   : RepairAttempt

Fires only when the latest report has repairable findings and the round cap
is not spent. The router already guarantees this; the node checks anyway.

Narrow deterministically before calling the model: pass the top-k fuzzy
candidates plus the original line, never the whole catalogue.

The model must be able to decline. patches=[] with the codes listed in
unrepaired is a valid, correct outcome. INV-1016's WidgetC at $350 sits
between WidgetA (250) and WidgetB (500) and matches neither -- forcing a
match there is the failure, not the empty result.

Three things keep this from manufacturing the success it is looking for:
only repairable findings are ever shown, so the model never learns that
QTY_EXCEEDS_STOCK fired and cannot help by rereading 20 as 2; the prompt
never says validation failed; and every returned patch is screened against
the allowlist the prompt was built from, because a prompt instruction is a
request, not a constraint.

Patches target specific fields. Never rewrite the extraction. Control then
returns to validate, which reads the extraction through these patches and
produces a second report; both persist.

Provenance is not yours to set. deps.llm records model and prompt_version
on StageMetrics from the request; the schema you pass to complete() must
contain only fields the model itself authors.

The model returns RepairOutput (patches + unrepaired). You wrap it in a
RepairAttempt with round and triggered_by -- those are yours, not the
model's.
"""

from __future__ import annotations

from typing import Any

from invoice_agent.config import StageLLMConfig
from invoice_agent.deps import Deps, NodeFn
from invoice_agent.llm.client import LLMError, LLMRequest, PromptSegment
from invoice_agent.models import (
    ExtractedInvoice,
    FieldPatch,
    Finding,
    FindingCode,
    InvoiceRun,
    RepairAttempt,
    RepairOutput,
    StageMetrics,
    ValidationReport,
)
from invoice_agent.nodes.prompts import REPAIR_V1

PROMPT_VERSION = "repair-v1"

# What the model is shown of the extraction, so it can see the field in context
# rather than in isolation.
_SUMMARY_FIELDS = (
    "invoice_number",
    "vendor_name",
    "currency",
    "total_amount",
    "subtotal",
    "issue_date",
    "due_date",
)


# --------------------------------------------------------------------------
# The allowlist
# --------------------------------------------------------------------------


def _target(finding: Finding) -> str | None:
    """The one path a finding licenses the model to patch, or None when the
    finding does not point at a field."""
    field = finding.scope.field
    if field is None:
        return None
    if finding.scope.kind == "line":
        return f"line_items[{finding.scope.line_index}].{field}"
    return field


def _allowlist(findings: list[Finding]) -> dict[str, Finding]:
    """Path -> the finding that licensed it. Built from the findings alone, so
    the set the prompt offers and the set enforced on the way back cannot
    drift apart."""
    targets: dict[str, Finding] = {}
    for finding in findings:
        path = _target(finding)
        if path is not None:
            targets.setdefault(path, finding)
    return targets


def _current_value(extraction: ExtractedInvoice, finding: Finding) -> object | None:
    """What was actually read for a finding's field -- what the prompt shows as
    "read as", and what a patch's old_value has to repeat."""
    field = finding.scope.field
    if finding.scope.kind != "line":
        return getattr(extraction, field, None)
    index = finding.scope.line_index
    if not 0 <= index < len(extraction.line_items):
        return None
    item = extraction.line_items[index]
    # canonical_item is not a field of the extraction: the value under
    # question is the raw name the document used.
    return item.raw_item_name if field == "canonical_item" else getattr(item, field, None)


def _codes(findings: list[Finding]) -> list[FindingCode]:
    """Distinct codes, in the order they were found."""
    seen: list[FindingCode] = []
    for finding in findings:
        if finding.code not in seen:
            seen.append(finding.code)
    return seen


# --------------------------------------------------------------------------
# The prompt payload
# --------------------------------------------------------------------------


def _text(value: object | None) -> str | None:
    return None if value is None else str(value)


def _shown(value: object | None) -> str:
    return "null" if value is None else f'"{value}"'


def _findings_block(
    targets: dict[str, Finding], extraction: ExtractedInvoice, report: ValidationReport
) -> str:
    candidates = {r.line_index: r.candidates_considered for r in report.resolutions}
    lines: list[str] = []
    for path, finding in targets.items():
        where = (
            f"line {finding.scope.line_index}" if finding.scope.kind == "line" else "invoice"
        )
        lines.append(f"  [{finding.code}] {where}, field {finding.scope.field}")
        lines.append(f"      read as: {_shown(_current_value(extraction, finding))}")
        lines.append(f"      you may patch: {path}")
        if finding.scope.kind == "line" and finding.scope.field == "canonical_item":
            offered = candidates.get(finding.scope.line_index) or []
            lines.append(
                "      choose from: "
                + (", ".join(offered) if offered else "(nothing in the catalogue is close)")
            )
    return "\n".join(lines)


def _extraction_block(extraction: ExtractedInvoice) -> str:
    lines = [f"  {field}: {_shown(getattr(extraction, field))}" for field in _SUMMARY_FIELDS]
    lines.append("  line_items:")
    for index, item in enumerate(extraction.line_items):
        lines.append(
            f"    [{index}] {item.raw_item_name!r} quantity={_shown(item.quantity)} "
            f"unit_price={_shown(item.unit_price)} line_total={_shown(item.line_total)}"
        )
    return "\n".join(lines)


def _build_request(
    run: InvoiceRun,
    report: ValidationReport,
    targets: dict[str, Finding],
    stage_cfg: StageLLMConfig,
) -> LLMRequest:
    extraction = run.extraction
    assert extraction is not None
    case = (
        "Findings you may address (only these):\n"
        f"{_findings_block(targets, extraction, report)}\n\n"
        f"Extraction as read:\n{_extraction_block(extraction)}\n\n"
        f"Source document (the authority):\n{run.source.raw_text}"
    )
    return LLMRequest(
        stage="repair",
        prompt_version=PROMPT_VERSION,
        model=stage_cfg.model,
        temperature=stage_cfg.temperature,
        max_tokens=stage_cfg.max_tokens,
        segments=[
            PromptSegment(text=REPAIR_V1, role="system", cacheable=True),
            PromptSegment(text=case, role="user", cacheable=False),
        ],
    )


# --------------------------------------------------------------------------
# Screening what came back
# --------------------------------------------------------------------------


def _old_value_agrees(patch: FieldPatch, current: object | None) -> bool:
    """A patch has to repeat the value it claims to be replacing.

    Cheap protection against the model editing a line it is not looking at:
    if it says it is replacing "WidgetB" and line 2 actually reads
    "WidgetA (rush order)", it has lost track of which finding it is on.
    """
    claimed = patch.old_value if patch.old_value not in ("null", "None", "") else None
    return claimed == _text(current)


def _screen(
    output: RepairOutput, targets: dict[str, Finding], extraction: ExtractedInvoice
) -> RepairOutput:
    """Keep only patches the findings licensed, then derive unrepaired.

    A patch to a field nobody questioned is the model editing something it was
    not asked about -- decision 5's failure mode in different clothing. It is
    dropped and its finding stands.

    unrepaired is recomputed rather than taken from the model, so it can never
    disagree with the patches beside it: a finding is repaired exactly when a
    patch for its path survived.
    """
    kept = [
        patch
        for patch in output.patches
        if patch.field_path in targets
        and _old_value_agrees(patch, _current_value(extraction, targets[patch.field_path]))
    ]
    patched = {patch.field_path for patch in kept}
    unrepaired = _codes(
        [finding for path, finding in targets.items() if path not in patched]
    )
    return RepairOutput(patches=kept, unrepaired=unrepaired)


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------


def make_repair(deps: Deps) -> NodeFn:
    """Build the repair node. Closes over Deps so the node itself stays testable."""

    stage_cfg = deps.settings.stage("repair")
    cap = deps.settings.repair_round_cap

    def repair(run: InvoiceRun) -> dict:
        assert run.extraction is not None, "repair runs after extract"

        log = deps.logger.bind(run.run_id)
        with log.stage("repair") as fields:
            report = run.validations[-1] if run.validations else None
            repairable = list(report.repairable) if report is not None else []
            rounds_used = 0 if run.repair is None else run.repair.round

            # The router already decided this, but a node that trusts its
            # caller for the round cap is one refactor away from a retry loop.
            if not repairable or rounds_used >= cap:
                fields["outcome"] = "skipped"
                return {}

            targets = _allowlist(repairable)
            if not targets:
                fields["outcome"] = "nothing_patchable"
                return {}

            triggered = _codes(list(targets.values()))
            fields["triggered_by"] = [str(code) for code in triggered]

            try:
                response = deps.llm.complete(
                    _build_request(run, report, targets, stage_cfg), RepairOutput
                )
                output = _screen(response.parsed, targets, run.extraction)
                metrics = response.metrics
            except LLMError as exc:
                # Not terminal. The findings survive intact, validate runs a
                # second pass, and triage routes the invoice to a human -- so
                # this is a repair that produced nothing, not a failed run.
                output = RepairOutput(patches=[], unrepaired=triggered)
                metrics = StageMetrics(model=stage_cfg.model, prompt_version=PROMPT_VERSION)
                fields["llm_error"] = str(exc)

            attempt = RepairAttempt(
                round=rounds_used + 1, triggered_by=triggered, output=output
            )
            fields["patches"] = len(output.patches)
            fields["unrepaired"] = [str(code) for code in output.unrepaired]

        update: dict[str, Any] = {
            "repair": attempt,
            "stage_metrics": {**run.stage_metrics, "repair": metrics},
        }
        return update

    return repair
