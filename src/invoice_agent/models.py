"""Shared data model. Transcribed from docs/contracts.md.

Frozen. Every component imports these types and nothing else crosses component
boundaries. If you need a shape change here, raise it -- do not edit in passing.

Money is Decimal, never float.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, computed_field

# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


# Money is Decimal everywhere, but Decimal's generated JSON Schema carries a
# negative-lookahead pattern, and constrained decoding engines cannot compile
# look-around -- the request 400s before the model ever runs. Advertise a plain
# string with an example instead. Validation is untouched: "1.2.3" and "-" are
# still rejected, and a string parses to an exact Decimal where a JSON number
# would arrive as a float.
Money = Annotated[
    Decimal,
    WithJsonSchema({"type": "string", "description": 'decimal amount, e.g. "1250.00"'}),
]

Severity = Literal["info", "warning", "blocking"]


class FindingCode(StrEnum):
    """Every way an invoice can fail.

    Each member carries its severity and repairability, so both live in exactly
    one place (invariant 7). ``Finding`` derives them rather than storing its
    own -- severity is a property of the kind of problem, and nothing escalates
    a single occurrence.

    ``repairable`` answers exactly one question: *could this be my misreading of
    the document?*  It is never *is this invoice bad?*  Quantity 20 against stock
    5 is legible and true -- retrying it invites the model to reread it as 2.

    The member value is the member name. Do not collapse this to a plain
    ``Enum`` with tuple values: Python aliases members that share a value, and
    seven of these twelve would silently vanish into each other.
    """

    def __new__(cls, code: str, severity: Severity, repairable: bool) -> FindingCode:
        obj = str.__new__(cls, code)
        obj._value_ = code
        obj.severity = severity
        obj.repairable = repairable
        return obj

    severity: Severity
    repairable: bool

    ITEM_NOT_FOUND = ("ITEM_NOT_FOUND", "blocking", True)
    QTY_EXCEEDS_STOCK = ("QTY_EXCEEDS_STOCK", "blocking", False)
    ZERO_STOCK_ITEM = ("ZERO_STOCK_ITEM", "blocking", False)
    NEGATIVE_QUANTITY = ("NEGATIVE_QUANTITY", "blocking", True)
    MISSING_REQUIRED_FIELD = ("MISSING_REQUIRED_FIELD", "blocking", True)
    UNPARSEABLE_DATE = ("UNPARSEABLE_DATE", "warning", True)
    TOTAL_MISMATCH = ("TOTAL_MISMATCH", "warning", False)
    DUPLICATE_INVOICE = ("DUPLICATE_INVOICE", "blocking", False)
    DUPLICATE_OF_PAID_INVOICE = ("DUPLICATE_OF_PAID_INVOICE", "blocking", False)
    # Same fingerprint as an already-claimed identity -- the two runs agree,
    # so this is informational, not a problem. Non-blocking and non-repairable
    # so it never triggers repair on its own; decisions.md 19 has triage treat
    # it as an override to skip recommend/critique too.
    EXACT_DUPLICATE = ("EXACT_DUPLICATE", "info", False)
    PAST_DUE_DATE = ("PAST_DUE_DATE", "info", False)
    UNKNOWN_VENDOR = ("UNKNOWN_VENDOR", "warning", False)
    CURRENCY_MISMATCH = ("CURRENCY_MISMATCH", "warning", False)



class Artifact(BaseModel):
    """Base for stage outputs. Frozen: stage outputs are append-only (invariant 6)."""

    model_config = ConfigDict(frozen=True)


class InvoiceScope(Artifact):
    kind: Literal["invoice"] = "invoice"
    field: str | None = None


class LineScope(Artifact):
    kind: Literal["line"] = "line"
    line_index: int
    field: str | None = None


Scope = Annotated[InvoiceScope | LineScope, Field(discriminator="kind")]


class Finding(Artifact):
    """One problem. Carries structured evidence, not just a message -- the
    approval agent reasons over ``expected`` / ``actual`` (invariant 5)."""

    code: FindingCode
    scope: Scope
    expected: str | None = None
    actual: str | None = None
    message: str

    # Derived from ``code``, not stored: severity is a property of the kind of
    # problem, and nothing escalates per occurrence. Computed rather than plain
    # properties so both still land in the run store and in approval prompts.
    @computed_field
    @property
    def severity(self) -> Severity:
        return self.code.severity

    @computed_field
    @property
    def repairable(self) -> bool:
        return self.code.repairable


# --------------------------------------------------------------------------
# Stage 0 -- loading
# --------------------------------------------------------------------------


class SourceDocument(Artifact):
    run_id: UUID
    source_path: str
    source_filename: str
    source_format: Literal["txt", "json", "csv", "xml", "pdf"]
    raw_text: str
    content_sha256: str
    text_extraction_ok: bool
    loaded_at: datetime


# --------------------------------------------------------------------------
# Stage 1 -- extraction
# --------------------------------------------------------------------------


class LineItem(Artifact):
    raw_item_name: str
    quantity: int | None = None
    unit_price: Money | None = None
    line_total: Money | None = None


class ExtractedInvoice(Artifact):
    """Exactly what the model produced, and nothing else.

    Every field here is authored by the LLM, because this class *is* the
    ``guided_json`` schema. Anything the caller already knows -- which model
    ran, which prompt version, what time it is -- must never appear, or the
    schema orders the model to invent it. That provenance lives in
    ``InvoiceRun.stage_metrics["extract"]``. Same rule for ``ApprovalDraft``,
    ``Critique`` and ``RepairOutput``.
    """

    invoice_number: str | None = None
    vendor_name: str | None = None
    currency: str | None = None
    total_amount: Money | None = None
    subtotal: Money | None = None
    issue_date: date | None = None
    due_date: date | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    revision_marker: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 1b -- deduplication
# --------------------------------------------------------------------------


class IdentityClaim(Artifact):
    """The result of deduplicate's attempt to claim (invoice_number, vendor).

    Its own slot on InvoiceRun rather than a ValidationReport appended to
    validations -- that field means "the output of a validation pass," and no
    validation ran here. finding is None on a clean claim; set to EXACT_DUPLICATE,
    DUPLICATE_INVOICE, or DUPLICATE_OF_PAID_INVOICE when the claim was lost.
    """

    acquired: bool
    holder_run_id: UUID
    fingerprint: str
    holder_fingerprint: str
    finding: Finding | None = None
    claimed_at: datetime


# --------------------------------------------------------------------------
# Stage 2 -- validation
# --------------------------------------------------------------------------


class ItemResolution(Artifact):
    line_index: int
    raw_item_name: str
    canonical_item: str | None = None
    method: Literal["exact", "fuzzy", "llm", "unresolved"]
    confidence: float
    candidates_considered: list[str] = Field(default_factory=list)


class ValidationReport(Artifact):
    """The complete picture for one pass. ``run.validations[-1]`` answers what
    the invoice is (``effective_invoice``), what is still wrong with it
    (``findings``), and which of its values an LLM wrote (``applied_patches``)
    -- downstream nodes read this report, not the raw extraction. The
    untouched originals stay on ``run.extraction`` and ``run.repair``."""

    # Bounded by settings.repair_round_cap, which is where that number lives.
    # Literal[1, 2] would pin the cap into the type as a second source of truth.
    pass_number: int = Field(ge=1)
    # The invoice this pass validated and downstream acts on. Pass 1: identical
    # to run.extraction. Pass 2: the extraction read through the applied
    # patches. Always populated, so no consumer needs a fallback branch.
    effective_invoice: ExtractedInvoice
    resolutions: list[ItemResolution] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    # The repair patches that actually took effect this pass. A patch in
    # run.repair but not here was rejected as malformed -- an unparseable
    # value, an unknown item, a path outside the grammar.
    applied_patches: list[FieldPatch] = Field(default_factory=list)
    aggregated_quantities: dict[str, int] = Field(default_factory=dict)
    validated_at: datetime

    # A roll-up of what was observed, not a decision -- triage decides. Derived
    # so a report can never claim "clean" while carrying blocking findings.
    @computed_field
    @property
    def status(self) -> Literal["clean", "flagged", "blocked"]:
        if not self.findings:
            return "clean"
        return "blocked" if self.blocking else "flagged"

    @property
    def invoice_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.scope.kind == "invoice"]

    @property
    def line_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.scope.kind == "line"]

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocking"]

    @property
    def repairable(self) -> list[Finding]:
        return [f for f in self.findings if f.repairable]


# --------------------------------------------------------------------------
# Stage 2b -- repair
# --------------------------------------------------------------------------


class FieldPatch(Artifact):
    field_path: str
    old_value: str | None = None
    new_value: str | None = None
    reason: str
    confidence: float


class RepairOutput(Artifact):
    """The model's half of a repair round -- the ``guided_json`` schema. An
    honest empty ``patches`` list is a valid outcome."""

    patches: list[FieldPatch] = Field(default_factory=list)
    unrepaired: list[FindingCode] = Field(default_factory=list)


class RepairAttempt(Artifact):
    """One repair round: what we asked and what came back. Patches state; never
    rewrites the extraction (invariant 4).

    Unlike the other three LLM artifacts this one is genuinely mixed -- round,
    triggered_by and candidates_offered are the caller's, so the model's half is
    nested rather than flattened.
    """

    round: int
    triggered_by: list[FindingCode] = Field(default_factory=list)
    output: RepairOutput = Field(default_factory=RepairOutput)

    @property
    def patches(self) -> list[FieldPatch]:
        return self.output.patches

    @property
    def unrepaired(self) -> list[FindingCode]:
        return self.output.unrepaired


# --------------------------------------------------------------------------
# Stage 3 -- approval
# --------------------------------------------------------------------------


class RuleHit(Artifact):
    rule_id: str
    description: str
    effect: Literal["force_reject", "force_review", "allow_auto_approve"]


class PolicyGate(Artifact):
    band: Literal["auto_approve", "review", "hard_block"]
    rule_hits: list[RuleHit] = Field(default_factory=list)
    evaluated_at: datetime


class ApprovalDraft(Artifact):
    """Advisory only. Never copied into ApprovalDecision.decision."""

    recommendation: Literal["approve", "reject", "escalate"]
    rationale: str
    confidence: float
    concerns: list[str] = Field(default_factory=list)
    evidence_cited: list[FindingCode] = Field(default_factory=list)


class Critique(Artifact):
    verdict: Literal["concur", "revise", "reject_reasoning"]
    issues: list[str] = Field(default_factory=list)
    unaddressed_evidence: list[FindingCode] = Field(default_factory=list)
    recommended_change: str | None = None


class ApprovalDecision(Artifact):
    """Computed by code. A hard rule beats the agent; the LLM never decides to
    spend money (invariant 8)."""

    decision: Literal["approved", "rejected", "needs_human_review"]
    decided_by: Literal["rule", "agent", "agent_after_critique"]
    rationale: str
    overridden_by_rule: str | None = None
    draft: ApprovalDraft | None = None
    critique: Critique | None = None
    decided_at: datetime


# --------------------------------------------------------------------------
# Stage 4 -- payment
# --------------------------------------------------------------------------


class PaymentResult(Artifact):
    idempotency_key: str
    status: Literal["paid", "skipped_duplicate", "skipped_not_approved", "failed"]
    amount: Money | None = None
    currency: str | None = None
    vendor: str | None = None
    provider_response: dict | None = None
    paid_at: datetime | None = None


# --------------------------------------------------------------------------
# The run record
# --------------------------------------------------------------------------


class StageMetrics(Artifact):
    """What one stage did: which model, how much it cost, when. Keyed by stage
    name on the run, so this is where an LLM stage's provenance lives -- see
    the note on ``ExtractedInvoice``. ``LLMClient`` fills model/prompt_version
    from the request; deterministic stages leave them None."""

    model: str | None = None
    prompt_version: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0
    cost_usd: Money = Decimal("0")


class InvoiceRun(BaseModel):
    """One invoice, end to end. Both the LangGraph state object and the
    persisted row. Deliberately not frozen -- the graph replaces it per node.
    Its nested artifacts are frozen, so history cannot be rewritten."""

    run_id: UUID
    status: Literal["running", "completed", "failed"]
    current_stage: str
    source: SourceDocument
    extraction: ExtractedInvoice | None = None
    identity: IdentityClaim | None = None
    validations: list[ValidationReport] = Field(default_factory=list)
    repair: RepairAttempt | None = None
    policy: PolicyGate | None = None
    # Carried between recommend -> critique -> decide. ApprovalDecision embeds
    # both again as the durable audit record; these are the state slots that
    # get them there.
    draft: ApprovalDraft | None = None
    critique: Critique | None = None
    decision: ApprovalDecision | None = None
    payment: PaymentResult | None = None
    error: str | None = None
    stage_metrics: dict[str, StageMetrics] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime | None = None


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------


class InventoryItem(Artifact):
    """A row of the inventory catalogue. ``unit_price`` is stored as TEXT in
    SQLite and parsed to Decimal here -- a REAL column would make money float."""

    item: str
    stock: int
    unit_price: Money | None = None
    category: str | None = None


class Vendor(Artifact):
    """A row of the vendor ledger.

    The approval agent reasons over this history. If a vendor is absent the
    correct output is UNKNOWN_VENDOR -- never an invented history.
    """

    name: str
    first_seen: date
    invoice_count: int
    status: Literal["active", "watchlist", "inactive"]


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------


class Identification(Artifact):
    """Result of claiming an (invoice_number, vendor) identity.

    ``acquired`` is False when another run got there first; compare
    ``holder_fingerprint`` to decide skip-vs-flag.
    """

    acquired: bool
    holder_run_id: UUID
    holder_fingerprint: str


class InvoiceRepository(Protocol):
    """The only way a component reads another component's output (invariant 10)."""

    def save(self, run: InvoiceRun) -> None: ...

    def find_by_content_hash(self, sha256: str) -> InvoiceRun | None: ...

    def find_by_invoice_number(self, number: str, vendor: str) -> list[InvoiceRun]: ...

    def load_inventory(self) -> dict[str, InventoryItem]: ...

    def find_vendor(self, name: str) -> Vendor | None: ...

    def identify(
        self, number: str, vendor: str, fingerprint: str, run_id: UUID
    ) -> Identification:
        """Atomically claim an invoice identity. Backed by a UNIQUE constraint."""
        ...
