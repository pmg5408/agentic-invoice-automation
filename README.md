# Agentic Invoice Automation

Paying a vendor invoice usually means a person doing four jobs: read the
document, check it against what was actually ordered and stocked, get it
approved, and send the payment. This project automates that entire flow.
Give it a folder of invoices and each one comes out the other end paid,
rejected, or queued for a human — with the reasoning recorded at every step.

It's built for messy invoices — the kind vendors actually send — and for
being able to explain itself afterwards: every extraction, check, correction,
and decision is persisted, so you can always see why an invoice ended up
where it did. It's also deliberately frugal: a clean invoice costs one LLM
call, and only the genuinely ambiguous ones get the full multi-agent
treatment.

## What happens to an invoice

    load → extract (LLM) → deduplicate → validate ⇄ repair (LLM) → triage → draft approval (LLM) → critique it (LLM) → decide → pay

Every format — PDF, CSV, JSON, XML, plain text, an invoice pasted into an
email — is first normalized to plain text, then an LLM extracts the
structured fields — invoice number, vendor, line items, amounts, dates —
against a strict schema. If a field isn't in the document it comes back null;
the model isn't allowed to guess.

Before doing anything expensive, the run claims the invoice's identity
(number + vendor) in the database. If another run already holds it, this one
is a duplicate and gets flagged or skipped — which also makes it safe to
process a whole batch concurrently.

Validation is deterministic, plain code: match each line item to the
inventory catalogue (exact, then fuzzy), check requested quantities against
stock, verify the arithmetic, flag unknown vendors, past-due dates, and
negative quantities.

Some failures deserve a second look. If validation failed because the model
may have *misread* the document — an ambiguous item name, a date it couldn't
parse — one repair call gets to re-read the original text and propose
corrections, then validation runs again. The repair model is also free to
decline — "I can't tell what this item is" is a valid answer, and better
than a forced match.

Triage then applies the policy rules. Most invoices end here: clean and
under the approval threshold means auto-approve, and a few things (a
zero-stock item, a duplicate of something already paid) block outright. Only
the ambiguous middle goes to the approval agents — one drafts a
recommendation to approve, reject, or escalate, with its reasoning; a second
gets the same evidence and tries to poke holes in the draft. The final
decision is computed from the policy rules plus both opinions, and a hard
rule always wins over the agents. Payment goes through a mock API with an
idempotency key, so re-running an invoice can never pay it twice.

## Running it

```bash
make setup    # uv venv on Python 3.14 + editable install
make seed     # build invoices.db from the seed data
invoice-agent --invoice-path=data/invoices/invoice_1001.txt
invoice-agent --batch=data/invoices/
```

LLM calls go to NVIDIA NIM's free tier (`minimaxai/minimax-m3`) — put
`NVIDIA_API_KEY` in `.env`. The test suite runs fully stubbed, no key and no
network needed:

```bash
make test
```

## Stack

Python 3.14 · LangGraph for orchestration · Pydantic models end to end
(model output is schema-validated before it enters the pipeline) · SQLite
for the catalogue, vendor history, and run store · rapidfuzz for item
matching. Money is `Decimal` throughout, never float.

## Status

Work in progress. The pipeline through validate/repair is on `main`;
triage through payment is in review. Next up: a read-only UI over the run
store — every stage's output is already persisted, so the queue, the repair
diffs, and the draft-vs-critique disagreements are all renderable from data
that's already there.
