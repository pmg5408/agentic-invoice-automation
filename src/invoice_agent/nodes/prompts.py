"""Prompt content for the LLM-calling nodes (extract, repair, recommend, critique).

Kept separate from each node's request-building/orchestration logic so the two
can change independently and a prompt rewrite is a small, readable diff. One
named constant per stage per version, so the name tracks the same version
string the node sends as LLMRequest.prompt_version -- EXTRACT_V1 backs
prompt_version="extract-v1", and a rewrite lands as EXTRACT_V2 rather than an
edit-in-place that erases what the old version said.
"""

from __future__ import annotations

EXTRACT_V1 = """You extract structured invoice data from a single business document.
The document may be plain text, a key-value list, JSON, XML, a CSV table, a fixed-width
ASCII table, or an invoice pasted into an email body. Extract only the invoice fields --
ignore email headers, greetings, and sign-offs.

Rules:
- If a field is not present in the document, its value is null and its field name goes
  in missing_fields. Never guess or infer a value that is not written down.
- An empty string in the source (e.g. a blank vendor name) is not a value -- treat it the
  same as absent: null, plus a missing_fields entry.
- A due date or issue date that is not a real calendar date (e.g. "yesterday") is
  unparseable -- null, plus a missing_fields entry. Do not resolve relative dates.
- Normalize invoice numbers to the form "INV-1234": a bare number like "1002" becomes
  "INV-1002"; "INV 1012" (space, no dash) becomes "INV-1012"; "INV-1001" is already correct.
- Copy each line item's name exactly as written in the item/description field, no more
  and no less. If a note is written directly inside that same field with no column of
  its own (e.g. "WidgetA (rush order)" as one continuous piece of text), keep it exactly
  as written. If the document instead has a separate "Notes" column or field for that
  line -- text that comes after the amount, or a distinct field like "note" -- do not
  merge it into the item name. A row "WidgetA  5  $240.00  $1,200.00  Volume discount"
  under headers "Item Qty Unit Price Amount Notes" has raw_item_name "WidgetA", not
  "WidgetA (Volume discount)" -- the note is a separate column, not part of the name.
- If the document states a revision or version marker (a "revision" field, "Rev 2",
  "R1", "Revised", etc.), copy it into revision_marker verbatim. Otherwise null.
- Currency: if the document states a currency explicitly (a code like "EUR" or a symbol
  other than "$"), use that. If it only uses "$" with nothing else stated, use "USD".
  If there is truly no currency signal at all, use null plus a missing_fields entry.
- This data sometimes has OCR-style corruption: a capital letter O standing in for the
  digit 0. For example "26-Jan-2O26" means 2026, and "$3,500.O0" means 3500.00. Read
  through this corruption to the intended number or date.
- Note anything that required judgment -- an inferred currency, a corrected digit, an
  ambiguous item name -- as a short entry in extraction_notes. This is optional context
  for a human reviewer, not a place to guess at missing data.

Example:

Document:
INVOICE

Vendor:
Inv #: 2044
Date: 2026-03-01
Due: yesterday

Items:
  GadgetY   qty: 4   unit price: $75.00

Total: $3OO.00

Correct output:
{"invoice_number": "INV-2044", "vendor_name": null, "currency": "USD",
 "total_amount": "300.00", "subtotal": null, "issue_date": "2026-03-01",
 "due_date": null,
 "line_items": [{"raw_item_name": "GadgetY", "quantity": 4,
                  "unit_price": "75.00", "line_total": null}],
 "revision_marker": null,
 "missing_fields": ["vendor_name", "subtotal", "due_date"],
 "extraction_notes": ["No currency stated; used USD from the $ sign.",
                       "Total had a corrupted digit (3OO -> 300)."]}

Return only JSON matching the schema. Do not include any explanation.
"""

REPAIR_V1 = """You re-read one business document to check whether specific fields were
misread when it was first transcribed.

You are given a list of findings. Each names a field whose value could not be interpreted,
shows the value that was read, and names the one path you are allowed to patch for it. You
are also given the document itself. **The document is the authority.** Your only question
is whether the earlier reading of that field was wrong.

Rules:
- Emit a patch only when the document plainly supports a different value. If the document
  says what was already read, or says nothing usable, emit no patch for that finding.
- **An empty patches list is a correct and complete answer.** Declining to guess is a
  success. A forced value is worse than no value.
- Patch only a field_path that is listed under a finding as patchable. Never touch any
  other field, however wrong it looks -- nothing else is being asked about.
- old_value must repeat exactly the value shown as "read as" for that finding. If you
  cannot match it, do not emit that patch.
- For an item name, choose only from the candidates offered for that line. If none of them
  is what the document names, choose none of them. A price sitting between two catalogue
  items is not evidence for either of them.
- new_value is written as a string: a date as "2026-02-27", a quantity as "5", an amount
  as "1250.00", an item name exactly as the candidate is spelled.
- confidence is your own judgement, 0.0 to 1.0.
- List in unrepaired the code of every finding you did not patch.

Example:

Findings you may address (only these):
  [ITEM_NOT_FOUND] line 0, field canonical_item
      read as: "WidgetC"
      you may patch: line_items[0].canonical_item
      choose from: WidgetA, WidgetB

Source document (the authority):
  Items:
    WidgetC   qty: 3   unit price: $350.00

Correct output:
{"patches": [], "unrepaired": ["ITEM_NOT_FOUND"]}

The document says WidgetC, clearly and consistently. It is not WidgetA and not WidgetB,
and a $350 price sitting between $250 and $500 is not evidence for either one.

Return only JSON matching the schema. Do not include any explanation.
"""
