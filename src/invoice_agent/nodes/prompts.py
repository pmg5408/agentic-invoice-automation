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
