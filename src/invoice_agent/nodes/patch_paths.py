"""The ``FieldPatch.field_path`` grammar, in one place.

``FieldPatch.field_path`` is a plain string, because the repair model writes it
as JSON. Two nodes have to agree on what those strings mean:

* ``repair`` writes them -- a finding's ``Scope`` becomes the one path that
  finding licenses the model to patch, and the path comes back on the patch.
* ``validate`` reads them -- a path says which field of the extraction to read
  through on the second pass.

Written out separately in each node, the two halves can drift: change the
format on one side and the other silently stops matching, patches get dropped
as unparseable, and the repair loop quietly does nothing. Both sides call the
same two functions here instead.

    build(LineScope(line_index=2, field="canonical_item"))  # "line_items[2].canonical_item"
    parse("line_items[2].canonical_item")                   # PatchPath(2, "canonical_item")
    build(InvoiceScope(field="due_date"))                   # "due_date"
    parse("due_date")                                       # PatchPath(None, "due_date")

Note that ``line_items[i].canonical_item`` names something the extraction does
not have -- ``canonical_item`` is a field of ``ItemResolution``, which is
validate's own output. The path is still the right way to say which line's item
identity is under question; validate applies it as a resolution override rather
than as a field copy.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from invoice_agent.models import Scope

_LINE_ITEMS = "line_items"
_LINE_PATH = re.compile(rf"^{_LINE_ITEMS}\[(\d+)\]\.(\w+)$")


class PatchPath(NamedTuple):
    """A parsed path. ``line_index`` is None for an invoice-level field."""

    line_index: int | None
    field: str


def build(scope: Scope) -> str | None:
    """The path naming the field a scope points at.

    None when the scope points at no particular field -- a whole-invoice
    finding has nothing patchable, so nothing is offered to the model.
    """
    if scope.field is None:
        return None
    if scope.kind == "line":
        return f"{_LINE_ITEMS}[{scope.line_index}].{scope.field}"
    return scope.field


def parse(field_path: str) -> PatchPath | None:
    """Read a path back. None when it is not a path this grammar produces.

    An unrecognised path is a patch aimed at something nobody named, so the
    caller drops it rather than guessing what was meant.
    """
    match = _LINE_PATH.match(field_path)
    if match is not None:
        return PatchPath(int(match[1]), match[2])
    if field_path.isidentifier():
        return PatchPath(None, field_path)
    return None
