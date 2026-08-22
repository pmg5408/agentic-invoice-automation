"""Node: load.

Brief : docs/components.md section 1 -- Loader
LLM   : No
In    : run.source.source_path (the graph is seeded with a stub SourceDocument)
Out   : SourceDocument

Convert txt / json / csv / xml / pdf to text. Preserve structure: pretty-print
JSON, keep CSV tabular, pdfplumber.extract_text() for PDF.

The loader does not know what an invoice is. It interprets nothing.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pdfplumber

from invoice_agent.deps import Deps, NodeFn
from invoice_agent.models import InvoiceRun, SourceDocument

# The extension -> format mapping. Owned here because reading a file is the
# loader's concern; cli.py imports this rather than keeping a second copy so
# it can reject unreadable files before a run starts.
SUFFIX_FORMATS: dict[str, str] = {
    ".txt": "txt",
    ".json": "json",
    ".csv": "csv",
    ".xml": "xml",
    ".pdf": "pdf",
}


class UnsupportedFormat(Exception):
    """The file extension maps to no loader. Guessing "txt" here would hand the
    extraction model mojibake, which it would confidently extract from."""


def _extract_text(path: Path, raw_bytes: bytes, source_format: str) -> tuple[str, bool]:
    """Return (raw_text, text_extraction_ok).

    txt/json/csv/xml pass through verbatim -- the file already carries its
    structure (pretty-printed JSON, tabular CSV), and reparsing JSON to
    "normalize" it would round-trip every amount through a Python float
    before re-serializing, silently rewriting "250.00" as "250.0". PDF has no
    text layer to preserve, so pdfplumber.extract_text() is the only option.
    """
    if source_format == "pdf":
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        return text, bool(text)

    text = raw_bytes.decode("utf-8")
    return text, True


def _load_document(run: InvoiceRun) -> tuple[SourceDocument, str | None]:
    """Read the file at run.source.source_path and build the real SourceDocument.

    Returns (document, error). error is None on a clean read; otherwise it is
    the text that belongs in InvoiceRun.error -- never a raised exception, per
    architecture.md's "loader cannot read file -> needs_human_review".
    """
    path = Path(run.source.source_path)
    source_format = SUFFIX_FORMATS.get(path.suffix.lower())
    if source_format is None:
        raise UnsupportedFormat(
            f"unsupported file type {path.suffix or path.name!r} -- "
            f"expected one of {', '.join(sorted(SUFFIX_FORMATS))}"
        )

    now = datetime.now(UTC)

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return (
            SourceDocument(
                run_id=run.run_id,
                source_path=str(path),
                source_filename=path.name,
                source_format=source_format,
                raw_text="",
                content_sha256="",
                text_extraction_ok=False,
                loaded_at=now,
            ),
            f"load: cannot read {path.name}: {exc}",
        )

    content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw_text, text_extraction_ok = _extract_text(path, raw_bytes, source_format)
        error = None
    except Exception as exc:  # noqa: BLE001 -- any extraction failure degrades to review
        raw_text, text_extraction_ok = "", False
        error = f"load: cannot extract text from {path.name}: {exc}"

    document = SourceDocument(
        run_id=run.run_id,
        source_path=str(path),
        source_filename=path.name,
        source_format=source_format,
        raw_text=raw_text,
        content_sha256=content_sha256,
        text_extraction_ok=text_extraction_ok,
        loaded_at=now,
    )
    return document, error


def make_load(deps: Deps) -> NodeFn:
    """Build the load node. Closes over Deps so the node itself stays testable."""

    def load(run: InvoiceRun) -> dict:
        log = deps.logger.bind(run.run_id)
        with log.stage("load") as fields:
            document, error = _load_document(run)
            fields["source_format"] = document.source_format
            fields["text_extraction_ok"] = document.text_extraction_ok

        update: dict[str, Any] = {"source": document}
        if error is not None:
            update["error"] = error
        return update

    return load
