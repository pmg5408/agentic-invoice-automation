"""Loader behaviour. Deterministic, no LLM -- golden file per format."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pytest

from invoice_agent.nodes.load import make_load

# A minimal, valid, single blank-page PDF with no text layer at all.
BLANK_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << >> /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 0 >>
stream

endstream
endobj
trailer
<< /Size 5 /Root 1 0 R >>
"""


def _seed(source_path: str, run_id=None):
    from datetime import UTC, datetime

    from invoice_agent.models import InvoiceRun, SourceDocument

    rid = run_id or uuid4()
    return InvoiceRun(
        run_id=rid,
        status="running",
        current_stage="load",
        source=SourceDocument(
            run_id=uuid4(),  # deliberately not rid -- load must replace it
            source_path=source_path,
            source_filename=Path(source_path).name,
            source_format="txt",
            raw_text="",
            content_sha256="",
            text_extraction_ok=True,
            loaded_at=datetime.now(UTC),
        ),
        started_at=datetime.now(UTC),
    )


class TestGoldenFiles:
    @pytest.mark.parametrize(
        "filename,expected_format",
        [
            ("invoice_1001.txt", "txt"),
            ("invoice_1006.csv", "csv"),
            ("invoice_1014.xml", "xml"),
            ("invoice_1004.json", "json"),
        ],
    )
    def test_text_formats_pass_through_verbatim(self, deps, invoice_dir, filename, expected_format):
        path = invoice_dir / filename
        run_id = uuid4()
        run = _seed(str(path), run_id=run_id)
        result = make_load(deps)(run)

        doc = result["source"]
        raw_bytes = path.read_bytes()
        assert doc.source_format == expected_format
        assert doc.raw_text == raw_bytes.decode("utf-8")
        assert doc.text_extraction_ok is True
        assert doc.content_sha256 == hashlib.sha256(raw_bytes).hexdigest()
        assert doc.source_filename == filename
        # The placeholder's throwaway run_id must be replaced by the real one.
        assert doc.run_id == run_id
        assert "error" not in result

    def test_row_per_line_csv_keeps_the_totals_block(self, deps, invoice_dir):
        # invoice_1007.csv: row-per-line-item with a trailing totals block --
        # csv.DictReader would clobber this; verbatim passthrough can't.
        path = invoice_dir / "invoice_1007.csv"
        result = make_load(deps)(_seed(str(path)))
        assert "Subtotal:" in result["source"].raw_text
        assert result["source"].raw_text.count("WidgetA") == 1

    def test_key_value_csv_keeps_repeated_item_keys(self, deps, invoice_dir):
        # invoice_1006.csv repeats the `item` key per line -- DictReader
        # would keep only the last one.
        path = invoice_dir / "invoice_1006.csv"
        result = make_load(deps)(_seed(str(path)))
        assert result["source"].raw_text.count("item,") == 2

    def test_pdf_with_text_layer_extracts_it(self, deps, invoice_dir):
        path = invoice_dir / "invoice_1011.pdf"
        result = make_load(deps)(_seed(str(path)))
        doc = result["source"]
        assert doc.text_extraction_ok is True
        assert "INV-1011" in doc.raw_text
        assert "Summit Manufacturing Co." in doc.raw_text

    def test_pdf_without_text_layer_is_flagged(self, deps, tmp_path):
        blank = tmp_path / "invoice_blank.pdf"
        blank.write_bytes(BLANK_PDF)
        result = make_load(deps)(_seed(str(blank)))
        doc = result["source"]
        assert doc.text_extraction_ok is False
        assert doc.raw_text == ""
        # No text layer isn't a read failure -- content_sha256 is still real.
        assert doc.content_sha256 == hashlib.sha256(BLANK_PDF).hexdigest()
        assert "error" not in result


class TestUnreadableFiles:
    def test_missing_file_is_flagged_not_raised(self, deps, tmp_path):
        missing = tmp_path / "invoice_9999.txt"
        result = make_load(deps)(_seed(str(missing)))
        doc = result["source"]
        assert doc.text_extraction_ok is False
        assert doc.content_sha256 == ""
        assert "cannot read" in result["error"]

    def test_corrupt_pdf_is_flagged_not_raised(self, deps, tmp_path):
        corrupt = tmp_path / "invoice_corrupt.pdf"
        corrupt.write_bytes(b"not actually a pdf")
        result = make_load(deps)(_seed(str(corrupt)))
        doc = result["source"]
        assert doc.text_extraction_ok is False
        assert doc.raw_text == ""
        # We could still hash the bytes even though we couldn't parse them.
        assert doc.content_sha256 == hashlib.sha256(b"not actually a pdf").hexdigest()
        assert "cannot extract text" in result["error"]

    def test_never_raises_a_stack_trace(self, deps, tmp_path):
        missing = tmp_path / "ghost.csv"
        # The only thing allowed to raise out of load() is an unsupported
        # extension, and cli.py already screens those out before a run starts.
        make_load(deps)(_seed(str(missing)))


class TestAllSampleInvoicesLoad:
    def test_every_file_in_data_invoices_loads_without_exception(self, deps, invoice_dir):
        files = sorted(invoice_dir.iterdir())
        assert len(files) == 20
        for path in files:
            result = make_load(deps)(_seed(str(path)))
            doc = result["source"]
            assert doc.content_sha256 != ""
            if doc.text_extraction_ok:
                assert doc.raw_text != ""
