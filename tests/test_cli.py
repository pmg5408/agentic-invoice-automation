"""Entry-point behaviour. seed_run is a pure function, so it tests directly."""

from __future__ import annotations

from pathlib import Path

import pytest

from invoice_agent.cli import SUFFIX_FORMATS, UnsupportedFormat, seed_run


class TestSeedRun:
    @pytest.mark.parametrize("suffix", sorted(SUFFIX_FORMATS))
    def test_every_supported_suffix_maps_to_its_format(self, suffix: str):
        run = seed_run(Path(f"data/invoices/invoice_1001{suffix}"))
        assert run.source.source_format == SUFFIX_FORMATS[suffix]

    def test_uppercase_suffix_is_accepted(self):
        assert seed_run(Path("INV.PDF")).source.source_format == "pdf"

    @pytest.mark.parametrize("name", ["invoice.docx", "invoice.eml", "invoice"])
    def test_unreadable_file_stops_rather_than_guessing(self, name: str):
        """Falling back to "txt" would hand the extraction model mojibake, and
        it would extract from it confidently. A clean stop beats a bad guess."""
        with pytest.raises(UnsupportedFormat) as exc:
            seed_run(Path(name))
        assert "unsupported file type" in str(exc.value)
