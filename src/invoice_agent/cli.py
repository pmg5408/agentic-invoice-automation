"""Command-line entry point.

    invoice-agent --invoice-path data/invoices/invoice_1001.txt
    invoice-agent --batch data/invoices/

Step 0 wires the pipeline together and runs it. Batch fan-out, metric roll-up
and the read-only UI belong to docs/components.md section 9; this is the
skeleton they build on.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from invoice_agent.config import get_settings
from invoice_agent.deps import Deps
from invoice_agent.graph import build_graph
from invoice_agent.llm.client import LLMClient, LLMError
from invoice_agent.llm.nvidia import build_provider
from invoice_agent.models import InvoiceRun, SourceDocument
from invoice_agent.nodes.load import SUFFIX_FORMATS, UnsupportedFormat
from invoice_agent.obs import JsonLogger
from invoice_agent.repository import SqliteInvoiceRepository


def build_deps() -> Deps:
    settings = get_settings()
    return Deps(
        settings=settings,
        repo=SqliteInvoiceRepository(settings.db_path, initialize=True),
        llm=LLMClient(build_provider(settings), max_retries=settings.max_llm_retries),
        logger=JsonLogger(),
    )


def seed_run(path: Path) -> InvoiceRun:
    """The initial state. `load` replaces `source` with the real thing -- it
    owns reading the file, so the CLI only records where to look."""
    source_format = SUFFIX_FORMATS.get(path.suffix.lower())
    if source_format is None:
        raise UnsupportedFormat(
            f"unsupported file type {path.suffix or path.name!r} -- "
            f"expected one of {', '.join(sorted(SUFFIX_FORMATS))}"
        )

    now = datetime.now(UTC)
    return InvoiceRun(
        run_id=uuid4(),
        status="running",
        current_stage="load",
        source=SourceDocument(
            run_id=uuid4(),
            source_path=str(path),
            source_filename=path.name,
            source_format=source_format,
            raw_text="",
            content_sha256="",
            text_extraction_ok=True,
            loaded_at=now,
        ),
        started_at=now,
    )


def resolve_targets(args: argparse.Namespace) -> list[Path]:
    if args.invoice_path:
        return [Path(args.invoice_path)]
    batch = Path(args.batch)
    return sorted(p for p in batch.iterdir() if p.suffix.lower() in SUFFIX_FORMATS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="invoice-agent", description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--invoice-path", help="a single invoice file")
    target.add_argument("--batch", help="a directory of invoices")
    args = parser.parse_args(argv)

    try:
        deps = build_deps()
    except LLMError as exc:
        # Never a stack trace as the terminal state.
        print(f"cannot start: {exc}", file=sys.stderr)
        return 1

    graph = build_graph(deps)
    exit_code = 0

    for path in resolve_targets(args):
        try:
            run = seed_run(path)
        except UnsupportedFormat as exc:
            # A file we cannot read is a clean stop, not a guess. Never a stack
            # trace as the terminal state.
            print(f"skipped {path.name}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            graph.invoke(run)
        except NotImplementedError as exc:
            print(f"pipeline stopped: {exc}", file=sys.stderr)
            return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
