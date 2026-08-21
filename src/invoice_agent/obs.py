"""JSON-lines logging. Every record carries a run_id.

Bind once per run and the run_id cannot be forgotten:

    log = deps.logger.bind(run.run_id)
    with log.stage("extract") as fields:
        ...
        fields["invoice_number"] = extraction.invoice_number
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, TextIO
from uuid import UUID


def _encode(value: Any) -> str:
    """Decimal becomes a string, not a float -- log lines are evidence too."""
    if isinstance(value, Decimal | UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return repr(value)


class BoundLogger:
    def __init__(self, logger: JsonLogger, run_id: UUID) -> None:
        self._logger = logger
        self._run_id = run_id

    def emit(self, event: str, **fields: Any) -> None:
        self._logger.write({"run_id": str(self._run_id), "event": event, **fields})

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Time a stage and emit start/end. Mutate the yielded dict to attach
        fields to the completion record."""
        extra: dict[str, Any] = {}
        started = time.perf_counter()
        self.emit("stage.start", stage=name)
        try:
            yield extra
        except Exception as exc:
            self.emit(
                "stage.error",
                stage=name,
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
                **extra,
            )
            raise
        self.emit(
            "stage.end",
            stage=name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            **extra,
        )


class JsonLogger:
    """One JSON object per line. Inject a StringIO in tests."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), **record}
        self._stream.write(json.dumps(record, default=_encode) + "\n")
        self._stream.flush()

    def bind(self, run_id: UUID) -> BoundLogger:
        return BoundLogger(self, run_id)
