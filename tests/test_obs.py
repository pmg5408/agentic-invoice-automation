"""JSON lines, run_id on every record."""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import uuid4

import pytest


def lines(stream) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_every_record_carries_run_id_and_timestamp(logger, log_stream):
    run_id = uuid4()
    logger.bind(run_id).emit("extracted", invoice_number="INV-1001")
    (record,) = lines(log_stream)
    assert record["run_id"] == str(run_id)
    assert record["event"] == "extracted"
    assert record["invoice_number"] == "INV-1001"
    assert "ts" in record


def test_output_is_one_json_object_per_line(logger, log_stream):
    log = logger.bind(uuid4())
    log.emit("a")
    log.emit("b")
    assert len(lines(log_stream)) == 2


def test_decimal_is_logged_as_string_not_float(logger, log_stream):
    logger.bind(uuid4()).emit("paid", amount=Decimal("22562.80"))
    assert lines(log_stream)[0]["amount"] == "22562.80"


def test_stage_emits_start_and_end_with_latency(logger, log_stream):
    with logger.bind(uuid4()).stage("extract"):
        pass
    start, end = lines(log_stream)
    assert start["event"] == "stage.start" and start["stage"] == "extract"
    assert end["event"] == "stage.end" and end["latency_ms"] >= 0


def test_stage_fields_land_on_the_completion_record(logger, log_stream):
    with logger.bind(uuid4()).stage("extract") as fields:
        fields["invoice_number"] = "INV-1002"
    assert lines(log_stream)[-1]["invoice_number"] == "INV-1002"


def test_stage_error_is_logged_and_reraised(logger, log_stream):
    with pytest.raises(ValueError), logger.bind(uuid4()).stage("extract"):
        raise ValueError("boom")
    error = lines(log_stream)[-1]
    assert error["event"] == "stage.error"
    assert error["error_type"] == "ValueError"
    assert error["error"] == "boom"
