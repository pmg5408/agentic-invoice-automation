"""The retry contract from architecture.md, exercised through the real client.

    schema validation fails -> retry once -> still failing -> needs_human_review
    model unreachable       -> retry with backoff -> needs_human_review
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from invoice_agent.llm.client import (
    LLMClient,
    LLMRequest,
    LLMSchemaError,
    LLMUnavailableError,
    PromptSegment,
)
from invoice_agent.llm.stub import RaisingProvider, StubProvider, stub_client
from invoice_agent.models import (
    ApprovalDraft,
    Critique,
    ExtractedInvoice,
    RepairOutput,
)


def request(stage: str = "extract") -> LLMRequest:
    return LLMRequest(
        stage=stage,
        prompt_version="v1",
        model="meta/llama-3.3-70b-instruct",
        segments=[
            PromptSegment(text="You extract invoices.", role="system", cacheable=True),
            PromptSegment(text="<schema and few-shot>", role="user", cacheable=True),
            PromptSegment(text="INVOICE\nTotal: $5,000.00", role="user"),
        ],
    )


class TestHappyPath:
    def test_returns_a_validated_model(self, make_extraction):
        expected = make_extraction()
        response = stub_client([expected]).complete(request(), ExtractedInvoice)
        assert response.parsed.invoice_number == "INV-1001"
        assert response.attempts == 1

    def test_metrics_are_recorded(self, make_extraction):
        response = stub_client([make_extraction()]).complete(request(), ExtractedInvoice)
        assert response.metrics.input_tokens == 100
        assert response.metrics.output_tokens == 50

    def test_tolerates_markdown_fenced_json(self, make_extraction):
        fenced = "```json\n" + make_extraction().model_dump_json() + "\n```"
        response = stub_client([fenced]).complete(request(), ExtractedInvoice)
        assert response.parsed.invoice_number == "INV-1001"

    def test_schema_is_handed_to_the_provider(self, make_extraction):
        provider = StubProvider([make_extraction()])
        LLMClient(provider, backoff_base_s=0.0).complete(request(), ExtractedInvoice)
        assert "invoice_number" in provider.schemas[0]["properties"]


class TestSchemaRetry:
    def test_malformed_then_valid_succeeds_on_the_retry(self, make_extraction):
        response = stub_client(['{"not": "an invoice"', make_extraction()]).complete(
            request(), ExtractedInvoice
        )
        assert response.attempts == 2
        assert response.parsed.invoice_number == "INV-1001"

    def test_two_failures_raise_llm_schema_error(self):
        with pytest.raises(LLMSchemaError, match="extract"):
            stub_client(['{"bad":', '{"still bad":']).complete(request(), ExtractedInvoice)

    def test_retry_count_honours_config(self):
        provider = StubProvider(['{"bad":'])
        client = LLMClient(provider, max_retries=1, backoff_base_s=0.0)
        with pytest.raises(LLMSchemaError):
            client.complete(request(), ExtractedInvoice)
        assert provider.call_count == 2  # initial + one retry

    def test_metrics_include_the_failed_attempt(self, make_extraction):
        response = stub_client(['{"bad":', make_extraction()]).complete(
            request(), ExtractedInvoice
        )
        # Both calls cost tokens; the run should say so.
        assert response.metrics.input_tokens == 200

    def test_retry_appends_the_error_without_touching_the_stable_prefix(self, make_extraction):
        provider = StubProvider(['{"bad":', make_extraction()])
        LLMClient(provider, backoff_base_s=0.0).complete(request(), ExtractedInvoice)
        first, second = provider.requests
        assert second.segments[: len(first.segments)] == first.segments
        assert second.segments[-1].cacheable is False
        assert first.cacheable_prefix_len == second.cacheable_prefix_len == 2


class TestTransportFailure:
    def test_unreachable_provider_raises_after_retries(self):
        provider = RaisingProvider()
        client = LLMClient(provider, max_retries=1, backoff_base_s=0.0)
        with pytest.raises(LLMUnavailableError, match="unreachable"):
            client.complete(request(), ExtractedInvoice)
        assert provider.call_count == 2


class TestPromptSegments:
    def test_cacheable_prefix_stops_at_the_first_variable_segment(self):
        assert request().cacheable_prefix_len == 2

    def test_segments_are_immutable(self):
        seg = PromptSegment(text="stable", cacheable=True)
        with pytest.raises(ValidationError):
            seg.text = "changed"


class TestProvenance:
    """Nodes never set which model ran or which prompt built the call -- the
    client already holds both on the request. Nothing to forget, and nothing
    for the LLM to invent."""

    def test_metrics_carry_model_and_prompt_version(self, make_extraction):
        response = stub_client([make_extraction()]).complete(request(), ExtractedInvoice)
        assert response.metrics.model == "meta/llama-3.3-70b-instruct"
        assert response.metrics.prompt_version == "v1"

    def test_provenance_survives_a_corrective_retry(self, make_extraction):
        client = stub_client(['{"bad":', make_extraction()])
        response = client.complete(request(), ExtractedInvoice)
        assert response.attempts == 2
        assert response.metrics.prompt_version == "v1"


class TestStructuredOutputShape:
    """How a model is asked for a shape is dispatched, not hardcoded. The mode
    is a measured property of the model behind the endpoint."""

    def test_json_schema_uses_response_format(self):
        from invoice_agent.llm.nvidia import _shape

        kw = _shape("json_schema", ExtractedInvoice.model_json_schema())
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True

    def test_nvext_uses_extra_body(self):
        from invoice_agent.llm.nvidia import _shape

        kw = _shape("nvext_guided_json", {"title": "X"})
        assert kw["extra_body"]["nvext"]["guided_json"] == {"title": "X"}

    def test_tool_use_forces_a_single_tool(self):
        from invoice_agent.llm.nvidia import _shape

        kw = _shape("tool_use", {"title": "X"})
        assert kw["tool_choice"]["function"]["name"] == "X"

    def test_unknown_mode_is_an_llm_error(self):
        from invoice_agent.llm.client import LLMError
        from invoice_agent.llm.nvidia import _shape

        with pytest.raises(LLMError):
            _shape("telepathy", {})


class TestMoneySchemaIsCompilable:
    """Constrained decoding cannot compile look-around. Pydantic's Decimal
    schema contains a negative lookahead, so money is advertised as a plain
    string; validation is unchanged."""

    @pytest.mark.parametrize(
        "payload", [ExtractedInvoice, RepairOutput, ApprovalDraft, Critique]
    )
    def test_no_lookaround_in_any_llm_schema(self, payload):
        assert "(?!" not in json.dumps(payload.model_json_schema())

    @pytest.mark.parametrize("bad", ["1.2.3", "-", "abc", ""])
    def test_bad_money_is_still_rejected(self, bad):
        with pytest.raises(ValidationError):
            ExtractedInvoice.model_validate_json(json.dumps({"total_amount": bad}))

    def test_money_parses_to_an_exact_decimal(self):
        inv = ExtractedInvoice.model_validate_json('{"total_amount":"22562.80"}')
        assert inv.total_amount == Decimal("22562.80")
