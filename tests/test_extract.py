"""Extractor behaviour. No API key needed -- everything runs against the stub."""

from __future__ import annotations

from dataclasses import replace

from invoice_agent.llm.client import LLMClient
from invoice_agent.llm.stub import StubProvider
from invoice_agent.models import StageMetrics
from invoice_agent.nodes.extract import PROMPT_VERSION, make_extract


def _client(responses, max_retries=1):
    provider = StubProvider(responses)
    return provider, LLMClient(provider, max_retries=max_retries, backoff_base_s=0.0)


class TestHappyPath:
    def test_calls_the_model_and_returns_the_parsed_extraction(
        self, deps, make_run, make_extraction
    ):
        extraction = make_extraction()
        provider, client = _client([extraction])
        node = make_extract(replace(deps, llm=client))

        result = node(make_run())

        assert result["extraction"] == extraction
        assert provider.call_count == 1
        metrics = result["stage_metrics"]["extract"]
        assert metrics.model == deps.settings.stage("extract").model
        assert metrics.prompt_version == PROMPT_VERSION

    def test_earlier_stage_metrics_survive_the_merge(self, deps, make_run, make_extraction):
        _, client = _client([make_extraction()])
        node = make_extract(replace(deps, llm=client))
        run = make_run(stage_metrics={"load": StageMetrics(latency_ms=5)})

        result = node(run)

        assert result["stage_metrics"]["load"].latency_ms == 5
        assert "extract" in result["stage_metrics"]

    def test_no_error_key_on_success(self, deps, make_run, make_extraction):
        _, client = _client([make_extraction()])
        node = make_extract(replace(deps, llm=client))
        assert "error" not in node(make_run())


class TestRetry:
    def test_malformed_then_valid_retries_once_and_succeeds(
        self, deps, make_run, make_extraction
    ):
        extraction = make_extraction()
        provider, client = _client(['{"bad":', extraction])
        node = make_extract(replace(deps, llm=client))

        result = node(make_run())

        assert result["extraction"] == extraction
        assert provider.call_count == 2

    def test_exhausted_retries_falls_back_to_an_empty_extraction(self, deps, make_run):
        provider, client = _client(['{"bad":', '{"still bad":'])
        node = make_extract(replace(deps, llm=client))

        result = node(make_run())

        extraction = result["extraction"]
        assert extraction.invoice_number is None
        assert extraction.line_items == []
        assert set(extraction.missing_fields) == {
            "invoice_number", "vendor_name", "currency", "total_amount",
            "subtotal", "issue_date", "due_date", "line_items", "revision_marker",
        }
        assert "error" in result
        assert provider.call_count == 2

    def test_fallback_still_records_which_model_was_asked(self, deps, make_run):
        _, client = _client(['{"bad":', '{"still bad":'])
        node = make_extract(replace(deps, llm=client))

        metrics = node(make_run())["stage_metrics"]["extract"]

        assert metrics.model == deps.settings.stage("extract").model
        assert metrics.input_tokens == 0
        assert metrics.cost_usd == 0


class TestTextExtractionNotOk:
    def test_skips_the_model_entirely(self, deps, make_run, make_source):
        provider, client = _client(["should never be read"])
        node = make_extract(replace(deps, llm=client))
        run = make_run(source=make_source(text_extraction_ok=False))

        result = node(run)

        assert provider.call_count == 0
        assert result["extraction"].missing_fields
        assert result["extraction"].line_items == []

    def test_no_error_key_since_not_calling_the_model_is_not_a_failure(
        self, deps, make_run, make_source
    ):
        _, client = _client(["should never be read"])
        node = make_extract(replace(deps, llm=client))
        run = make_run(source=make_source(text_extraction_ok=False))
        assert "error" not in node(run)


class TestPromptShape:
    def test_stable_instructions_come_first_and_are_cacheable(
        self, deps, make_run, make_extraction
    ):
        provider, client = _client([make_extraction()])
        node = make_extract(replace(deps, llm=client))
        node(make_run())

        request = provider.requests[0]
        assert request.segments[0].role == "system"
        assert request.segments[0].cacheable is True

    def test_invoice_text_is_last_and_not_cacheable(
        self, deps, make_run, make_source, make_extraction
    ):
        provider, client = _client([make_extraction()])
        node = make_extract(replace(deps, llm=client))
        run = make_run(source=make_source(text="UNIQUE-TOKEN-42"))

        node(run)

        last = provider.requests[0].segments[-1]
        assert last.cacheable is False
        assert "UNIQUE-TOKEN-42" in last.text
