"""Stub provider for tests. No network, no API key.

Only components 2, 5 and 7 need this. Components 1, 3, 4, 6 and 8 read
structured fields, so their tests construct ExtractedInvoice / Finding /
PolicyGate literals directly -- an LLM has no place in them.

Each canned response is **either a model instance or a raw string**. Strings
matter: they drive the parse, validate and retry path in client.py, which is the
code most likely to be wrong.

    stub_client([extraction])                     # happy path
    stub_client(['{"invoice_number": '])          # never validates -> LLMSchemaError
    stub_client(['{"bad":', extraction])          # retry, then succeed
"""

from __future__ import annotations

from pydantic import BaseModel

from invoice_agent.llm.client import LLMClient, LLMRequest, RawCompletion

Response = BaseModel | str


class StubProvider:
    """Implements LLMProvider. Returns canned responses in order.

    Records every request it received, so a test can assert on prompt structure
    -- that the stable prefix really is stable, for instance.
    """

    def __init__(self, responses: list[Response], *, latency_ms: int = 1) -> None:
        if not responses:
            raise ValueError("StubProvider needs at least one response")
        self._responses = list(responses)
        self._latency_ms = latency_ms
        self.requests: list[LLMRequest] = []
        self.schemas: list[dict] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def invoke(self, request: LLMRequest, schema: dict) -> RawCompletion:
        self.requests.append(request)
        self.schemas.append(schema)
        # Past the end, keep returning the last response, so a test that only
        # cares about the happy path does not have to count calls.
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        text = response.model_dump_json() if isinstance(response, BaseModel) else response
        return RawCompletion(
            text=text,
            input_tokens=100,
            output_tokens=50,
            latency_ms=self._latency_ms,
        )


class RaisingProvider:
    """Implements LLMProvider. Always raises -- for the transport-failure path."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or ConnectionError("simulated network failure")
        self.call_count = 0

    def invoke(self, request: LLMRequest, schema: dict) -> RawCompletion:
        self.call_count += 1
        raise self._error


def stub_client(responses: list[Response], *, max_retries: int = 1) -> LLMClient:
    """A real LLMClient backed by canned responses.

    Deliberately the real client: tests exercise the actual parse/validate/retry
    logic rather than a second implementation of it.
    """
    return LLMClient(StubProvider(responses), max_retries=max_retries, backoff_base_s=0.0)
