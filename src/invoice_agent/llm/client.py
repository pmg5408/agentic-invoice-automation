"""The LLM boundary. One interface so the provider can be swapped.

Two layers, deliberately:

* ``LLMProvider`` -- what an adapter implements. One raw call, no parsing, no
  retry. Adding a provider means adding one of these and a profile in config.
* ``LLMClient``   -- what nodes use. Wraps a provider with parse -> validate ->
  retry, so architecture.md's failure contract holds uniformly instead of being
  reimplemented in four nodes.

Prompts are ordered ``PromptSegment`` lists, not f-strings. Stable content
first, marked ``cacheable=True``. The NVIDIA adapter ignores the flag; a
caching provider reads it and places its breakpoints. Writing prompts this way
now is what keeps enabling prompt caching later a config change rather than a
rewrite of every prompt and its fixtures.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from invoice_agent.models import StageMetrics


class LLMError(Exception):
    """Base for every LLM failure. A node catching this routes the run to
    needs_human_review -- never a stack trace as the terminal state."""


class LLMSchemaError(LLMError):
    """Output never validated against the schema, even after retries."""


class LLMUnavailableError(LLMError):
    """Transport failure: unreachable, timed out, rate limited."""


class PromptSegment(BaseModel):
    """One ordered chunk of a prompt.

    ``cacheable`` marks content that is byte-identical across invoices --
    instructions, schema, few-shot examples. Never put a timestamp, run id, or
    invoice text in a cacheable segment.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    role: Literal["system", "user"] = "user"
    cacheable: bool = False


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    prompt_version: str
    segments: list[PromptSegment]
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096

    @property
    def cacheable_prefix_len(self) -> int:
        """How many leading segments are stable. Adapters that support caching
        place their breakpoint here."""
        count = 0
        for seg in self.segments:
            if not seg.cacheable:
                break
            count += 1
        return count


class RawCompletion(BaseModel):
    """What an adapter returns: text plus usage. No parsing."""

    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0


class LLMResponse[T: BaseModel](BaseModel):
    """What a node receives: a validated model, the raw text behind it, and
    metrics summed across every attempt including the failed ones."""

    model_config = ConfigDict(frozen=True)

    parsed: T
    raw_text: str
    attempts: int
    metrics: StageMetrics


class LLMProvider(Protocol):
    """One raw call to a model. Implemented per provider."""

    def invoke(self, request: LLMRequest, schema: dict) -> RawCompletion: ...


class LLMClient:
    """Parse, validate, retry. Provider-agnostic by construction."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_retries: int = 1,
        backoff_base_s: float = 0.5,
    ) -> None:
        self._provider = provider
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    def complete[T: BaseModel](
        self, request: LLMRequest, response_model: type[T]
    ) -> LLMResponse[T]:
        """Call the model until the output validates, or give up.

        Raises LLMSchemaError or LLMUnavailableError. Both are LLMError, and a
        node turns either into a terminal needs_human_review.
        """
        schema = response_model.model_json_schema()
        totals = _MetricAccumulator(request)
        attempt_request = request
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 2):
            try:
                raw = self._provider.invoke(attempt_request, schema)
            except LLMError:
                raise
            except Exception as exc:  # transport
                last_error = exc
                totals.record_failure()
                if attempt > self._max_retries:
                    raise LLMUnavailableError(
                        f"{request.stage}: provider unreachable after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
                continue

            totals.add(raw)
            try:
                parsed = response_model.model_validate_json(_strip_fences(raw.text))
            except ValidationError as exc:
                last_error = exc
                if attempt > self._max_retries:
                    break
                # Corrective retry: hand the model its own error back. Appended
                # as a non-cacheable trailing segment so the stable prefix is
                # untouched.
                attempt_request = request.model_copy(
                    update={
                        "segments": [
                            *request.segments,
                            PromptSegment(
                                text=(
                                    "Your previous response did not match the required "
                                    f"schema:\n{exc}\n\nReturn only valid JSON matching "
                                    "the schema. Do not explain."
                                ),
                                role="user",
                                cacheable=False,
                            ),
                        ]
                    }
                )
                continue

            return LLMResponse[response_model](
                parsed=parsed,
                raw_text=raw.text,
                attempts=attempt,
                metrics=totals.build(),
            )

        raise LLMSchemaError(
            f"{request.stage}: output failed schema validation after "
            f"{self._max_retries + 1} attempts: {last_error}"
        )


class _MetricAccumulator:
    """Sums usage across attempts. A run that retried cost twice; the metrics
    should say so."""

    def __init__(self, request: LLMRequest) -> None:
        self.request = request
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.latency_ms = 0

    def add(self, raw: RawCompletion) -> None:
        self.input_tokens += raw.input_tokens
        self.output_tokens += raw.output_tokens
        self.cached_tokens += raw.cached_tokens
        self.latency_ms += raw.latency_ms

    def record_failure(self) -> None:
        """Transport failures produce no usage but still burn wall clock."""

    def build(self) -> StageMetrics:
        return StageMetrics(
            # Provenance comes off the request, so no node ever sets it and no
            # LLM is ever asked to invent it.
            model=self.request.model,
            prompt_version=self.request.prompt_version,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            latency_ms=self.latency_ms,
            # NVIDIA's free tier bills nothing. A paid adapter fills this in;
            # nothing downstream changes when it does.
            cost_usd=Decimal("0"),
        )


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences despite guided decoding."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()
