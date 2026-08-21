"""NVIDIA NIM adapter (build.nvidia.com).

OpenAI-compatible at https://integrate.api.nvidia.com/v1, with one important
deviation: NIM does **not** accept ``response_format={"type": "json_schema"}``
for LLMs. It wants the schema under its own extension:

    extra_body={"nvext": {"guided_json": <schema>}}

That quirk is the reason nodes never build a raw request. They hand an
LLMRequest to LLMClient and get back a validated model.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI

from invoice_agent.config import ProviderProfile, Settings
from invoice_agent.llm.client import LLMRequest, LLMUnavailableError, RawCompletion


def _messages(request: LLMRequest) -> list[dict[str, str]]:
    """Collapse consecutive same-role segments into one message each, keeping
    order. Segment boundaries are a caching concern, not a chat-protocol one."""
    messages: list[dict[str, str]] = []
    for seg in request.segments:
        if messages and messages[-1]["role"] == seg.role:
            messages[-1]["content"] += "\n\n" + seg.text
        else:
            messages.append({"role": seg.role, "content": seg.text})
    return messages


class NvidiaProvider:
    """Implements LLMProvider. One call, no parsing, no retry."""

    def __init__(self, profile: ProviderProfile, *, timeout_s: int = 60) -> None:
        api_key = os.environ.get(profile.api_key_env)
        if not api_key:
            raise LLMUnavailableError(
                f"{profile.api_key_env} is not set. Get a free key at "
                "https://build.nvidia.com (no card required) and put it in .env"
            )
        self._profile = profile
        self._client = OpenAI(
            base_url=profile.base_url, api_key=api_key, timeout=float(timeout_s)
        )

    def invoke(self, request: LLMRequest, schema: dict) -> RawCompletion:
        started = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=request.model,
            messages=_messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_body={"nvext": {"guided_json": schema}},
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = completion.usage
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        return RawCompletion(
            text=completion.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            # NIM reports no cache hits today; the field is here so a caching
            # provider populates it without any change downstream.
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            latency_ms=latency_ms,
        )


def build_provider(settings: Settings) -> NvidiaProvider:
    """Construct the provider named by config. The one place a provider name
    maps to an adapter."""
    profile = settings.profile
    if profile.name == "nvidia":
        return NvidiaProvider(profile, timeout_s=settings.request_timeout_s)
    raise ValueError(f"no adapter for provider {profile.name!r}")
