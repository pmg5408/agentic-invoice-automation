"""Prove the NVIDIA NIM path works end to end before anything is built on it.

    python scripts/smoke_llm.py

Needs NVIDIA_API_KEY. Get one free at https://build.nvidia.com -- no card.

This checks the one assumption the extractor rests on: that NIM's
`nvext.guided_json` really does constrain output to a Pydantic schema. If this
fails, the structured_output_mode in config.py is wrong for the chosen model,
not the prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

from invoice_agent.config import get_settings
from invoice_agent.llm.client import LLMClient, LLMError, LLMRequest, PromptSegment
from invoice_agent.llm.nvidia import build_provider
from invoice_agent.models import ExtractedInvoice

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "invoices" / "invoice_1001.txt"

INSTRUCTIONS = """You extract structured data from invoices.

Return only JSON matching the schema. Rules:
- A field that is absent is null, plus an entry in missing_fields. Never guess.
- Normalize invoice numbers to the form INV-1234.
- Keep raw_item_name exactly as written in the document.
- Set model to the model id and prompt_version to "smoke-v1".
"""


def main() -> int:
    settings = get_settings()
    stage = settings.stage("extract")

    try:
        client = LLMClient(build_provider(settings), max_retries=settings.max_llm_retries)
    except LLMError as exc:
        print(f"cannot reach provider: {exc}", file=sys.stderr)
        return 1

    request = LLMRequest(
        stage="extract",
        prompt_version="smoke-v1",
        model=stage.model,
        temperature=stage.temperature,
        max_tokens=stage.max_tokens,
        segments=[
            # Stable prefix: identical for every invoice.
            PromptSegment(text=INSTRUCTIONS, role="system", cacheable=True),
            # Variable tail: this invoice only.
            PromptSegment(text=f"Invoice document:\n\n{SAMPLE.read_text()}", role="user"),
        ],
    )

    print(f"provider : {settings.profile.name} ({settings.profile.structured_output_mode})")
    print(f"model    : {stage.model}")
    print(f"invoice  : {SAMPLE.name}\n")

    try:
        response = client.complete(request, ExtractedInvoice)
    except LLMError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    parsed = response.parsed
    print(f"attempts : {response.attempts}")
    print(f"tokens   : in={response.metrics.input_tokens} out={response.metrics.output_tokens}")
    print(f"latency  : {response.metrics.latency_ms} ms\n")
    print(f"invoice_number : {parsed.invoice_number}")
    print(f"vendor_name    : {parsed.vendor_name}")
    print(f"total_amount   : {parsed.total_amount}")
    print(f"line_items     : {[li.raw_item_name for li in parsed.line_items]}")

    ok = parsed.invoice_number == "INV-1001" and parsed.total_amount is not None
    print("\nguided_json constrains output to the schema: OK")
    print("values look right: " + ("OK" if ok else "CHECK -- schema held but content is off"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
