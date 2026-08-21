"""Config is the demo surface: thresholds must be overridable without a code edit."""

from decimal import Decimal

import pytest

from invoice_agent.config import PROVIDERS, Settings


def test_defaults_match_the_docs():
    s = Settings()
    assert s.approval_threshold_usd == Decimal("10000")
    assert s.repair_round_cap == 1
    assert s.catalogue_currency == "USD"


def test_threshold_is_decimal_not_float():
    s = Settings(approval_threshold_usd="0.1")
    assert s.approval_threshold_usd == Decimal("0.1")
    assert isinstance(s.approval_threshold_usd, Decimal)


def test_threshold_overridable_from_env(monkeypatch):
    monkeypatch.setenv("INVOICE_AGENT_APPROVAL_THRESHOLD_USD", "500")
    assert Settings().approval_threshold_usd == Decimal("500")


def test_stage_model_overridable_from_env(monkeypatch):
    monkeypatch.setenv("INVOICE_AGENT_STAGES__EXTRACT__MODEL", "openai/gpt-oss-120b")
    assert Settings().stage("extract").model == "openai/gpt-oss-120b"


def test_every_llm_stage_has_a_model():
    s = Settings()
    for stage in ("extract", "repair", "recommend", "critique"):
        assert s.stage(stage).model


def test_extraction_is_deterministic():
    # Transcription must not vary run to run.
    assert Settings().stage("extract").temperature == 0.0


def test_unknown_stage_names_itself_in_the_error():
    with pytest.raises(ValueError, match="triage"):
        Settings().stage("triage")


def test_unknown_provider_names_itself_in_the_error():
    with pytest.raises(ValueError, match="anthropic"):
        _ = Settings(provider="anthropic").profile


class TestNvidiaProfile:
    def test_asks_for_shape_with_response_format(self):
        assert PROVIDERS["nvidia"].structured_output_mode == "json_schema"

    def test_declares_no_prompt_cache(self):
        # The batch runner reads this to decide whether to warm before fanning out.
        assert PROVIDERS["nvidia"].supports_prompt_cache is False
