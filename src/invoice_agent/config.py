"""Tunables. Thresholds live here, never inline in component code.

Every value is env-overridable with the ``INVOICE_AGENT_`` prefix, so dropping
the approval threshold to $500 mid-demo is a flag, not an edit:

    INVOICE_AGENT_APPROVAL_THRESHOLD_USD=500 invoice-agent --invoice-path=...

Nested values use a double underscore:

    INVOICE_AGENT_STAGES__EXTRACT__MODEL=openai/gpt-oss-120b

Importing this module loads ``.env`` into the process environment, which is how
provider API keys reach the adapters.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider API keys are read from os.environ by the adapters, and env_file below
# only feeds this Settings object -- it never touches the process environment.
# Real environment variables still win; .env only fills the gaps.
load_dotenv(override=False)

StructuredOutputMode = Literal["nvext_guided_json", "json_schema", "tool_use"]


class ProviderProfile(BaseModel):
    """Everything provider-specific, in one object.

    Adding a provider means adding a profile here plus its adapter in llm/.
    No node, prompt, or test changes -- they only ever see LLMClient.
    """

    name: str
    base_url: str
    api_key_env: str
    structured_output_mode: StructuredOutputMode
    supports_prompt_cache: bool


PROVIDERS: dict[str, ProviderProfile] = {
    "nvidia": ProviderProfile(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        # NIM does not accept response_format=json_schema for LLMs. It wants
        # extra_body={"nvext": {"guided_json": ...}}. See llm/nvidia.py.
        structured_output_mode="nvext_guided_json",
        # Free tier reports no cached tokens and documents no prompt caching.
        # The batch runner reads this flag to decide whether to warm the cache
        # before fanning out; False means fan out immediately.
        supports_prompt_cache=False,
    ),
}


class StageLLMConfig(BaseModel):
    """Per-stage model choice. The critic can run a stronger model than the
    extractor without either node knowing."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096


_EXTRACTION_MODEL = "meta/llama-3.3-70b-instruct"
_REASONING_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INVOICE_AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    # --- provider ---------------------------------------------------------
    provider: str = "nvidia"

    stages: dict[str, StageLLMConfig] = Field(
        default_factory=lambda: {
            # Deterministic transcription: temperature 0.
            "extract": StageLLMConfig(model=_EXTRACTION_MODEL, temperature=0.0),
            "repair": StageLLMConfig(model=_EXTRACTION_MODEL, temperature=0.0),
            # Judgement calls: a reasoning-tuned model, still low temperature
            # because we want the rationale reproducible in a demo.
            "recommend": StageLLMConfig(model=_REASONING_MODEL, temperature=0.2),
            "critique": StageLLMConfig(model=_REASONING_MODEL, temperature=0.2),
        }
    )

    request_timeout_s: int = 60
    # Schema validation fails -> retry this many times -> needs_human_review.
    max_llm_retries: int = 1

    # --- approval ---------------------------------------------------------
    approval_threshold_usd: Decimal = Decimal("10000")

    # --- canonicalization -------------------------------------------------
    # Score at or above this with a clear margin over the runner-up -> `fuzzy`.
    fuzzy_accept_cutoff: float = 88.0
    # Below this, an item is not even offered to repair as a candidate.
    fuzzy_candidate_cutoff: float = 60.0
    # Winner must beat the runner-up by this much, else `unresolved`.
    # Without it, WidgetA vs WidgetB both score high against "WidgetC".
    fuzzy_margin: float = 8.0
    # How many candidates repair gets to choose between.
    fuzzy_top_k: int = 3

    # --- repair -----------------------------------------------------------
    # A retry loop that retries until success will eventually manufacture
    # success. One round, capped.
    repair_round_cap: int = 1

    # --- validation -------------------------------------------------------
    catalogue_currency: str = "USD"
    # Total vs sum-of-lines tolerance, in currency units.
    total_mismatch_tolerance: Decimal = Decimal("0.01")

    # --- storage ----------------------------------------------------------
    db_path: str = "invoices.db"
    seed_dir: str = "data/seed"

    @property
    def profile(self) -> ProviderProfile:
        try:
            return PROVIDERS[self.provider]
        except KeyError:
            known = ", ".join(sorted(PROVIDERS))
            raise ValueError(f"unknown provider {self.provider!r}; known: {known}") from None

    def stage(self, name: str) -> StageLLMConfig:
        try:
            return self.stages[name]
        except KeyError:
            known = ", ".join(sorted(self.stages))
            raise ValueError(f"no LLM config for stage {name!r}; configured: {known}") from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Call ``get_settings.cache_clear()`` in tests that
    manipulate the environment."""
    return Settings()
