"""What every node is handed.

Nodes are built by factories that close over Deps, so a component test calls
the node directly with a fake Deps -- no LangGraph knowledge required to
implement or test one node.

    def test_validate_aggregates_quantities(deps, make_extraction):
        node = make_validate(deps)
        result = node(run)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from invoice_agent.config import Settings
from invoice_agent.llm.client import LLMClient
from invoice_agent.models import InventoryItem, InvoiceRepository, InvoiceRun
from invoice_agent.obs import JsonLogger

# A node reads the run and returns a partial state update, LangGraph style.
# Return only the keys you changed.
NodeFn = Callable[[InvoiceRun], dict[str, Any]]


@dataclass(frozen=True)
class Deps:
    settings: Settings
    repo: InvoiceRepository
    llm: LLMClient
    logger: JsonLogger

    def inventory(self) -> dict[str, InventoryItem]:
        """Convenience: the catalogue, keyed by canonical item name."""
        return self.repo.load_inventory()
