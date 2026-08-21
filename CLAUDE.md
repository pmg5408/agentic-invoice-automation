# Working in this repo

Design lives in `docs/`. Read `contracts.md` first, then your own section of
`components.md`, then `decisions.md`. Skip the rest. This file is operations only.

## Commands

```bash
make setup   # uv venv on 3.14 + editable install
make test    # full suite, no API key needed, must stay green
make lint    # ruff
make seed    # rebuild invoices.db from data/seed/
```

Python is **3.14** — PEP 695 generics (`def f[T: BaseModel]`), `StrEnum` and
`datetime.UTC` are all available and preferred. Ruff enforces `py314`.

## Ground rules

- **`models.py` is frozen.** It is the transcription of `contracts.md` and every
  component imports it. If your component needs a shape that isn't there, say so
  and stop — do not add a field in passing, and do not define a local variant.
- **Nodes get `Deps`, never globals.** Implement `make_<node>(deps) -> NodeFn` and
  read `deps.settings` / `deps.repo` / `deps.llm` / `deps.logger`. Test a node by
  calling it directly; no LangGraph needed.
- **Return partial state updates, and rebuild lists rather than mutating:**
  `return {"validations": run.validations + [report]}`. Artifacts are frozen.
- **Thresholds come from `deps.settings`.** No magic numbers in component code.

## Testing against an LLM

Only `extract`, `repair`, `recommend` and `critique` call a model. Everything else
constructs `ExtractedInvoice` / `Finding` / `PolicyGate` literals directly.

```python
from invoice_agent.llm.stub import stub_client
stub_client([extraction])                # happy path
stub_client(['{"bad":', extraction])     # malformed, then valid -> exercises retry
```

A model instance is serialized for you; a raw string drives the parse/validate/retry
path in `llm/client.py`. `conftest.py` has `deps`, `repo`, `stub_llm`, `make_extraction`,
`make_run`, `logger` — use them rather than standing up your own database.

Real calls go through `deps.llm.complete(request, SomeModel)`. Never build a raw
provider request; NIM's `nvext.guided_json` quirk is the adapter's problem, not yours.
