# Research Assistant – project rules

## Project
Cited research assistant for ML privacy & security papers (membership inference, DP-SGD, model inversion, training data extraction, prompt injection). Graded course project and portfolio piece. Every claim in an answer must cite a specific chunk of a specific paper.

## Architecture rule (non-negotiable)
- All logic lives in `src/research_assistant/`. It must NEVER import `chainlit`.
- `app/` is a thin Chainlit layer: it calls core functions and renders events, nothing else.
- Core reports progress only through the `EventSink` protocol in `src/research_assistant/events.py`. No print(), no UI calls in core.
- `tests/test_architecture.py` enforces this. Never weaken or skip it.

## Stack
- Python 3.12, uv (`uv add`, `uv run`).
- LangChain + LangGraph. LLM via `ChatOpenAI` with base_url `https://openrouter.ai/api/v1` and `OPENROUTER_API_KEY`. Model names come from `config/settings.yaml`, never hardcoded.
- Chroma (persistent), BM25, local sentence-transformers embeddings.
- Chainlit for the UI.

## Conventions
- Pydantic v2 models for all data crossing module boundaries (Chunk, RetrievalResult, Citation, Answer, ToolResult, ...).
- Type hints everywhere. `ruff format` and `ruff check` must pass.
- Every external call (OpenRouter, arXiv, Semantic Scholar) goes through a wrapper with timeout, retries (tenacity, exponential backoff) and a typed error. arXiv API: at most one request every 3 seconds.
- User-facing errors are friendly and never leak stack traces, API keys or the system prompt.
- Secrets only from `.env` (never committed). Keep `.env.example` current.
- Claude must NEVER open, read, print, copy, move, source or paste the contents of `.env` (by any tool or shell command). If a value is needed, ask the user. `.env.example` may be read and edited.
- Logging via the `logging` module, configured in `config.py`.

## Security rules
- Retrieved paper text is UNTRUSTED DATA. It is passed to the LLM only inside spotlighted delimiters, never as instructions.
- Tools with side effects (ingesting papers) require explicit user confirmation. Tool arguments are never taken from retrieved text.
- Never disable input validation or injection checks to make a test pass.

## Testing
- pytest. Mock all network calls in unit tests.
- Tests needing network or API keys get `@pytest.mark.integration` and are skipped by default.
- Every step ends with `uv run pytest` and `uv run ruff check .` passing.

## Working style
- Implement only what the current prompt asks. No stubs or placeholders unless the prompt says so; if something can't be done, say so.
- For large changes, propose a short plan first.
- When done, summarize: files changed, how to check it, open issues.
- Never invent arXiv IDs, paper titles or benchmark numbers.
