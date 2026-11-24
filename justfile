# Task runner: `uv run just` lists the recipes. `just` itself is a dev dependency
# (rust-just from PyPI), so a fresh clone only needs uv. Every recipe runs through uv.
#
# Examples:
#   uv run just corpus 5                 # first 5 papers of data/papers.yaml only
#   uv run just app 8001                 # web UI on another port
#   uv run just eval --split dev --strategies hybrid full

default_eval := "--split dev --strategies baseline_dense baseline_dense_tuned hybrid full"
default_verification := "--split dev --strategies baseline_dense_tuned --verification off flag regenerate"

# List the recipes
default:
    @just --list --unsorted

# Install dependencies and create .env from .env.example if it does not exist
setup:
    uv sync
    @if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example: fill in the keys."; else echo ".env exists (not touched)."; fi

# Download the papers in data/papers.yaml from arXiv (1 request / 3 s); limit = first N only
corpus limit="":
    uv run python scripts/build_corpus.py {{ if limit != "" { "--limit " + limit } else { "" } }}

# Parse, scan, chunk and index the downloaded papers (incremental; skips indexed papers)
ingest *args:
    uv run python scripts/ingest.py {{args}}

# Create the SQLite chat-history database
db:
    uv run python scripts/init_db.py

# Check configuration, secrets, strategies and the search index
check:
    uv run python scripts/check_setup.py

# Start the web UI (http://localhost:<port>) after the startup checks
app port="8000": check
    uv run chainlit run app/main.py --port {{port}}

# Linters, then the unit tests (network/API tests are skipped)
test: lint
    uv run pytest

# ruff check + ruff format --check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Offline A/B evaluation of retrieval strategies (paid API calls)
eval *args:
    uv run python eval/run_eval.py {{ if args == "" { default_eval } else { args } }}

# Compare claim-verification modes, scored by the independent models.eval_judge (paid)
eval-verification *args:
    uv run python eval/run_eval.py {{ if args == "" { default_verification } else { args } }}

# Hand-label claims of a run and report judge agreement + flag precision/recall
label run *args:
    uv run python eval/label_claims.py {{run}} {{args}}
