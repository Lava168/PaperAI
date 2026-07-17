# Development Guide

## Requirements

- Python 3.10+
- SQLite 3
- A modern browser

## Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python backend/server.py
```

The application, REST API, and OpenAPI documentation are served from the same process at `/`, `/api`, and `/docs`.

## Backend modules

| Module | Responsibility |
| --- | --- |
| `api.py` | HTTP routes, uploads, exports, static frontend |
| `config.py` | Environment-derived immutable settings |
| `db.py` | SQLite schema and transaction boundary |
| `repository.py` | Project, document, run, artifact, and event persistence |
| `materials.py` | Safe filenames and research-file text extraction |
| `provider.py` | OpenAI-compatible model calls |
| `workflow.py` | Background execution, checkpoints, resume, prompt context |
| `catalog.py` | Agent registry and workflow definitions |
| `literature.py` | Crossref search and DOI lookup |
| `exporter.py` | Markdown, Word, and LaTeX output |

## Adding a workflow

Add the workflow sequence to `WORKFLOWS` in `catalog.py`. Each tuple contains a step identifier and an agent identifier. Add a task description to `STEP_TASKS`; the engine will persist, resume, monitor, and expose the new workflow automatically.

## Testing

Tests use a temporary database and a deterministic fake model provider:

```bash
pytest -q
ruff check backend tests
```

Model-backed integration tests should use a separate opt-in marker and must never require a real API key in CI.

## Production notes

The included thread executor is intended for one local service process. For multi-instance production deployment, replace it with a durable queue such as Celery, Dramatiq, or a managed job system, and use PostgreSQL/object storage behind the same repository interface.
