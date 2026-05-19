# Development Guide

This guide explains how to run and extend the Scientific Paper Writing Agent System locally.

## Requirements

- Python 3.10 or later
- A modern browser
- No required Python package installation for the current local server

## Run locally

From the repository root:

```bash
python backend/server.py
```

Open the frontend:

```text
http://127.0.0.1:8080/
```

## Main backend files

- `backend/server.py`: local HTTP server and API routes
- `agents/*.md`: agent instructions
- `workflows/PAPER_WORKFLOW.md`: end-to-end writing workflow
- `templates/*.md`: reusable manuscript templates
- `checklists/*.md`: final quality checks

## Main frontend files

- `frontend/index.html`: page structure
- `frontend/styles.css`: visual styling
- `frontend/app.js`: browser logic and API calls

## Development workflow

1. Update or add agent instructions in `agents/`.
2. Update shared workflow rules in `workflows/`.
3. Add reusable output formats in `templates/`.
4. Test through the local frontend.
5. Review generated output for evidence linkage and unsupported claims.

## Quality principles

- Do not fabricate citations, methods, datasets, metrics, or statistical tests.
- Mark missing evidence clearly.
- Keep scientific writing cautious and traceable.
- Separate observed results from interpretation.
- Preserve consistent terminology across all sections.
