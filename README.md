# PaperAI

PaperAI is a standalone, evidence-first scientific writing agent. It turns research files into traceable manuscript artifacts through a persistent multi-agent workflow rather than a single prompt.

## What it does

- Creates persistent paper projects backed by SQLite
- Extracts text from PDF, DOCX, XLSX, CSV, TSV, PPTX, Markdown, text, and source files
- Coordinates nine specialist agents across resumable workflows
- Preserves every intermediate artifact and run event
- Stops after outline and peer review for optional human approval
- Flags unsupported content as `[EVIDENCE NEEDED]`, `[CITATION NEEDED]`, or `[METHOD DETAIL NEEDED]`
- Searches Crossref metadata and resolves DOI records
- Works with Qwen, OpenAI, or another OpenAI-compatible chat-completions API
- Exports Markdown, Word, and LaTeX
- Provides a local browser workspace and documented REST API

## Agent workflow

```text
Research files
    ↓
Intake → Claim–Evidence Map → Outline → Literature
    ↓
Methods → Results → Figures & Tables → Discussion
    ↓
Citation Audit → Peer Review → Final Manuscript
```

The full workflow uses the Orchestrator, Literature, Outline, Methods, Results, Figure & Table, Discussion, Citation, and Reviewer agents. All generated steps are stored separately and supplied as context to later steps.

## Quick start

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Load the variables from `.env` in your shell, set `PAPERAI_MODEL_API_KEY`, then run:

```bash
python backend/server.py
```

Open `http://127.0.0.1:8080`. API documentation is available at `http://127.0.0.1:8080/docs`.

If no model key is configured, choose **Prompt only** in the interface. PaperAI will execute the same workflow and export self-contained prompt packages for another model or coding agent.

## Model configuration

PaperAI uses the OpenAI-compatible `POST /chat/completions` protocol:

```dotenv
PAPERAI_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
PAPERAI_MODEL_API_KEY=your-key
PAPERAI_MODEL=qwen-plus
```

Aliases for the original `DASHSCOPE_API_KEY`, `QWEN_BASE_URL`, and `QWEN_MODEL` variables remain supported.

## Data and privacy

Project metadata, extracted text, run state, and outputs are stored under `data/` by default. Uploaded files are not sent anywhere except the configured model API as part of the run context. Crossref endpoints only receive explicit literature search queries or DOI lookups.

For sensitive or unpublished research, configure a trusted private model endpoint and protect the service behind authentication before exposing it beyond localhost.

## API summary

| Endpoint | Purpose |
| --- | --- |
| `POST /api/projects` | Create a paper project |
| `POST /api/projects/{id}/documents` | Upload and extract a source file |
| `POST /api/projects/{id}/runs` | Start a workflow |
| `GET /api/runs/{id}` | Read status and artifacts |
| `POST /api/runs/{id}/resume` | Approve or reject a checkpoint |
| `GET /api/runs/{id}/export?format=docx` | Export the manuscript |
| `GET /api/literature/search?q=...` | Search Crossref metadata |
| `GET /api/literature/doi/{doi}` | Resolve a DOI |

See `docs/API_REFERENCE.md` for request examples.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check backend tests
```

## Docker

```bash
docker compose up --build
```

The container stores persistent state in the `paperai-data` volume and serves PaperAI on port 8080.

## Deploy to Render

The repository includes a production-oriented `render.yaml` Blueprint. It creates a Docker web service, checks `/api/health`, and mounts a 1 GB persistent disk at `/app/data` so projects, uploads, and SQLite state survive restarts.

1. Push the repository to GitHub.
2. In Render, choose **New → Blueprint** and connect this repository.
3. Enter `PAPERAI_MODEL_API_KEY` when Render asks for the unsynced secret.
4. Apply the Blueprint and wait for the health check to pass.

The Blueprint uses a paid Starter web service because Render does not support persistent disks on Free web services. Use a private or trusted OpenAI-compatible model endpoint for unpublished research.

## License

MIT
