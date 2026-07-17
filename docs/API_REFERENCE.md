# API Reference

Base URL: `http://127.0.0.1:8080`. Interactive OpenAPI documentation is served at `/docs`.

## Create a project

```http
POST /api/projects
Content-Type: application/json

{
  "title": "A robust scientific agent",
  "topic": "Evaluation of evidence-grounded manuscript generation",
  "venue": "Target journal",
  "paper_type": "method",
  "language": "English",
  "instructions": "Do not change the registered primary outcome."
}
```

## Upload a source

```bash
curl -F 'file=@results.xlsx' http://127.0.0.1:8080/api/projects/PROJECT_ID/documents
```

Supported extensions are PDF, DOCX, XLSX/XLSM, CSV/TSV, PPTX, Markdown, text, TeX, JSON, YAML, XML, HTML, Python, and R. The per-file limit is 50 MB.

## Start a workflow

```http
POST /api/projects/PROJECT_ID/runs
Content-Type: application/json

{
  "workflow": "full_paper",
  "execution_mode": "model",
  "human_review": true
}
```

Available workflows: `full_paper`, `outline`, `methods`, `results`, and `review`. Use `prompt` execution mode when a model key is unavailable.

The endpoint returns `202 Accepted`. Poll `GET /api/runs/RUN_ID` or fetch incremental events from `GET /api/runs/RUN_ID/events?after=EVENT_ID`.

## Resume a checkpoint

```http
POST /api/runs/RUN_ID/resume
Content-Type: application/json

{
  "approved": true,
  "feedback": "Keep the novelty claim conservative and add the ablation limitation."
}
```

Sending `approved: false` moves the run to `cancelled`.

## Export

```text
GET /api/runs/RUN_ID/export?format=md
GET /api/runs/RUN_ID/export?format=docx
GET /api/runs/RUN_ID/export?format=tex
```

If a `final_manuscript` artifact exists it is exported directly. Otherwise all relevant artifacts are combined into a review bundle.

## Literature metadata

```text
GET /api/literature/search?q=evidence-grounded+scientific+writing&rows=8
GET /api/literature/doi/10.1000/example
```

These endpoints retrieve Crossref metadata. Metadata resolution confirms that a bibliographic record exists; it does not establish that the work supports a particular manuscript claim.

## Compatibility endpoint

The original `POST /api/run-agent` endpoint remains available for simple single-agent prompt/model calls. New integrations should use projects and runs.
