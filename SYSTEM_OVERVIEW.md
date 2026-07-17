# System Overview

PaperAI 2.0 is a standalone scientific writing service with five layers.

## Architecture

```text
Browser workspace / API client
              ↓
        FastAPI service
              ↓
Project repository ── Material extractors ── Crossref client
              ↓
       Workflow engine
              ↓
OpenAI-compatible model provider
              ↓
 SQLite state + source files + exports
```

## Persistent objects

- **Project**: paper brief, venue, type, output language, and immutable constraints.
- **Document**: original source path, checksum, media type, extraction status, and extracted text.
- **Run**: workflow, execution mode, model, state, progress, and checkpoint policy.
- **Artifact**: the complete Markdown output from one agent step.
- **Event**: timestamped workflow lifecycle information for monitoring and debugging.

## Workflow state machine

Runs move through `queued`, `running`, `waiting_approval`, `completed`, `failed`, or `cancelled`. Each completed artifact is stored before the next task starts, so a process can resume without regenerating prior work.

The full workflow contains intake, claim–evidence mapping, outline, literature, methods, results, figures and tables, discussion, citation audit, peer review, and final integration. Optional approval gates follow outline and peer review.

## Safety rules

- Never manufacture citations, methods, datasets, sample sizes, metrics, tests, or results.
- Keep source excerpts and prior artifacts visible to each downstream agent.
- Use explicit gap markers instead of silently filling missing information.
- Treat Crossref results as bibliographic metadata, not proof that a paper supports a claim.
- Keep human feedback as a first-class artifact visible to subsequent agents.

## Deployment boundary

The default service binds to localhost and has no authentication. This is appropriate for personal local use. Public or team deployment requires an identity layer, access controls, encrypted secrets, TLS, backups, request limits, and an approved data-retention policy.
