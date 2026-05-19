# PaperAI Local Frontend

This folder contains the local single-page application for the Scientific Paper Writing Agent System.

## What changed

The frontend is now a complete local visual workspace instead of a simple three-column console.

It includes:

- Dashboard overview
- Agent Center
- New Paper Project form
- Manuscript Workspace
- Claim-Evidence Map editor
- Reviewer risk panel
- Browser-session Run History
- Prompt generation, backend run, model run, copy, and download actions

## Start

From the repository root:

```bash
python backend/server.py
```

Then open:

```text
http://127.0.0.1:8080/
```

No login, database, or cloud deployment is required.

## How it works

1. Open the local app.
2. Choose a specialist writing agent in Agent Center.
3. Create a paper project with title, topic, venue, paper type, task, and source materials.
4. Add or edit claim-evidence rows.
5. Generate a structured prompt or call the local backend.
6. Review the output in Workspace.
7. Copy or download the result as Markdown.

## Backend API used by the frontend

```text
GET  /api/health
GET  /api/agents
POST /api/run-agent
```

## Local storage

The frontend uses browser localStorage for:

- Run history
- Claim-evidence map rows

This is intentionally local-only for the first version.

## Frontend files

```text
frontend/
  index.html   # SPA layout
  styles.css   # visual design and responsive layout
  app.js       # local state, navigation, backend calls, claim map, history
  README.md
```

## Design direction

The goal is to make PaperAI feel like a usable local product while keeping the architecture simple:

- No authentication
- No database
- No deployment dependency
- No frontend build step
- Still starts with `python backend/server.py`
