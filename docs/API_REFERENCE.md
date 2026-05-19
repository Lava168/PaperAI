# API Reference

The local backend is implemented in `backend/server.py` and serves both the frontend and API routes.

## Base URL

```text
http://127.0.0.1:8080
```

## GET `/api/health`

Returns backend health information.

### Example response

```json
{
  "ok": true,
  "service": "paperai"
}
```

## GET `/api/agents`

Returns the available writing agents and their descriptions.

### Example response

```json
{
  "agents": [
    {
      "id": "orchestrator",
      "name": "Orchestrator Agent",
      "use": "Coordinate a full manuscript from materials to revision plan."
    }
  ]
}
```

## POST `/api/run-agent`

Builds or runs an agent task based on the selected agent and user input.

### Request body

```json
{
  "agentId": "orchestrator",
  "title": "Example paper title",
  "topic": "Example research topic",
  "venue": "Target journal or conference",
  "paperType": "Empirical paper",
  "task": "Draft an outline and claim-evidence map",
  "materials": "List of available figures, tables, notes, and files",
  "desiredOutput": "Markdown outline",
  "mode": "prompt"
}
```

### Important fields

| Field | Description |
|---|---|
| `agentId` | Selected agent key, such as `orchestrator`, `methods`, or `reviewer` |
| `title` | Working manuscript title |
| `topic` | Research topic or study description |
| `venue` | Target venue or writing style |
| `paperType` | Study or manuscript type |
| `task` | Specific task for the agent |
| `materials` | Source materials the user wants the agent to use |
| `desiredOutput` | Expected output format |
| `mode` | Usually `prompt`; model-backed generation can be configured separately |

## Error handling

The backend should return clear JSON errors for invalid agent IDs, malformed input, or unavailable model configuration.

## Design notes

- Keep API responses easy for the frontend to render.
- Avoid storing user materials by default.
- Keep generated outputs traceable to the selected agent and workflow files.
- Prefer Markdown output for portability.
