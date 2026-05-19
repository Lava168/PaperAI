# Frontend Console

This is the browser frontend for operating the Scientific Paper Writing Agent System through the local backend.

## Start

From the repository root:

```bash
python backend/server.py
```

Then open:

```text
http://127.0.0.1:8080/
```

## How It Works

The frontend calls the local backend:

1. Select the agent you want to use.
2. Enter the paper title, topic, target venue, paper type, task, and materials.
3. Click `Run Backend`.
4. The backend loads the selected agent file and reference templates.
5. The backend returns a Cursor-ready prompt.
6. The frontend displays the returned Cursor-ready prompt.

You can then copy the prompt into Cursor when you want Cursor to execute the writing task.

## Run Qwen Directly

To call Qwen from the webpage, set your DashScope API key before starting the backend:

```bash
export DASHSCOPE_API_KEY="your_dashscope_key"
export QWEN_MODEL="qwen3.6-max-preview"
python backend/server.py
```

Then click `Run Qwen` in the frontend. The generated manuscript content appears in the webpage and downloads through your browser as a `.md` file. By default, the output is not saved on the server.

Defaults:

```text
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.6-max-preview
```

Alternative model names available in the frontend include `qwen-max-latest`, `qwen3-max`, `qwen-plus-latest`, and `qwen-turbo-latest`.

## Why This Design

This avoids API keys, authentication, remote deployment, and model-provider lock-in. The backend gives you a real local API while keeping the final writing execution under Cursor's control.

## Backend API

```text
GET  /api/health
GET  /api/agents
POST /api/run-agent
```

The current request shape is:

```json
{
  "agentId": "results",
  "projectPath": "/path/to/project",
  "title": "Paper title",
  "topic": "Paper topic",
  "venue": "Nature Medicine",
  "paperType": "Empirical research paper",
  "desiredOutput": "Full manuscript draft",
  "model": "qwen3.6-max-preview",
  "task": "Write the Results section",
  "materials": "paper/figures, outputs/summary.csv",
  "executeModel": true,
  "saveToServer": false
}
```
