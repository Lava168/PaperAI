#!/usr/bin/env python3
"""Local backend for the Scientific Paper Writing Agent System.

This server has no third-party dependencies. It serves the frontend and exposes
API endpoints that read the agent system files, build a Cursor-ready task
package, and save each run under runs/.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
RUNS = ROOT / "runs"
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.6-max-preview")

AGENTS = {
    "orchestrator": {
        "name": "Orchestrator Agent",
        "file": "agents/ORCHESTRATOR_AGENT.md",
        "use": "Coordinate a full manuscript from materials to revision plan.",
    },
    "literature": {
        "name": "Literature Agent",
        "file": "agents/LITERATURE_AGENT.md",
        "use": "Frame related work, research gaps, and citation needs.",
    },
    "outline": {
        "name": "Outline Agent",
        "file": "agents/OUTLINE_AGENT.md",
        "use": "Create titles, abstract plans, contributions, and section structure.",
    },
    "methods": {
        "name": "Methods Agent",
        "file": "agents/METHODS_AGENT.md",
        "use": "Write reproducible Methods from datasets, code, configs, and protocols.",
    },
    "results": {
        "name": "Results Agent",
        "file": "agents/RESULTS_AGENT.md",
        "use": "Write Results from figures, tables, metrics, and statistical tests.",
    },
    "figure_table": {
        "name": "Figure Table Agent",
        "file": "agents/FIGURE_TABLE_AGENT.md",
        "use": "Write standalone captions, table titles, and table notes.",
    },
    "discussion": {
        "name": "Discussion Agent",
        "file": "agents/DISCUSSION_AGENT.md",
        "use": "Write Discussion, limitations, future work, and conclusion.",
    },
    "citation": {
        "name": "Citation Agent",
        "file": "agents/CITATION_AGENT.md",
        "use": "Check citations, missing references, and novelty claim risk.",
    },
    "reviewer": {
        "name": "Reviewer Agent",
        "file": "agents/REVIEWER_AGENT.md",
        "use": "Review a manuscript like a critical peer reviewer.",
    },
}

REFERENCE_FILES = [
    "workflows/PAPER_WORKFLOW.md",
    "templates/CLAIM_EVIDENCE_MAP.md",
    "templates/MANUSCRIPT_TEMPLATE.md",
    "templates/FIGURE_TABLE_TEMPLATE.md",
    "checklists/SUBMISSION_QUALITY_CHECKLIST.md",
]


def read_text(relative_path: str) -> str:
    path = (ROOT / relative_path).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise ValueError(f"Refusing to read outside project: {relative_path}")
    return path.read_text(encoding="utf-8")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return value.strip("-") or "run"


def build_prompt(payload: dict[str, Any], agent: dict[str, str]) -> str:
    project_path = payload.get("projectPath") or "[PROJECT_PATH]"
    title = payload.get("title") or "[PAPER_TITLE]"
    topic = payload.get("topic") or "[PAPER_TOPIC]"
    venue = payload.get("venue") or "[TARGET_VENUE_OR_STYLE]"
    paper_type = payload.get("paperType") or "[PAPER_TYPE]"
    desired_output = payload.get("desiredOutput") or "Full manuscript draft"
    task = payload.get("task") or "[DESCRIBE_THE_WRITING_TASK]"
    materials = payload.get("materials") or "[LIST_FILES_FIGURES_TABLES_NOTES_TO_READ]"

    return f"""Read {ROOT / agent["file"]}.
Also read {ROOT / "workflows/PAPER_WORKFLOW.md"} and {ROOT / "templates/CLAIM_EVIDENCE_MAP.md"}.

Operate as the {agent["name"]}.

Project or manuscript path:
{project_path}

Paper title:
{title}

Paper topic:
{topic}

Paper type:
{paper_type}

Target venue or writing style:
{venue}

Desired output:
{desired_output}

Task:
{task}

Materials, figures, tables, notes, or files to inspect:
{materials}

Output requirements:
1. Write in English unless I explicitly ask otherwise.
2. Do not invent citations, methods, data, metrics, or statistical tests.
3. Mark missing support as [EVIDENCE NEEDED], [CITATION NEEDED], or [METHOD DETAIL NEEDED].
4. Keep claims linked to specific evidence.
5. Generate manuscript-ready content for the requested title/topic.
6. Provide a concise revision checklist after the draft output.
"""


def build_run_package(payload: dict[str, Any]) -> dict[str, Any]:
    agent_id = payload.get("agentId") or "orchestrator"
    if agent_id not in AGENTS:
        raise ValueError(f"Unknown agentId: {agent_id}")

    agent = AGENTS[agent_id]
    agent_text = read_text(agent["file"])
    references = {path: read_text(path) for path in REFERENCE_FILES}
    prompt = build_prompt(payload, agent)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{slugify(agent_id)}"
    run_path = RUNS / f"{run_id}.md"

    model_output = None
    if payload.get("executeModel"):
        model_output = call_qwen(prompt, payload.get("model") or QWEN_MODEL)

    run_markdown = f"""# Agent Run: {agent["name"]}

Run ID: `{run_id}`

Created: `{datetime.now().isoformat(timespec="seconds")}`

## Request

```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```

## Cursor Prompt

```text
{prompt}
```

## Loaded Agent Instruction

```markdown
{agent_text}
```

## Loaded Reference Files

{chr(10).join(f"- `{path}`" for path in references)}
"""
    if model_output:
        run_markdown += f"""

## Qwen Output

```markdown
{model_output}
```
"""

    saved_path = None
    if payload.get("saveToServer"):
        RUNS.mkdir(exist_ok=True)
        run_path.write_text(run_markdown, encoding="utf-8")
        saved_path = str(run_path)

    response = {
        "ok": True,
        "runId": run_id,
        "savedPath": saved_path,
        "agent": {"id": agent_id, **agent},
        "prompt": prompt,
        "downloadContent": model_output or prompt,
        "downloadFilename": f"{run_id}.md",
        "loadedFiles": [agent["file"], *REFERENCE_FILES],
        "message": (
            "Backend generated the agent prompt. The browser can download it locally."
        ),
    }
    if model_output:
        response["model"] = payload.get("model") or QWEN_MODEL
        response["modelOutput"] = model_output
        response["message"] = "Backend called Qwen and returned manuscript content for browser download."
    return response


def call_qwen(prompt: str, model: str) -> str:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing DASHSCOPE_API_KEY. Set it before starting the backend, for example: "
            "export DASHSCOPE_API_KEY='your_key'"
        )

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a rigorous English scientific paper writing agent. "
                    "Follow the provided agent instructions exactly. Do not invent citations, "
                    "methods, metrics, datasets, or statistical tests."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    request = Request(
        f"{QWEN_BASE_URL}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"Qwen API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ValueError(f"Qwen API request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Qwen API response: {data}") from exc


class Handler(BaseHTTPRequestHandler):
    server_version = "PaperAgentBackend/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "system": "scientific-paper-writing-agent-system",
                    "qwenConfigured": bool(
                        os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
                    ),
                    "qwenBaseUrl": QWEN_BASE_URL,
                    "defaultQwenModel": QWEN_MODEL,
                }
            )
            return

        if path == "/api/agents":
            agents = [{"id": key, **value} for key, value in AGENTS.items()]
            self.send_json({"ok": True, "agents": agents})
            return

        self.serve_frontend(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path != "/api/run-agent":
            self.send_json({"ok": False, "error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_json()
            result = build_run_package(payload)
            self.send_json(result)
        except Exception as exc:  # noqa: BLE001 - return useful local API errors
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def serve_frontend(self, request_path: str) -> None:
        if request_path in {"", "/"}:
            file_path = FRONTEND / "index.html"
        else:
            cleaned = unquote(request_path).lstrip("/")
            if cleaned.startswith("frontend/"):
                cleaned = cleaned.removeprefix("frontend/")
            file_path = (FRONTEND / cleaned).resolve()

        if FRONTEND not in file_path.parents and file_path != FRONTEND:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if file_path.is_dir():
            file_path = file_path / "index.html"

        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}")


def main() -> None:
    port = 8080
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving frontend and API at http://127.0.0.1:{port}/")
    print("API endpoints: GET /api/health, GET /api/agents, POST /api/run-agent")
    server.serve_forever()


if __name__ == "__main__":
    main()
