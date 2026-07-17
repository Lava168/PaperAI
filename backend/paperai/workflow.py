from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .catalog import AGENTS, CHECKPOINT_STEPS, STEP_TASKS, WORKFLOWS
from .config import ROOT, Settings
from .db import utcnow
from .provider import ModelProvider
from .repository import Repository


SYSTEM_RULES = """You are part of PaperAI, an evidence-first scientific writing system.
Never invent citations, methods, datasets, sample sizes, metrics, statistical tests, or numerical results.
Distinguish observation, interpretation, and speculation. Use cautious scientific language.
Mark unsupported claims [EVIDENCE NEEDED], missing references [CITATION NEEDED], and missing methodological details [METHOD DETAIL NEEDED].
Return manuscript-ready Markdown and preserve exact terminology and values from the source package."""


class WorkflowEngine:
    def __init__(self, repository: Repository, provider: ModelProvider, settings: Settings):
        self.repository = repository
        self.provider = provider
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="paperai-run")
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def start(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._active:
                return
            self._active.add(run_id)
        self.executor.submit(self._execute_safely, run_id)

    def _execute_safely(self, run_id: str) -> None:
        try:
            self.execute(run_id)
        except Exception as exc:
            self.repository.update_run(run_id, status="failed", error=str(exc))
            self.repository.db.event(run_id, "run.failed", str(exc), {"trace": traceback.format_exc(limit=8)})
        finally:
            with self._lock:
                self._active.discard(run_id)

    def execute(self, run_id: str) -> None:
        run = self.repository.require_run(run_id)
        if run["status"] in {"completed", "cancelled"}:
            return
        project = self.repository.require_project(run["project_id"])
        documents = self.repository.list_documents(project["id"], include_text=True)
        source_package = self._source_package(project, documents)
        completed = {artifact["step"] for artifact in self.repository.artifacts(run_id)}
        steps = WORKFLOWS[run["workflow"]]
        self.repository.update_run(run_id, status="running", error=None)
        self.repository.db.event(run_id, "run.started", "Workflow started")

        for index, (step, agent_id) in enumerate(steps):
            if step in completed:
                continue
            self.repository.update_run(run_id, current_step=step, progress=int(index / len(steps) * 100))
            self.repository.db.event(run_id, "step.started", f"{AGENTS[agent_id].name} started {step}", {"step": step, "agent": agent_id})
            context = self._artifact_context(run_id)
            prompt = self._build_prompt(project, step, agent_id, source_package, context)
            if run["execution_mode"] == "prompt":
                content = f"# Prompt package: {step}\n\n{prompt}"
            else:
                content = self.provider.complete(SYSTEM_RULES + "\n\n" + AGENTS[agent_id].instruction(), prompt)
            self.repository.save_artifact(run_id, step, agent_id, content, {"model": run["model"]})
            self.repository.db.event(run_id, "step.completed", f"Completed {step}", {"step": step})

            if run["human_review"] and step in CHECKPOINT_STEPS and index < len(steps) - 1:
                self.repository.update_run(run_id, status="waiting_approval", progress=int((index + 1) / len(steps) * 100))
                self.repository.db.event(run_id, "run.waiting_approval", f"Approval required after {step}", {"step": step})
                return

        self.repository.update_run(run_id, status="completed", current_step=None, progress=100, completed_at=utcnow())
        self.repository.db.event(run_id, "run.completed", "Workflow completed")

    def resume(self, run_id: str, approved: bool, feedback: str) -> None:
        run = self.repository.require_run(run_id)
        if run["status"] != "waiting_approval":
            raise ValueError("Run is not waiting for approval")
        if not approved:
            self.repository.update_run(run_id, status="cancelled", completed_at=utcnow())
            self.repository.db.event(run_id, "run.cancelled", feedback or "Run rejected at checkpoint")
            return
        if feedback.strip():
            self.repository.save_artifact(run_id, f"feedback_{run['current_step']}", "human", feedback.strip(), {"kind": "human_feedback"})
        self.repository.db.event(run_id, "run.resumed", "Checkpoint approved", {"feedback": feedback})
        self.repository.update_run(run_id, status="queued")
        self.start(run_id)

    def _source_package(self, project: dict[str, Any], documents: list[dict[str, Any]]) -> str:
        sections = [
            "# Project brief",
            f"Title: {project['title']}",
            f"Topic: {project['topic'] or '[NOT PROVIDED]'}",
            f"Paper type: {project['paper_type']}",
            f"Target venue/style: {project['venue'] or '[NOT PROVIDED]'}",
            f"Output language: {project['language']}",
            f"Project instructions: {project['instructions'] or '[NONE]'}",
            "\n# Source materials",
        ]
        remaining = self.settings.max_source_chars
        for document in documents:
            header = f"\n## {document['filename']}\n"
            text = document.get("extracted_text", "")
            excerpt = text[: max(0, remaining - len(header))]
            sections.append(header + (excerpt or "[NO EXTRACTABLE TEXT]"))
            remaining -= len(header) + len(excerpt)
            if remaining <= 0:
                sections.append("\n[SOURCE PACKAGE TRUNCATED]" )
                break
        if not documents:
            sections.append("[NO SOURCE FILES UPLOADED]")
        return "\n".join(sections)

    def _artifact_context(self, run_id: str) -> str:
        artifacts = self.repository.artifacts(run_id)
        if not artifacts:
            return "[NO PRIOR ARTIFACTS]"
        blocks = []
        budget = 80000
        for artifact in artifacts:
            block = f"## Prior artifact: {artifact['step']}\n{artifact['content']}\n"
            blocks.append(block[:budget])
            budget -= len(block)
            if budget <= 0:
                break
        return "\n".join(blocks)

    def _build_prompt(self, project: dict[str, Any], step: str, agent_id: str, sources: str, context: str) -> str:
        workflow_rules = (ROOT / "workflows/PAPER_WORKFLOW.md").read_text(encoding="utf-8")
        return f"""# Current task
Step: {step}
Agent: {AGENTS[agent_id].name}
Objective: {STEP_TASKS[step]}

# Required output
Produce a self-contained Markdown artifact for this step. Explicitly list evidence gaps and decisions that require researcher confirmation.

{sources}

# Outputs from earlier steps
{context}

# Shared workflow reference
{workflow_rules}
"""
