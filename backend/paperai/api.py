from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .catalog import AGENTS, WORKFLOWS
from .config import ROOT, Settings, load_settings
from .db import Database
from .exporter import combined_markdown, markdown_to_docx, markdown_to_latex
from .literature import CrossrefClient
from .materials import extract_text, guessed_media_type, safe_filename, sha256_bytes
from .provider import ModelProvider, OpenAICompatibleProvider
from .repository import Repository
from .schemas import LegacyAgentRequest, ProjectCreate, ProjectUpdate, ResumeRun, RunCreate
from .workflow import SYSTEM_RULES, WorkflowEngine


def create_app(settings: Settings | None = None, provider: ModelProvider | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    repository = Repository(database)
    model_provider = provider or OpenAICompatibleProvider(settings)
    engine = WorkflowEngine(repository, model_provider, settings)
    literature = CrossrefClient()

    app = FastAPI(title="PaperAI", version=__version__, description="Evidence-first autonomous scientific writing agent")
    app.state.settings = settings
    app.state.repository = repository
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def handle_error(exc: Exception) -> HTTPException:
        if isinstance(exc, KeyError):
            return HTTPException(404, str(exc).strip("'"))
        if isinstance(exc, ValueError):
            return HTTPException(400, str(exc))
        return HTTPException(500, str(exc))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "paperai",
            "version": __version__,
            "modelConfigured": settings.model_configured,
            "model": model_provider.model_name,
            "workflows": list(WORKFLOWS),
        }

    @app.get("/api/agents")
    def agents() -> dict[str, Any]:
        return {"ok": True, "agents": [{"id": item.id, "name": item.name, "use": item.purpose} for item in AGENTS.values()]}

    @app.get("/api/literature/search")
    def search_literature(q: str = Query(min_length=2, max_length=500), rows: int = Query(8, ge=1, le=20)) -> dict[str, Any]:
        try:
            return {"ok": True, "items": literature.search(q, rows)}
        except Exception as exc:
            raise HTTPException(502, f"Crossref search failed: {exc}") from exc

    @app.get("/api/literature/doi/{doi:path}")
    def resolve_doi(doi: str) -> dict[str, Any]:
        try:
            return {"ok": True, "item": literature.resolve(doi)}
        except Exception as exc:
            raise HTTPException(502, f"DOI lookup failed: {exc}") from exc

    @app.post("/api/projects", status_code=201)
    def create_project(payload: ProjectCreate) -> dict[str, Any]:
        return repository.create_project(payload.model_dump())

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return repository.list_projects()

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        try:
            project = repository.require_project(project_id)
            project["documents"] = repository.list_documents(project_id)
            project["runs"] = repository.list_runs(project_id)
            return project
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.patch("/api/projects/{project_id}")
    def update_project(project_id: str, payload: ProjectUpdate) -> dict[str, Any]:
        try:
            return repository.update_project(project_id, payload.model_dump(exclude_unset=True))
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.delete("/api/projects/{project_id}", status_code=204)
    def delete_project(project_id: str) -> None:
        try:
            repository.delete_project(project_id)
            directory = settings.data_dir / "projects" / project_id
            if directory.exists():
                shutil.rmtree(directory)
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.post("/api/projects/{project_id}/documents", status_code=201)
    async def upload_document(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            repository.require_project(project_id)
            filename = safe_filename(file.filename or "source")
            content = await file.read()
            if not content:
                raise ValueError("Uploaded file is empty")
            if len(content) > 50 * 1024 * 1024:
                raise ValueError("File exceeds the 50 MB limit")
            digest = sha256_bytes(content)
            directory = settings.data_dir / "projects" / project_id / "sources"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{digest[:12]}-{filename}"
            path.write_bytes(content)
            status, text = "ready", ""
            try:
                text = extract_text(path, file.content_type)
            except Exception as extraction_error:
                status = f"error: {str(extraction_error)[:300]}"
            return repository.add_document(
                {
                    "project_id": project_id,
                    "filename": filename,
                    "media_type": file.content_type or guessed_media_type(filename),
                    "size": len(content),
                    "sha256": digest,
                    "stored_path": str(path),
                    "extracted_text": text,
                    "extraction_status": status,
                }
            )
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.get("/api/projects/{project_id}/documents")
    def list_documents(project_id: str) -> list[dict[str, Any]]:
        try:
            repository.require_project(project_id)
            return repository.list_documents(project_id)
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.delete("/api/projects/{project_id}/documents/{document_id}", status_code=204)
    def delete_document(project_id: str, document_id: str) -> None:
        try:
            item = repository.delete_document(project_id, document_id)
            path = Path(item["stored_path"])
            if path.exists() and settings.data_dir in path.parents:
                path.unlink()
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.post("/api/projects/{project_id}/runs", status_code=202)
    def create_run(project_id: str, payload: RunCreate) -> dict[str, Any]:
        try:
            if payload.execution_mode == "model" and not settings.model_configured and provider is None:
                raise ValueError("Model is not configured. Set PAPERAI_MODEL_API_KEY or choose prompt mode.")
            run = repository.create_run(project_id, payload.workflow, model_provider.model_name, payload.execution_mode, payload.human_review)
            engine.start(run["id"])
            return run
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.get("/api/projects/{project_id}/runs")
    def list_runs(project_id: str) -> list[dict[str, Any]]:
        try:
            repository.require_project(project_id)
            return repository.list_runs(project_id)
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            run = repository.require_run(run_id)
            run["artifacts"] = repository.artifacts(run_id)
            return run
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.get("/api/runs/{run_id}/events")
    def get_events(run_id: str, after: int = Query(0, ge=0)) -> list[dict[str, Any]]:
        try:
            repository.require_run(run_id)
            return repository.events(run_id, after)
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str, payload: ResumeRun) -> dict[str, Any]:
        try:
            engine.resume(run_id, payload.approved, payload.feedback)
            return repository.require_run(run_id)
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.get("/api/runs/{run_id}/export")
    def export_run(run_id: str, format: str = Query("md", pattern="^(md|docx|tex)$")):
        try:
            run = repository.require_run(run_id)
            project = repository.require_project(run["project_id"])
            content = combined_markdown(project, repository.artifacts(run_id))
            stem = safe_filename(project["title"]).rsplit(".", 1)[0] or "manuscript"
            export_dir = settings.data_dir / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            if format == "md":
                return PlainTextResponse(content, headers={"Content-Disposition": f'attachment; filename="{stem}.md"'})
            if format == "tex":
                return PlainTextResponse(markdown_to_latex(content), media_type="application/x-tex", headers={"Content-Disposition": f'attachment; filename="{stem}.tex"'})
            path = export_dir / f"{run_id}-{stem}.docx"
            markdown_to_docx(content, path)
            return FileResponse(path, filename=f"{stem}.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as exc:
            raise handle_error(exc) from exc

    @app.post("/api/run-agent")
    def legacy_run(payload: LegacyAgentRequest) -> dict[str, Any]:
        try:
            if payload.agentId not in AGENTS:
                raise ValueError(f"Unknown agentId: {payload.agentId}")
            agent = AGENTS[payload.agentId]
            prompt = f"""Title: {payload.title}\nTopic: {payload.topic}\nVenue: {payload.venue}\nPaper type: {payload.paperType}\nTask: {payload.task}\nMaterials: {payload.materials}\nDesired output: {payload.desiredOutput}"""
            output = None
            if payload.executeModel:
                output = model_provider.complete(SYSTEM_RULES + "\n\n" + agent.instruction(), prompt)
            return {"ok": True, "agent": {"id": agent.id, "name": agent.name}, "prompt": prompt, "modelOutput": output, "downloadContent": output or prompt}
        except Exception as exc:
            raise handle_error(exc) from exc

    frontend = ROOT / "frontend"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
