from __future__ import annotations

import json
from typing import Any

from .db import Database, new_id, utcnow


class Repository:
    def __init__(self, db: Database):
        self.db = db

    def create_project(self, data: dict[str, Any]) -> dict[str, Any]:
        project_id, now = new_id("prj"), utcnow()
        values = (
            project_id,
            data["title"],
            data.get("topic", ""),
            data.get("venue", ""),
            data.get("paper_type", "empirical"),
            data.get("language", "English"),
            data.get("instructions", ""),
            now,
            now,
        )
        self.db.execute(
            "INSERT INTO projects(id,title,topic,venue,paper_type,language,instructions,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            values,
        )
        return self.require_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        return self.db.all(
            "SELECT p.*, (SELECT count(*) FROM documents d WHERE d.project_id=p.id) AS document_count, "
            "(SELECT count(*) FROM runs r WHERE r.project_id=p.id) AS run_count "
            "FROM projects p ORDER BY p.updated_at DESC"
        )

    def require_project(self, project_id: str) -> dict[str, Any]:
        item = self.db.one("SELECT * FROM projects WHERE id=?", (project_id,))
        if not item:
            raise KeyError("Project not found")
        return item

    def update_project(self, project_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        self.require_project(project_id)
        allowed = {"title", "topic", "venue", "paper_type", "language", "instructions"}
        filtered = {key: value for key, value in changes.items() if key in allowed and value is not None}
        if filtered:
            filtered["updated_at"] = utcnow()
            assignments = ",".join(f"{key}=?" for key in filtered)
            self.db.execute(f"UPDATE projects SET {assignments} WHERE id=?", (*filtered.values(), project_id))
        return self.require_project(project_id)

    def delete_project(self, project_id: str) -> None:
        self.require_project(project_id)
        self.db.execute("DELETE FROM projects WHERE id=?", (project_id,))

    def add_document(self, values: dict[str, Any]) -> dict[str, Any]:
        document_id = new_id("doc")
        self.db.execute(
            "INSERT INTO documents(id,project_id,filename,media_type,size,sha256,stored_path,extracted_text,extraction_status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                document_id,
                values["project_id"],
                values["filename"],
                values["media_type"],
                values["size"],
                values["sha256"],
                values["stored_path"],
                values.get("extracted_text", ""),
                values.get("extraction_status", "ready"),
                utcnow(),
            ),
        )
        return self.db.one("SELECT id,project_id,filename,media_type,size,sha256,extraction_status,created_at FROM documents WHERE id=?", (document_id,))

    def list_documents(self, project_id: str, include_text: bool = False) -> list[dict[str, Any]]:
        fields = "*" if include_text else "id,project_id,filename,media_type,size,sha256,extraction_status,created_at"
        return self.db.all(f"SELECT {fields} FROM documents WHERE project_id=? ORDER BY created_at", (project_id,))

    def delete_document(self, project_id: str, document_id: str) -> dict[str, Any]:
        item = self.db.one("SELECT * FROM documents WHERE id=? AND project_id=?", (document_id, project_id))
        if not item:
            raise KeyError("Document not found")
        self.db.execute("DELETE FROM documents WHERE id=?", (document_id,))
        return item

    def create_run(self, project_id: str, workflow: str, model: str, execution_mode: str, human_review: bool) -> dict[str, Any]:
        self.require_project(project_id)
        run_id, now = new_id("run"), utcnow()
        self.db.execute(
            "INSERT INTO runs(id,project_id,workflow,status,current_step,progress,model,execution_mode,human_review,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, project_id, workflow, "queued", None, 0, model, execution_mode, int(human_review), now, now),
        )
        self.db.event(run_id, "run.created", "Run queued", {"workflow": workflow})
        return self.require_run(run_id)

    def require_run(self, run_id: str) -> dict[str, Any]:
        run = self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if not run:
            raise KeyError("Run not found")
        run["human_review"] = bool(run["human_review"])
        return run

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC", (project_id,))
        for row in rows:
            row["human_review"] = bool(row["human_review"])
        return rows

    def update_run(self, run_id: str, **changes: Any) -> dict[str, Any]:
        if not changes:
            return self.require_run(run_id)
        changes["updated_at"] = utcnow()
        assignments = ",".join(f"{key}=?" for key in changes)
        self.db.execute(f"UPDATE runs SET {assignments} WHERE id=?", (*changes.values(), run_id))
        return self.require_run(run_id)

    def save_artifact(self, run_id: str, step: str, agent_id: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        artifact_id = new_id("art")
        self.db.execute(
            "INSERT INTO artifacts(id,run_id,step,agent_id,content,metadata_json,created_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id,step) DO UPDATE SET agent_id=excluded.agent_id,content=excluded.content,metadata_json=excluded.metadata_json,created_at=excluded.created_at",
            (artifact_id, run_id, step, agent_id, content, json.dumps(metadata or {}, ensure_ascii=False), utcnow()),
        )
        return self.db.one("SELECT * FROM artifacts WHERE run_id=? AND step=?", (run_id, step))

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM artifacts WHERE run_id=? ORDER BY rowid", (run_id,))
        for row in rows:
            row["metadata"] = json.loads(row.pop("metadata_json"))
        return rows

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, after))
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows
