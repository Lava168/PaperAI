from __future__ import annotations

import time
from dataclasses import replace

from fastapi.testclient import TestClient

from backend.paperai.api import create_app
from backend.paperai.config import load_settings


class FakeProvider:
    model_name = "fake-scientist"

    def complete(self, system: str, user: str) -> str:
        step = next((line.split(":", 1)[1].strip() for line in user.splitlines() if line.startswith("Step:")), "unknown")
        return f"# {step}\n\nEvidence-grounded test output."


def make_client(tmp_path) -> TestClient:
    settings = replace(
        load_settings(),
        data_dir=tmp_path,
        database_path=tmp_path / "test.db",
        model_api_key="test-key",
        model_name="fake-scientist",
    )
    return TestClient(create_app(settings, FakeProvider()))


def wait_for_run(client: TestClient, run_id: str) -> dict:
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"completed", "failed", "waiting_approval"}:
            return run
        time.sleep(0.02)
    raise AssertionError("Run did not finish")


def test_project_upload_and_prompt_workflow(tmp_path):
    with make_client(tmp_path) as client:
        project_response = client.post(
            "/api/projects",
            json={"title": "Test paper", "topic": "Reliable scientific agents", "paper_type": "method"},
        )
        assert project_response.status_code == 201
        project = project_response.json()

        upload = client.post(
            f"/api/projects/{project['id']}/documents",
            files={"file": ("results.txt", b"Accuracy: 0.91; baseline: 0.84", "text/plain")},
        )
        assert upload.status_code == 201
        assert upload.json()["extraction_status"] == "ready"

        response = client.post(
            f"/api/projects/{project['id']}/runs",
            json={"workflow": "outline", "execution_mode": "prompt", "human_review": False},
        )
        assert response.status_code == 202
        run = wait_for_run(client, response.json()["id"])
        assert run["status"] == "completed"
        assert [item["step"] for item in run["artifacts"]] == ["intake", "claim_evidence", "outline"]
        assert "Accuracy: 0.91" in run["artifacts"][0]["content"]

        exported = client.get(f"/api/runs/{run['id']}/export?format=md")
        assert exported.status_code == 200
        assert "Prompt package" in exported.text


def test_model_workflow_and_human_checkpoint(tmp_path):
    with make_client(tmp_path) as client:
        project = client.post("/api/projects", json={"title": "Review-gated paper"}).json()
        response = client.post(
            f"/api/projects/{project['id']}/runs",
            json={"workflow": "full_paper", "execution_mode": "model", "human_review": True},
        )
        run = wait_for_run(client, response.json()["id"])
        assert run["status"] == "waiting_approval"
        assert run["current_step"] == "outline"

        resumed = client.post(
            f"/api/runs/{run['id']}/resume",
            json={"approved": True, "feedback": "Keep the contribution conservative."},
        )
        assert resumed.status_code == 202
        run = wait_for_run(client, run["id"])
        assert run["status"] == "waiting_approval"
        assert run["current_step"] == "peer_review"
        assert any(item["step"] == "feedback_outline" for item in run["artifacts"])


def test_reject_unconfigured_model(tmp_path):
    settings = replace(load_settings(), data_dir=tmp_path, database_path=tmp_path / "test.db", model_api_key="")
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/projects", json={"title": "No model"}).json()
        response = client.post(f"/api/projects/{project['id']}/runs", json={"execution_mode": "model"})
        assert response.status_code == 400
        assert "Model is not configured" in response.json()["detail"]
