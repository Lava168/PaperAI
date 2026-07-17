from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    topic: str = Field(default="", max_length=5000)
    venue: str = Field(default="", max_length=300)
    paper_type: str = Field(default="empirical", max_length=100)
    language: str = Field(default="English", max_length=50)
    instructions: str = Field(default="", max_length=10000)


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    topic: str | None = Field(default=None, max_length=5000)
    venue: str | None = Field(default=None, max_length=300)
    paper_type: str | None = Field(default=None, max_length=100)
    language: str | None = Field(default=None, max_length=50)
    instructions: str | None = Field(default=None, max_length=10000)


class RunCreate(BaseModel):
    workflow: Literal["full_paper", "outline", "methods", "results", "review"] = "full_paper"
    execution_mode: Literal["model", "prompt"] = "model"
    human_review: bool = False


class ResumeRun(BaseModel):
    approved: bool = True
    feedback: str = Field(default="", max_length=10000)


class LegacyAgentRequest(BaseModel):
    agentId: str = "orchestrator"
    title: str = "Untitled paper"
    topic: str = ""
    venue: str = ""
    paperType: str = "empirical"
    task: str = ""
    materials: str = ""
    desiredOutput: str = "Markdown"
    executeModel: bool = False
    saveToServer: bool = False
