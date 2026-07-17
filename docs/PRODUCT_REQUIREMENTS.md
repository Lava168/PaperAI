# Product Requirements

## Product

PaperAI is a standalone scientific writing agent for researchers who need a traceable path from source material to an English manuscript.

## Primary users

- Graduate students and researchers preparing papers or thesis chapters
- Clinical, biomedical, computational, and data-science teams
- Research groups that need a repeatable internal pre-submission review process

## Core jobs

1. Preserve the relationship between manuscript claims and source evidence.
2. Coordinate specialist writing and review tasks without losing terminology or decisions.
3. Make missing evidence, citations, and method details visible instead of fabricating them.
4. Keep intermediate outputs, human feedback, and revision history recoverable.
5. Export portable manuscript drafts without locking users into the interface.

## Version 2 capabilities

- Persistent projects, uploaded sources, runs, artifacts, and events
- Extraction for common research document and tabular formats
- Five resumable workflows and nine specialist agent roles
- OpenAI-compatible model execution plus model-free prompt mode
- Human checkpoints after outline and peer review
- Crossref literature metadata discovery and DOI resolution
- Local web workspace and REST/OpenAPI interface
- Markdown, Word, and LaTeX export
- Automated API and workflow tests

## Explicit non-goals

- PaperAI does not guarantee journal acceptance.
- Crossref metadata does not prove that a source supports a claim.
- PaperAI is not a replacement for researcher judgment, statistical review, ethics review, or clinical governance.
- The local edition is not a public multi-tenant SaaS and includes no authentication or billing.

## Success criteria

- A first-time user can create a project, upload evidence, run an outline workflow, inspect artifacts, and export the result.
- Interrupted or approval-gated workflows resume without repeating completed stages.
- Every stage receives the paper brief, available source text, and prior artifacts.
- Missing support remains explicitly marked in downstream outputs.
- Tests validate the central project, source, workflow, checkpoint, and export path.

## Next production milestones

- Durable external job queue and multi-instance deployment
- Team accounts, roles, encryption, audit policy, and retention controls
- Object storage and PostgreSQL repository implementations
- Reference-library import/export and claim-level citation verification UI
- Token budgeting, retrieval indexing, and per-project model cost reporting
