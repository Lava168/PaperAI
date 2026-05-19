# Product Requirements

## Product name

PaperAI: Scientific Paper Writing Agent System

## Goal

Help users draft, revise, and quality-check English scientific manuscripts with a structured multi-agent workflow.

## Target users

- Graduate students writing thesis chapters or journal papers
- Researchers preparing manuscripts
- Clinical or biomedical research teams
- Computational research teams
- Product or data teams turning experiments into formal reports

## Core user problems

1. Scientific papers require many linked sections and consistent terminology.
2. Results, figures, methods, and claims often become disconnected during drafting.
3. Citation and evidence gaps are hard to track manually.
4. Drafts need reviewer-style critique before submission.
5. Users need reusable prompts and workflows rather than one-off writing help.

## Core features

### Agent selection

Users can select a specialist agent such as Orchestrator, Literature, Methods, Results, Discussion, Citation, or Reviewer.

### Prompt package generation

The system builds a structured prompt using the selected agent instruction, workflow files, templates, and user inputs.

### Local frontend console

Users can operate the agent system through a browser-based local interface.

### Output download

Generated or assembled content can be downloaded as a Markdown file.

### Quality guardrails

The system should identify missing evidence, missing citations, unsupported novelty claims, vague methods, and inconsistent terminology.

## Non-goals for the initial version

- It is not a reference manager.
- It is not a full journal submission portal.
- It does not guarantee acceptance by any venue.
- It should not invent citations, datasets, experiments, or numeric results.

## Success criteria

- A new user can understand the project from the README.
- A user can run the local frontend from the repository root.
- Each agent has a clear role and expected output.
- Generated prompts are structured enough to use in Cursor or another assistant workflow.
- Manuscript quality checks explicitly flag missing evidence and citation risks.

## Future features

- Project-level source inventory upload
- Manuscript section versioning
- Claim-evidence map editor
- Citation ledger editor
- Reviewer objection tracking
- Export bundles for journal submission preparation
- Automated tests for backend routes and prompt assembly
