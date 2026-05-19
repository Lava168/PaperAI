# System Overview

This project defines a modular agent system for English scientific paper writing. The system is organized around one orchestrator agent and eight specialist agents.

## Agent Architecture

```text
Research materials
  |
  v
ORCHESTRATOR_AGENT
  |
  +--> LITERATURE_AGENT
  +--> OUTLINE_AGENT
  +--> METHODS_AGENT
  +--> RESULTS_AGENT
  +--> FIGURE_TABLE_AGENT
  +--> DISCUSSION_AGENT
  +--> CITATION_AGENT
  +--> REVIEWER_AGENT
  |
  v
Manuscript draft + revision plan + quality report
```

## Agent Responsibilities

### ORCHESTRATOR_AGENT

Owns the full paper story and coordinates all other agents. It builds the manuscript plan, assigns tasks, merges outputs, detects inconsistencies, and maintains the claim-evidence map.

### LITERATURE_AGENT

Builds the related-work structure, identifies research gaps, checks whether novelty claims are justified, and marks missing citations.

### OUTLINE_AGENT

Creates the paper title, abstract plan, section structure, contribution list, and paragraph-level outline.

### METHODS_AGENT

Writes reproducible methods from code, experiment notes, configs, datasets, and protocols.

### RESULTS_AGENT

Turns tables, metrics, statistical tests, and figures into accurate results prose without overclaiming.

### FIGURE_TABLE_AGENT

Writes standalone figure captions, table titles, table notes, and visual result summaries.

### DISCUSSION_AGENT

Interprets results, compares them with prior work, writes limitations, and frames future work.

### CITATION_AGENT

Checks citation integrity. It should never invent references. Unverified citations must be marked as `[CITATION NEEDED]`.

### REVIEWER_AGENT

Reviews the manuscript like a critical peer reviewer. It identifies unsupported claims, missing experiments, unclear methods, weak figures, and likely reviewer objections.

## Data Objects

The system uses these shared objects:

- `paper_brief`: title idea, field, target venue, study type, intended contribution.
- `source_inventory`: code files, data files, tables, figures, notes, existing drafts.
- `claim_evidence_map`: each claim linked to evidence.
- `manuscript_outline`: section and paragraph plan.
- `draft_sections`: generated manuscript sections.
- `citation_ledger`: verified, missing, and uncertain citations.
- `review_report`: reviewer-style critique and revision tasks.

## Operating Rules

- Draft from evidence, not from enthusiasm.
- Separate observation, interpretation, and speculation.
- Prefer cautious scientific wording.
- Do not fabricate methods, numbers, citations, datasets, or statistical tests.
- Mark missing information explicitly.
- Keep section outputs compatible with a full manuscript.
- Preserve consistent terminology across title, abstract, figures, tables, and conclusion.

## Standard Workflow

1. Inventory project materials.
2. Define the central contribution.
3. Build the claim-evidence map.
4. Create the manuscript outline.
5. Draft Methods and Results from concrete evidence.
6. Draft Introduction, Related Work, Discussion, and Conclusion.
7. Write captions and table notes.
8. Run citation and reviewer checks.
9. Revise until claims, evidence, and wording align.
