# Scientific Paper Writing Agent System

A repository-style agent workflow for writing, revising, and quality-checking English scientific papers.

This system is designed for general scientific manuscripts, including empirical papers, benchmark papers, method papers, clinical/biomedical studies, computational research, and thesis chapters. It does not replace the researcher; it structures the writing process so each claim is tied to evidence, citations are handled cautiously, figures and tables are explained clearly, and the final manuscript is reviewer-ready.

## What This System Provides

- A multi-agent writing workflow
- Agent role definitions and reusable prompts
- A complete manuscript drafting pipeline
- Section-level writing templates
- Claim-evidence mapping templates
- Figure and table writing guidance
- Reviewer-style quality checks
- Citation-safety rules
- Local frontend console and backend API
- Product, API, and development documentation

## Directory Structure

```text
PaperAI/
  README.md
  SYSTEM_OVERVIEW.md
  CONTRIBUTING.md
  LICENSE
  .env.example
  .gitignore
  agents/
    ORCHESTRATOR_AGENT.md
    LITERATURE_AGENT.md
    OUTLINE_AGENT.md
    METHODS_AGENT.md
    RESULTS_AGENT.md
    FIGURE_TABLE_AGENT.md
    DISCUSSION_AGENT.md
    REVIEWER_AGENT.md
    CITATION_AGENT.md
  workflows/
    PAPER_WORKFLOW.md
  templates/
    MANUSCRIPT_TEMPLATE.md
    CLAIM_EVIDENCE_MAP.md
    FIGURE_TABLE_TEMPLATE.md
  checklists/
    SUBMISSION_QUALITY_CHECKLIST.md
  config/
    agent_system.yaml
  figure_kits/
    ara-net/                 # MedIA / ARA-Net publication figure scripts
  skills/
    aranet-media-figures/    # pointer to Cursor skill
  .cursor/skills/
    aranet-media-figures/    # Cursor skill: regenerate/edit ARA-Net figures
  backend/
    server.py
  frontend/
    index.html
    styles.css
    app.js
    README.md
  docs/
    DEVELOPMENT.md
    PRODUCT_REQUIREMENTS.md
    API_REFERENCE.md
  .github/workflows/
    basic-checks.yml
```

## ARA-Net MedIA figure kit

Reusable matplotlib figure scripts and a Cursor skill for ARA-Net / MedIA-style panels (pipeline, atlas directionality, laser pathology outlines, error-group structure):

```text
figure_kits/ara-net/
.cursor/skills/aranet-media-figures/
```

**Full workspace figure archive** (all chapters, ~95 scripts): see `figure_kits/MANIFEST.md`
(`neurogate/`, `pathwaypro/`, `a2c-node/`, `cued-ad/`, `shared/`).

See `figure_kits/ara-net/README.md` and invoke the skill **aranet-media-figures** when regenerating those figures.

## Frontend Console

This repository includes a local frontend and backend for operating the agent system.

Start it from the repository root:

```bash
python backend/server.py
```

Open:

```text
http://127.0.0.1:8080/
```

Use the frontend to enter a paper title/topic, select what content to generate, and call the backend. The backend reads the selected agent instructions and reference templates, then either returns a Cursor-ready prompt or calls Qwen to generate manuscript text.

By default, generated content is downloaded by the browser as a `.md` file. It is not saved on the server unless `saveToServer` is explicitly sent as `true` to the API.

To call Qwen directly from the webpage, configure your local environment before starting the backend. See `.env.example` for the available variables.

Click `Run Qwen` in the frontend to generate manuscript content. The output will appear in the webpage and download through your browser.

Backend API:

```text
GET  /api/health
GET  /api/agents
POST /api/run-agent
```

## Documentation

- `SYSTEM_OVERVIEW.md`: architecture and agent responsibilities
- `docs/DEVELOPMENT.md`: local development guide
- `docs/PRODUCT_REQUIREMENTS.md`: product goals, users, features, and success criteria
- `docs/API_REFERENCE.md`: local backend API reference
- `CONTRIBUTING.md`: contribution workflow and standards

## Recommended Use

Use the orchestrator first:

```text
Read agents/ORCHESTRATOR_AGENT.md and workflows/PAPER_WORKFLOW.md.
Help me draft a paper from the materials in [project path].
Target venue/style: [journal/conference/preprint/thesis].
Output language: English.
```

Then call specialized agents as needed:

```text
Use agents/RESULTS_AGENT.md to write the Results section from these tables and figures.
```

```text
Use agents/FIGURE_TABLE_AGENT.md to write standalone captions for all figures.
```

```text
Use agents/REVIEWER_AGENT.md to review the manuscript like a critical peer reviewer.
```

## Core Principle

Every scientific claim must be traceable to one of the following:

- A result table
- A figure
- A statistical test
- A method description
- A verified citation
- A stated limitation

If the evidence is missing, mark the claim as `[EVIDENCE NEEDED]` or `[CITATION NEEDED]`.

## Output Standard

The system should produce writing that is:

- Accurate
- Cautious
- Evidence-linked
- Citation-aware
- Section-appropriate
- Reviewer-ready
- Free of unsupported novelty claims

## License

MIT
