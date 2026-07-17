from __future__ import annotations

from dataclasses import dataclass
from .config import ROOT


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    purpose: str
    instruction_file: str

    def instruction(self) -> str:
        return (ROOT / self.instruction_file).read_text(encoding="utf-8")


AGENTS = {
    "orchestrator": AgentDefinition("orchestrator", "Orchestrator", "Plan and integrate the full manuscript.", "agents/ORCHESTRATOR_AGENT.md"),
    "literature": AgentDefinition("literature", "Literature", "Frame prior work and audit novelty claims.", "agents/LITERATURE_AGENT.md"),
    "outline": AgentDefinition("outline", "Outline", "Design the manuscript story and section plan.", "agents/OUTLINE_AGENT.md"),
    "methods": AgentDefinition("methods", "Methods", "Draft reproducible methods from source materials.", "agents/METHODS_AGENT.md"),
    "results": AgentDefinition("results", "Results", "Report results without exceeding the evidence.", "agents/RESULTS_AGENT.md"),
    "figure_table": AgentDefinition("figure_table", "Figures & Tables", "Create standalone captions and table notes.", "agents/FIGURE_TABLE_AGENT.md"),
    "discussion": AgentDefinition("discussion", "Discussion", "Interpret findings, limitations, and implications.", "agents/DISCUSSION_AGENT.md"),
    "citation": AgentDefinition("citation", "Citation Auditor", "Identify unsupported or unverifiable references.", "agents/CITATION_AGENT.md"),
    "reviewer": AgentDefinition("reviewer", "Peer Reviewer", "Find scientific, reporting, and reproducibility risks.", "agents/REVIEWER_AGENT.md"),
}


WORKFLOWS: dict[str, list[tuple[str, str]]] = {
    "full_paper": [
        ("intake", "orchestrator"),
        ("claim_evidence", "orchestrator"),
        ("outline", "outline"),
        ("literature", "literature"),
        ("methods", "methods"),
        ("results", "results"),
        ("figures_tables", "figure_table"),
        ("discussion", "discussion"),
        ("citation_audit", "citation"),
        ("peer_review", "reviewer"),
        ("final_manuscript", "orchestrator"),
    ],
    "outline": [("intake", "orchestrator"), ("claim_evidence", "orchestrator"), ("outline", "outline")],
    "methods": [("intake", "orchestrator"), ("methods", "methods"), ("methods_review", "reviewer")],
    "results": [("intake", "orchestrator"), ("results", "results"), ("figures_tables", "figure_table"), ("results_review", "reviewer")],
    "review": [("citation_audit", "citation"), ("peer_review", "reviewer"), ("revision_plan", "orchestrator")],
}


STEP_TASKS = {
    "intake": "Create a paper brief, source inventory, central contribution, evidence gaps, and a prioritized work plan.",
    "claim_evidence": "Create a rigorous claim-evidence map. Label every unsupported item [EVIDENCE NEEDED] or [CITATION NEEDED].",
    "outline": "Create title options, an abstract plan, contribution bullets, section hierarchy, and paragraph-level outline.",
    "literature": "Draft the Introduction and Related Work structure using only supplied citations. Do not invent references.",
    "methods": "Draft a reproducible Methods section. Preserve exact details and label every missing detail [METHOD DETAIL NEEDED].",
    "methods_review": "Audit the Methods section for reproducibility, leakage, bias, missing parameters, and unclear statistical analysis.",
    "results": "Draft the Results section from supplied evidence. Include exact values when available and avoid causal overclaiming.",
    "figures_tables": "Draft standalone captions, table titles, notes, and a figure/table placement plan from available materials.",
    "results_review": "Review result reporting, comparisons, uncertainty, statistical claims, and figure/table references.",
    "discussion": "Draft Discussion, limitations, implications, future work, and a concise conclusion grounded in results.",
    "citation_audit": "Audit every citation and externally sourced claim. Produce a ledger of verified, uncertain, and missing citations.",
    "peer_review": "Act as a demanding peer reviewer. Return major concerns, minor concerns, required revisions, and a verdict rationale.",
    "revision_plan": "Turn the audits into a prioritized, actionable revision plan with acceptance criteria.",
    "final_manuscript": "Integrate the approved artifacts into one coherent manuscript. Resolve reviewer issues when evidence allows; retain explicit gap markers otherwise.",
}


CHECKPOINT_STEPS = {"outline", "peer_review"}
