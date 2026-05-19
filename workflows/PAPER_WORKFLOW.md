# Paper Workflow

This workflow turns research materials into an English scientific manuscript.

## Phase 1: Intake

Collect:

- Target paper type: empirical, method, benchmark, review, clinical, theory, thesis chapter.
- Target venue or writing style.
- Existing draft, if any.
- Figures, tables, metrics, and statistical results.
- Dataset and method descriptions.
- Code/configs/logs that determine methods.
- Bibliography or key papers.

Output:

```markdown
## Paper Brief
- Working title:
- Field:
- Paper type:
- Target venue/style:
- Central contribution:
- Primary evidence:
- Main limitations:
```

## Phase 2: Source Inventory

Create an inventory of all usable materials.

Output:

```markdown
## Source Inventory
| Source | Type | What it supports | Status |
| --- | --- | --- | --- |
| [file/table/figure] | [data/code/result/draft] | [claim or section] | [ready/needs check] |
```

## Phase 3: Claim-Evidence Map

Before drafting, map claims to evidence.

Use `templates/CLAIM_EVIDENCE_MAP.md`.

Rules:

- No major claim should enter the abstract without evidence.
- Novelty claims require literature support.
- Clinical, causal, or mechanistic claims require extra caution.
- Missing support must be marked `[EVIDENCE NEEDED]`.

## Phase 4: Outline

Use `agents/OUTLINE_AGENT.md`.

Output:

- Title options
- Abstract outline
- Section structure
- Contribution bullets
- Paragraph-level plan
- Figure/table placement plan

## Phase 5: Methods Draft

Use `agents/METHODS_AGENT.md`.

Write Methods before Results when the project materials are complex. This reduces hallucination and ensures the evaluation protocol is clear.

Methods must include:

- Dataset/cohort/source
- Inclusion/exclusion rules
- Preprocessing
- Model or procedure
- Training or experimental setup
- Evaluation metrics
- Statistical analysis
- Reproducibility details

## Phase 6: Results Draft

Use `agents/RESULTS_AGENT.md`.

Results prose should:

- Start each paragraph with the question tested.
- Report the primary number and uncertainty.
- Point to the relevant figure or table.
- Avoid interpreting beyond the evidence.
- Mention negative or mixed results when relevant.

## Phase 7: Figures And Tables

Use `agents/FIGURE_TABLE_AGENT.md`.

Every figure caption must be standalone:

- Main takeaway first
- Panel descriptions
- Dataset/model/metric definitions
- Statistical notation
- Cautious interpretation

Every table must include:

- Specific title
- Clear metric definitions
- Notes for abbreviations and uncertainty
- Valid comparison boundaries

## Phase 8: Introduction And Related Work

Use `agents/LITERATURE_AGENT.md` and `agents/OUTLINE_AGENT.md`.

Introduction flow:

1. Scientific or clinical problem
2. Gap in current methods or knowledge
3. Proposed approach
4. Evidence-backed contributions

Related Work should be organized by concept, not by paper list.

## Phase 9: Discussion And Conclusion

Use `agents/DISCUSSION_AGENT.md`.

Discussion flow:

1. Principal findings
2. Interpretation
3. Comparison with prior work
4. Implications
5. Limitations
6. Future work

Conclusion should be short and should not introduce new evidence.

## Phase 10: Citation Check

Use `agents/CITATION_AGENT.md`.

Check:

- Are citations real?
- Do citations support the exact claim?
- Are key related works missing?
- Are novelty claims justified?
- Are unverifiable references marked `[CITATION NEEDED]`?

## Phase 11: Reviewer Simulation

Use `agents/REVIEWER_AGENT.md`.

Generate:

- Major concerns
- Minor concerns
- Missing experiments or analyses
- Overclaims
- Method reproducibility gaps
- Figure/table issues
- Revision checklist

## Phase 12: Final Revision

Use the reviewer report to revise:

- Abstract
- Introduction contributions
- Methods reproducibility
- Results precision
- Discussion caution
- Figure captions
- Limitations

Run `checklists/SUBMISSION_QUALITY_CHECKLIST.md` before final delivery.
