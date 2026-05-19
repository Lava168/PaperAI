# Claim Evidence Map

Use this before drafting and update it during revision.

## Central Contribution

```markdown
[One sentence stating what the paper contributes and what evidence supports it.]
```

## Claim Table

```markdown
| ID | Claim | Evidence | Location in paper | Support strength | Missing items | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | [Main claim] | [Figure/Table/Result/Citation] | [Abstract/Intro/Results] | [Strong/Moderate/Weak] | [None/EVIDENCE NEEDED/CITATION NEEDED] | [Low/Medium/High] |
```

## Claim Types

### Method Claim

Example:

```markdown
The proposed method improves [task] performance compared with [baseline].
```

Required evidence:

- Baseline comparison
- Same dataset/split
- Metric
- Uncertainty or statistical test

### Mechanism Claim

Example:

```markdown
The module contributes to improved robustness under [condition].
```

Required evidence:

- Ablation
- Sensitivity analysis
- Robustness experiment
- Alternative explanation check

### Generalization Claim

Example:

```markdown
The method generalizes across [site/domain/dataset/condition].
```

Required evidence:

- External validation or cross-domain evaluation
- Cohort/domain description
- Performance and uncertainty

### Scientific Interpretation Claim

Example:

```markdown
The observed pattern is consistent with [known process/theory].
```

Required evidence:

- Result
- Citation
- Cautious language
- Alternative explanation

### Clinical Or Applied Claim

Example:

```markdown
The method may support [clinical/applied use].
```

Required evidence:

- Validation design
- Clinically meaningful metric
- Safety or limitation discussion
- Prospective/external validation if deployment is implied

## Risk Labels

- Low: directly supported by result and citation.
- Medium: supported but requires cautious wording.
- High: missing evidence, weak comparison, or likely reviewer objection.

## Revision Rule

If a claim has high risk, either:

1. Add evidence.
2. Move it to limitations/future work.
3. Weaken the wording.
4. Remove it.
