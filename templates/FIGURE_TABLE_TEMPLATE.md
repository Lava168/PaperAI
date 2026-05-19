# Figure And Table Template

## Figure Caption Template

```markdown
Figure X. [Main takeaway]. (A) [Panel A description]. (B) [Panel B description]. (C) [Panel C description]. [Dataset/materials/model/condition sentence]. [Metric, uncertainty, or statistical notation sentence]. [Cautious interpretation sentence.]
```

## Caption Checklist

- [ ] Main takeaway appears in the first sentence.
- [ ] Every panel is described.
- [ ] Axes, colors, and symbols are defined.
- [ ] Dataset, model, group, or condition is identified when relevant.
- [ ] Metric and uncertainty are defined.
- [ ] Abbreviations are defined.
- [ ] Interpretation is cautious and evidence-backed.

## Table Title Template

```markdown
Table X. [Specific comparison or summary shown in the table].
```

## Table Note Template

```markdown
Values are reported as [mean +/- SD / median (IQR) / estimate (95% CI)] unless otherwise indicated. [Metric] denotes [definition]. Bold values indicate [valid comparison rule]. Abbreviations: [A], [B], [C].
```

## Table Checklist

- [ ] Title is specific.
- [ ] Rows and columns are understandable without the main text.
- [ ] Units are included.
- [ ] Metrics are defined.
- [ ] Uncertainty is defined.
- [ ] Significance markers are explained.
- [ ] Abbreviations are defined.
- [ ] Best values are highlighted only for valid comparisons.

## Common Figure Types

### Workflow Figure

Use for methods or pipeline overview.

Required:

- Inputs
- Main processing steps
- Outputs
- Evaluation stage

### Benchmark Figure

Use for performance comparisons.

Required:

- Compared methods
- Dataset or split
- Metric
- Error bars
- Direction of improvement

### Ablation Figure

Use for component analysis.

Required:

- Component removed or changed
- Fixed evaluation protocol
- Metric
- Interpretation of change

### Robustness Figure

Use for sensitivity, domain shift, perturbation, or subgroup analysis.

Required:

- Robustness condition
- Range of perturbation or domains
- Metric
- Stability interpretation

### Qualitative Figure

Use for examples, case studies, visualizations, or interpretation.

Required:

- Selection criterion
- What each row/column shows
- Warning that examples are illustrative, not population-level evidence
