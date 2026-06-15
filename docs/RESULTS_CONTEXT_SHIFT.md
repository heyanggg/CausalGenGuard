# Context Shift Results Notes

This document should contain final result tables only after the official
multi-seed experiments are run.

Current development reports suggest that the promising direction is not
`causal_only` detection, but synthetic target-context adaptation.  Keep this
interpretation in the paper unless stronger causal detection results are later
obtained.

Recommended final tables:

```text
Table 1. Main context-shift results by dataset and transition.
Table 2. Low-data target adaptation results.
Table 3. Ablation of raw synthetic / TOF / Causal-TOF.
Table 4. Synthetic quality and filtering report.
```

Do not manually type numbers into the paper.  Generate them from
`evaluation/summarize_results.py` or a reproducible result CSV.
