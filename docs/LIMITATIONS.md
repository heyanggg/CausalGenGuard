# Limitations

1. The causal branch should currently be treated as a filtering and explanation
   module, not as the main detector.
2. A single-seed or FR-only run is not enough for final claims.
3. Target-context anomaly labels are required before reporting final F1, AUROC,
   and AUPRC under context shift.
4. SmartGuard-style named attack injection must be balanced by anomaly type;
   random injection can skip many attack types.
5. SmartGen online LLM calls should remain optional.  Offline synthetic data is
   preferred for reproducible experiments.
6. Generated outputs, checkpoints, and binary arrays should not be committed to
   Git.
