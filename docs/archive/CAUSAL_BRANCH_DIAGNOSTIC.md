# Causal Branch Diagnostic

Project path: `/home/heyang/projects/CausalGenGuard`
Environment: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`

## Run Bounds

- Max normal sequences: 100
- Epochs: 3
- Toy normal count: 120
- Real FR split: train=60, val=20, test=20

## Toy Causal Sanity Check

- Passed: `True`
- causal_only AUROC: 1.0000
- causal_only F1: 0.9931
- Top edges: `[{'src': 'B', 'dst': 'C', 'weight': 0.4954049587249756}, {'src': 'B', 'dst': 'A', 'weight': 0.4242289960384369}, {'src': 'A', 'dst': 'C', 'weight': 0.31274282932281494}, {'src': 'A', 'dst': 'B', 'weight': 0.025095243006944656}, {'src': 'C', 'dst': 'A', 'weight': 0.014780508354306221}]`

## Real FR Diagnostic

- causal_only AUROC: 0.4125
- quantile threshold F1: 0.0000
- oracle best F1: 0.6667
- edge_density: 0.8534
- nonzero_edges: 693
- top_edge_stability: `{'stable': True, 'jaccard_top_k': 0.9047619047619048, 'top_k': 20}`
- score_distribution normal_mean: 0.3721514086279759
- score_distribution anomaly_mean: -0.02952577129279463
- injection_report: `{'per_type_target': 20, 'success': {'edge_break': 20, 'edge_injection': 20, 'lag_delay': 20}, 'skipped': {'edge_break': 23, 'edge_injection': 2, 'lag_delay': 8}, 'injected_count': 60}`

## Outputs

- `outputs/diagnostics/causal_branch/toy_causal_sanity.json`
- `outputs/diagnostics/causal_branch/real_fr_causal_diagnostic.json`
- `outputs/diagnostics/causal_branch/top_50_edges.csv`
- `outputs/diagnostics/causal_branch/edge_density.json`
- `outputs/diagnostics/causal_branch/score_distribution.json`
- `outputs/diagnostics/causal_branch/threshold_sweep.csv`

## Diagnosis

- Toy causal passed: `True`
- Real FR top edges stable: `True`
- Causal score separability: `weak`
- Main problem judgement: `real FR causal scores have weak separability; likely data/anomaly construction and limited semantic coverage`
- Recommend keeping causal branch as main innovation: `False`
