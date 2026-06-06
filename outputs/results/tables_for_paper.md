# Tables For Paper

Generated from local files under `outputs/results`. Missing result categories are reported instead of filled with synthetic values.

## Main context shift results

| dataset | transition | method | precision | recall | f1 | fpr | fnr | auroc | auprc | delta_f1_vs_smartguard | adaptation_gain_f1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy | unknown | FusionDetector | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 0.0000 | 0.2500 | 0.5000 |  |  |  |
| toy | unknown | FusionDetector | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 0.0000 | 0.2500 | 0.5000 |  |  | score distribution plot failed: No module named 'matplotlib'; causal graph diff plot for toy-2 failed: No module named 'matplotlib'; causal graph diff plot for toy-3 failed: No module named 'matplotlib' |
| toy | winter->spring | Full CausalGenGuard + CausalTOF + causal-aware NWRL | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.4167 | -0.6667 |  | Causal-TOF enabled; causal-aware NWRL fine-tune enabled after causal centrality estimation; Causal-TOF filtered synthetic sequences from 4 to 1; causal-aware NWRL fine-tune applied after causal centrality estimation |

## Low-data adaptation results

No results found for this table.

## Causal anomaly results

No results found for this table.

## Ablation results

| dataset | transition | method | precision | recall | f1 | fpr | fnr | auroc | auprc | delta_f1_vs_smartguard | adaptation_gain_f1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy | winter->spring | Full CausalGenGuard + CausalTOF + causal-aware NWRL | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.4167 | -0.6667 |  | Causal-TOF enabled; causal-aware NWRL fine-tune enabled after causal centrality estimation; Causal-TOF filtered synthetic sequences from 4 to 1; causal-aware NWRL fine-tune applied after causal centrality estimation |
| toy | winter->spring | SmartGuard + Causal | 1.0000 | 0.5000 | 0.6667 | 0.0000 | 0.5000 | 0.7500 | 0.8333 | 0.0000 |  | source normal plus behavior causal graph deviation; SmartGen augmentation disabled by experiment config |
| toy | winter->spring | SmartGuard + SmartGen | 1.0000 | 0.5000 | 0.6667 | 0.0000 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |  | synthetic target normal augmentation without causal branch |
| toy | winter->spring | SmartGuard + SmartGen + Causal | 1.0000 | 0.5000 | 0.6667 | 0.0000 | 0.5000 | 0.7500 | 0.8333 | 0.0000 |  | SmartGen augmentation and causal branch, no Causal-TOF |
| toy | winter->spring | SmartGuard only | 1.0000 | 0.5000 | 0.6667 | 0.0000 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |  | reconstruction score only; SmartGen augmentation disabled by experiment config |

## Runtime results if available

No results found for this table.

## Missing Or Skipped

- Low-data adaptation results: no rows found.
- Causal anomaly results: no rows found.
- Runtime results: no runtime fields found.
- Oracle-target results: no rows found; adaptation_gain_f1 left blank.
- outputs/results/fr_smoke_eval.json: no metric dictionary found; skipped
- outputs/results/fr_smoke_eval_explain.json: no metric dictionary found; skipped
