# FINAL_EXPERIMENT_RESULTS

This report summarizes the FR/SP seasonal context-shift results over three seeds (42, 43, 44). Values are reported as mean ± sample standard deviation over seeds.

## Full detection metrics

| Dataset | Method | Precision | Recall | F1 | Target-normal FPR | FNR | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FR | Source-only | 0.3838 ± 0.0632 | 0.2053 ± 0.0681 | 0.2664 ± 0.0722 | 0.6362 ± 0.0424 | 0.7947 ± 0.0681 | 0.2178 ± 0.0077 | 0.5026 ± 0.0028 |
| FR | Raw synthetic | 0.3535 ± 0.1350 | 0.0547 ± 0.0147 | 0.0945 ± 0.0271 | 0.2116 ± 0.0814 | 0.9453 ± 0.0147 | 0.2727 ± 0.0102 | 0.5330 ± 0.0100 |
| FR | TOF synthetic | 0.4224 ± 0.0423 | 0.0567 ± 0.0050 | 0.0999 ± 0.0086 | 0.1548 ± 0.0241 | 0.9433 ± 0.0050 | 0.2908 ± 0.0040 | 0.5431 ± 0.0022 |
| FR | Causal-TOF synthetic | 0.3701 ± 0.0578 | 0.0600 ± 0.0243 | 0.1005 ± 0.0313 | 0.2235 ± 0.1536 | 0.9400 ± 0.0243 | 0.2692 ± 0.0133 | 0.5310 ± 0.0113 |
| FR | Oracle target | 0.6260 ± 0.0673 | 0.0673 ± 0.0202 | 0.1213 ± 0.0345 | 0.0767 ± 0.0061 | 0.9327 ± 0.0202 | 0.6171 ± 0.0073 | 0.7071 ± 0.0073 |
| SP | Source-only | 0.0220 ± 0.0012 | 0.2417 ± 0.0144 | 0.0403 ± 0.0022 | 0.8607 ± 0.0042 | 0.7583 ± 0.0144 | 0.1150 ± 0.0710 | 0.0500 ± 0.0148 |
| SP | Raw synthetic | 0.0814 ± 0.0288 | 0.2583 ± 0.0144 | 0.1226 ± 0.0341 | 0.2493 ± 0.0699 | 0.7417 ± 0.0144 | 0.4168 ± 0.0674 | 0.0717 ± 0.0136 |
| SP | TOF synthetic | 0.0719 ± 0.0482 | 0.2667 ± 0.0144 | 0.1081 ± 0.0579 | 0.3687 ± 0.2266 | 0.7333 ± 0.0144 | 0.3758 ± 0.0545 | 0.0713 ± 0.0168 |
| SP | Causal-TOF synthetic | 0.0474 ± 0.0140 | 0.2500 ± 0.0000 | 0.0791 ± 0.0203 | 0.4333 ± 0.1602 | 0.7500 ± 0.0000 | 0.3140 ± 0.0169 | 0.0596 ± 0.0073 |
| SP | Oracle target | 0.1222 ± 0.1072 | 0.0167 ± 0.0144 | 0.0293 ± 0.0254 | 0.0107 ± 0.0031 | 0.9833 ± 0.0144 | 0.8022 ± 0.0134 | 0.2162 ± 0.0262 |

## Synthetic filtering statistics

| Dataset | Method | Synthetic Count | Kept Count | Rejected Count | Adaptation Gain |
| --- | --- | --- | --- | --- | --- |
| FR | Source-only | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |  |
| FR | Raw synthetic | 500 | 500.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.4246 ± 0.1163 |
| FR | TOF synthetic | 500 | 489.6667 ± 0.5774 | 10.3333 ± 0.5774 | 0.4815 ± 0.0542 |
| FR | Causal-TOF synthetic | 500 | 441.0000 ± 0.0000 | 59.0000 ± 0.0000 | 0.4127 ± 0.1148 |
| FR | Oracle target | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5595 ± 0.0481 |
| SP | Source-only | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |  |
| SP | Raw synthetic | 500 | 500.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6113 ± 0.0662 |
| SP | TOF synthetic | 500 | 470.0000 ± 0.0000 | 30.0000 ± 0.0000 | 0.4920 ± 0.2227 |
| SP | Causal-TOF synthetic | 500 | 423.0000 ± 0.0000 | 77.0000 ± 0.0000 | 0.4273 ± 0.1642 |
| SP | Oracle target | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8500 ± 0.0020 |

## Main observations

- On both FR and SP, source-only has very high target-context normal FPR, showing strong false alarms under seasonal context shift.
- Raw synthetic adaptation substantially reduces target-normal FPR on both datasets.
- TOF synthetic gives the lowest FR FPR in this run, while raw synthetic gives the lowest SP FPR among non-oracle methods.
- Causal-TOF is not consistently the best filtering strategy; it should be described as a diagnostic/filtering ablation rather than the main performance winner.
- Recall is low and FNR is high for many methods, so the safest paper claim is reduced false positive rate under context shift, not overall SOTA anomaly detection performance.