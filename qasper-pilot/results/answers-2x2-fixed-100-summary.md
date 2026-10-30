# Answer-level pilot summary

Source answers: `results/answers-2x2-fixed-100.jsonl`  
Source retrieval: `results/2x2-fixed-100.jsonl`

| Method | Questions | Skipped (context overflow) | Mean answer F1 | Mean context tokens |
|---|---:|---:|---:|---:|
| gold_evidence | 100 | 0 | 0.445 | n/a |
| adaptive_k | 87 | 13 | 0.280 | 1162.1 |
| fixed_k_5 | 100 | 0 | 0.263 | 633.0 |
| fixed_k_10 | 100 | 0 | 0.310 | 1268.2 |

**adaptive_minus_fixed_k_5** (n=87): answer F1 mean diff +0.033 (95% CI [-0.024, +0.087]); context tokens mean diff +530.1 (95% CI [+391.1, +672.4])  
**adaptive_minus_fixed_k_10** (n=87): answer F1 mean diff -0.018 (95% CI [-0.058, +0.025]); context tokens mean diff -104.4 (95% CI [-243.3, +36.8])  
