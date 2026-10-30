# Answer-level pilot summary

Source answers: `results/answers-2x2-semantic-100-p10.jsonl`  
Source retrieval: `results/2x2-semantic-100-p10.jsonl`

| Method | Questions | Skipped (context overflow) | Mean answer F1 | Mean context tokens |
|---|---:|---:|---:|---:|
| gold_evidence | 100 | 0 | 0.445 | n/a |
| adaptive_k | 100 | 0 | 0.233 | 438.5 |
| fixed_k_5 | 100 | 0 | 0.227 | 241.6 |
| fixed_k_10 | 100 | 0 | 0.256 | 507.9 |

**adaptive_minus_fixed_k_5** (n=100): answer F1 mean diff +0.005 (95% CI [-0.033, +0.044]); context tokens mean diff +196.9 (95% CI [+130.8, +279.4])  
**adaptive_minus_fixed_k_10** (n=100): answer F1 mean diff -0.023 (95% CI [-0.072, +0.024]); context tokens mean diff -69.5 (95% CI [-137.9, +14.8])  
