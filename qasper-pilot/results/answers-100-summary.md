# Answer-level pilot summary

Source answers: `results/answers-100-v2.jsonl`  
Source retrieval: `results/feasibility-100-128.jsonl`

| Method | Questions | Skipped (context overflow) | Mean answer F1 | Mean context tokens |
|---|---:|---:|---:|---:|
| gold_evidence | 100 | 0 | 0.445 | n/a |
| adaptive_k | 97 | 3 | 0.289 | 1534.9 |
| fixed_k_5 | 100 | 0 | 0.263 | 633.0 |
| fixed_k_10 | 100 | 0 | 0.310 | 1268.2 |

**adaptive_minus_fixed_k_5** (n=97): answer F1 mean diff +0.032 (95% CI [-0.021, +0.085]); context tokens mean diff +902.2 (95% CI [+667.0, +1164.8])  
**adaptive_minus_fixed_k_10** (n=97): answer F1 mean diff -0.010 (95% CI [-0.051, +0.028]); context tokens mean diff +267.1 (95% CI [+30.7, +529.0])  
