# Answer-level pilot summary

Source answers: `results/answers-10.jsonl`  
Source retrieval: `results/pilot-10.jsonl`

| Method | Questions | Mean answer F1 | Mean context tokens |
|---|---:|---:|---:|
| gold_evidence | 10 | 0.338 | n/a |
| adaptive_k | 10 | 0.165 | 1281.2 |
| fixed_k_5 | 10 | 0.212 | 640.1 |
| fixed_k_10 | 10 | 0.201 | 1274.6 |

**adaptive_minus_fixed_k_5** (n=10): answer F1 mean diff -0.047 (95% CI [-0.300, +0.147]); context tokens mean diff +641.1 (95% CI [+191.7, +1282.1])  
**adaptive_minus_fixed_k_10** (n=10): answer F1 mean diff -0.036 (95% CI [-0.116, +0.037]); context tokens mean diff +6.6 (95% CI [-444.5, +647.6])  
