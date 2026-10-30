# Upgraded HoloBench retrieval analysis

Questions: 90

| Method | Mean k | Mean tokens | Recall | Precision | Relevant F1 |
|---|---:|---:|---:|---:|---:|
| adaptive_k | 178.9 | 5220.1 | 0.667 | 0.686 | 0.586 |
| fixed_tokens_1000 | 32.0 | 981.8 | 0.166 | 0.798 | 0.274 |
| fixed_tokens_5000 | 164.2 | 4982.0 | 0.736 | 0.670 | 0.696 |
| fixed_tokens_10000 | 351.2 | 9984.4 | 0.880 | 0.375 | 0.515 |
| full_context | 635.7 | 16982.0 | 1.000 | 0.241 | 0.380 |

## Paired Adaptive-k minus token-budget baselines

| Baseline | Δ Recall (95% CI) | Δ Relevant F1 (95% CI) | Δ Tokens (95% CI) |
|---|---:|---:|---:|
| fixed_tokens_1000 | +0.500 [+0.433, +0.567] | +0.312 [+0.270, +0.358] | +4238.3 [+3480.5, +5001.6] |
| fixed_tokens_5000 | -0.069 [-0.127, -0.012] | -0.110 [-0.148, -0.074] | +238.0 [-518.7, +1003.2] |
| fixed_tokens_10000 | -0.214 [-0.274, -0.158] | +0.071 [+0.017, +0.125] | -4764.4 [-5520.4, -3999.3] |

## Per-split check: Adaptive-k vs fixed 5k-token budget

| Split | Adaptive recall | 5k recall | Adaptive F1 | 5k F1 | Adaptive tokens | 5k tokens |
|---|---:|---:|---:|---:|---:|---:|
| college_2 | 0.614 | 0.681 | 0.496 | 0.624 | 5807 | 4987 |
| flight_4 | 0.719 | 0.807 | 0.655 | 0.729 | 5318 | 4989 |
| soccer_1 | 0.587 | 0.672 | 0.587 | 0.669 | 4411 | 4969 |
| store_1 | 0.555 | 0.761 | 0.607 | 0.783 | 3280 | 4980 |
| wine_1 | 0.860 | 0.761 | 0.585 | 0.675 | 7285 | 4985 |

**Important:** these fixed token budgets use the exact BGE tokenizer counts stored by the local pilot. They are intended as fair matched-cost baselines inside this study, not as a literal reproduction of the paper's approximate fixed-token implementation.
