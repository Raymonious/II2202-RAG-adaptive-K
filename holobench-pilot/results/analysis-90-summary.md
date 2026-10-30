# HoloBench retrieval pilot: does adaptive-k show its expected tendency?

90 questions (18 × 5 splits: wine_1, college_2, flight_4, store_1,
soccer_1). BGE-large-en-v1.5 on MPS. `max_context_size=20000`,
`info_amount=5000`, `merge_strategy=uniform` (paper's smallest tested
`info_amount`; `max_context_size` trimmed from their 100000 for local
tractability). Selector: largest-gap, `ignore_extreme=0.05`,
`ignore_extreme_tail=0.1`, `retrieve_more=5` — HoloBench's own repo defaults
(`general_scripts/run_holobench.sh`), verified against the upstream
`find_threshold_largest_gap` in `check_selector.py` (812 comparisons passed).
Candidate pool per question: 424–887 chunks (median 573) — far larger than
QASPER's 20–50.

| Method | Mean k | Mean tokens | Relevant recall | Relevant precision | Relevant F1 |
|---|---:|---:|---:|---:|---:|
| full_context | 635.7 | 16982 | 1.000 | 0.241 | 0.380 |
| adaptive_k | 178.9 | 5220 | 0.667 | 0.686 | 0.586 |
| fixed_k_10 | 10 | 390 | 0.066 | 0.783 | 0.120 |
| fixed_k_5 | 5 | 194 | 0.033 | 0.778 | 0.062 |

**adaptive_k vs full_context:** 69% fewer tokens (5220 vs 16982) for a real
but partial recall trade-off (0.667 vs 1.000), and a ~2.8x precision gain
(0.686 vs 0.241) — exactly the efficiency/quality trade-off adaptive-k is
designed to make.

**adaptive_k vs fixed_k_5/10:** adaptive-k recall is +0.634 (95% CI
[0.565, 0.704]) over fixed_k_5 and +0.601 (CI [0.534, 0.669]) over fixed_k_10
— both clearly non-zero. Fixed_k=5/10 are essentially non-functional here
(recall 0.03–0.07, 0/90 questions reaching 90% recall) because HoloBench's
relevant sets are typically dozens to 100+ rows; a fixed k of 5 or 10 can
never capture that regardless of ranking quality. This makes the fixed-k=5/10
comparison a mismatch for this domain rather than a fair baseline — included
for continuity with the QASPER pilot's method names, not as the main
evidence here.

**Conclusion:** in this larger-candidate-pool regime, adaptive-k behaves as
expected — it finds a much smaller-than-full-context k (178.9 of ~573 median
chunks) that keeps most of the relevant signal and meaningfully improves
precision over full context, while naive tiny fixed-k baselines fail
outright. This supports treating the QASPER pilot's null result
(`qasper-pilot/results/answers-100-summary.md`) as a genuine
dataset/setup-dependence finding rather than a broken pipeline: the
mechanism itself works when the candidate pool is large enough to have real
structure to adapt to.

Not measured here: answer-level accuracy (HoloBench's gold answer is a SQL
result, not free text — scoring it needs structured-value comparison or an
LLM judge, deliberately out of scope for this retrieval-level sanity check).
