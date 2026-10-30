# QASPER retrieval pilot results

Ten training questions, one per paper, seed 42. BGE-large-en-v1.5 on MPS; fixed-size source-token windows (128), largest gap, 10% tail exclusion, five-chunk buffer. No answer generation or formal evidence-quality evaluation has been run.

| Paper | Available chunks | k before buffer | Adaptive k | Fixed k | Retrieval seconds |
|---|---:|---:|---:|---:|---:|
| 1904.02357 | 21 | 1 | 6 | 5 | 1.08 |
| 1702.06589 | 25 | 20 | 25 | 5 | 1.37 |
| 1808.07625 | 50 | 2 | 7 | 5 | 2.29 |
| 1906.06442 | 39 | 1 | 6 | 5 | 1.93 |
| 1910.04006 | 20 | 2 | 7 | 5 | 1.20 |
| 2003.05377 | 23 | 18 | 23 | 5 | 1.20 |
| 1811.07684 | 26 | 3 | 8 | 5 | 1.09 |
| 2001.05284 | 26 | 1 | 6 | 5 | 1.50 |
| 1805.05581 | 41 | 1 | 6 | 5 | 1.92 |
| 1808.09409 | 45 | 2 | 7 | 5 | 2.03 |

Total measured retrieval time: 15.60 seconds, excluding model download/load and dataset preparation.

Adaptive-k selected every chunk for 2 of 10 questions. Mean selected k: 10.1. This small pilot establishes that the pipeline runs; it does not establish an accuracy or efficiency advantage over fixed retrieval.

Token totals are BGE-tokenizer counts summed over chunks, excluding prompt overhead. They are not answer-model token counts.

The one-question preliminary run retrieved the annotated baseline sentence for its question. Other questions have not been manually evaluated.

Checks passed: 10 unique questions, finite descending scores, unique selected chunk IDs and selections matching the ranking. Full environment and configuration are saved alongside the JSONL.
