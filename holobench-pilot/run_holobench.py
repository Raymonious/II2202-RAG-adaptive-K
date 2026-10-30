"""Local HoloBench retrieval pilot: adaptive-k vs fixed-k, retrieval-level only.

Sanity check requested after the QASPER pilot: does Taguchi et al.'s
largest-gap adaptive-k selector show its expected behavior (fewer tokens,
comparable/better gold-relevance recall than fixed-k) in a setting closer to
one of the paper's own benchmarks (HoloBench), before drawing conclusions
from its QASPER-specific result? No answer generation here, retrieval only
-- see qasper-pilot/README.md for why (HoloBench's gold answer is a SQL
result, not free text).

`choose_k` below extends qasper-pilot/run_qasper.py's already-verified
largest-gap selector with head-exclusion (`ignore_extreme`), to match
HoloBench's repo-default `--ignore_extreme 0.05 --ignore_extreme_tail 0.1
--retrieve_more 5` (see adaptive-k-retrieval/adaptive-k-retrieval/
general_scripts/run_holobench.sh). Verified against the upstream
`find_threshold_largest_gap` the same way qasper-pilot/check_selector.py
verified the tail-only version -- see check_selector.py in this directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

REPO_SRC = Path(__file__).resolve().parents[1] / 'adaptive-k-retrieval' / 'adaptive-k-retrieval'
sys.path.insert(0, str(REPO_SRC))

SPLITS = ('wine_1', 'college_2', 'flight_4', 'store_1', 'soccer_1')


def choose_k(sorted_scores, buffer=5, tail_fraction=0.1, extreme_fraction=0.0):
    """Largest-gap selection with optional head exclusion (ignore_extreme).

    Matches retriever.py's find_threshold_largest_gap: exclude the first
    floor((n-1)*extreme_fraction) gaps and the last floor((n-1)*tail_fraction)
    gaps, select the first largest remaining drop, add `buffer` more, clamp.
    """
    n = len(sorted_scores)
    if not n:
        return 0, 0
    if n == 1:
        return 1, 1
    drops = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
    head_excluded = int((n - 1) * extreme_fraction)
    tail_excluded = int((n - 1) * tail_fraction)
    searchable = range(head_excluded, len(drops) - tail_excluded)
    gap_index = max(searchable, key=drops.__getitem__)
    before_buffer = gap_index + 1
    return before_buffer, min(n, before_buffer + buffer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument('--max-context-size', type=int, default=20000)
    parser.add_argument('--info-amount', type=int, default=5000)
    parser.add_argument('--merge-strategy', default='uniform',
                        choices=['uniform', 'begin', 'end', 'middle', 'bimodal'])
    parser.add_argument('--model', default='BAAI/bge-large-en-v1.5')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fixed-k', nargs='+', type=int, default=[5, 10])
    parser.add_argument('--buffer', type=int, default=5)
    parser.add_argument('--ignore-extreme', type=float, default=0.05)
    parser.add_argument('--ignore-extreme-tail', type=float, default=0.1)
    parser.add_argument('--out', type=Path, default=Path('results/pilot.jsonl'))
    args = parser.parse_args()

    os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / 'qasper-pilot' / 'data' / 'hf-cache'))
    from datasets import load_dataset
    from load_context import load_context
    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; rerun with --device cpu.')
    model = SentenceTransformer(args.model, device=args.device)
    tokenizer = model.tokenizer

    queries = load_dataset('megagonlabs/holobench', name='queries')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update({
            'versions': {p: importlib.metadata.version(p) for p in
                         ['sentence-transformers', 'transformers', 'torch', 'datasets', 'duckdb']},
        })
        args.out.with_suffix('.config.json').write_text(json.dumps(config, indent=2))

        number = 0
        total = sum(len(queries[s]) for s in args.splits)
        for split in args.splits:
            db = load_dataset('megagonlabs/holobench', name=split)
            for ins in queries[split]:
                number += 1
                question_start = time.perf_counter()
                context, gold_answer, relevant_indices = load_context(
                    ins['query'], db, max_context_size=args.max_context_size,
                    info_amount=args.info_amount, merge_strategy=args.merge_strategy)
                chunks = context.split('\n')
                if not chunks:
                    raise ValueError(f'No chunks for {split}/{ins["question"][:40]}')
                embeddings = model.encode(chunks, batch_size=args.batch_size,
                                          normalize_embeddings=True, convert_to_numpy=True)
                query_emb = model.encode([ins['question']], normalize_embeddings=True,
                                         convert_to_numpy=True)[0]
                scores = (embeddings @ query_emb).tolist()
                ranking = sorted(range(len(chunks)), key=lambda i: -scores[i])
                sorted_scores = [scores[i] for i in ranking]
                before_buffer, k = choose_k(sorted_scores, args.buffer,
                                            args.ignore_extreme_tail, args.ignore_extreme)
                chunk_tokens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in chunks]

                selections = {}
                adaptive_ids = ranking[:k]
                selections['adaptive_k'] = {
                    'k': len(adaptive_ids), 'chunk_ids': adaptive_ids,
                    'retrieved_embedding_tokens': sum(chunk_tokens[i] for i in adaptive_ids),
                }
                for fk in args.fixed_k:
                    ids = ranking[:fk]
                    selections[f'fixed_k_{fk}'] = {
                        'k': len(ids), 'chunk_ids': ids,
                        'retrieved_embedding_tokens': sum(chunk_tokens[i] for i in ids),
                    }

                record = {
                    'question_id': f'{split}:{number}', 'split': split,
                    'question': ins['question'], 'sql_query': ins['query'],
                    'num_chunks': len(chunks), 'chunk_tokens': chunk_tokens,
                    'ranked_chunk_ids': ranking, 'sorted_scores': sorted_scores,
                    'k_before_buffer': before_buffer,
                    'retrieves_everything': k == len(chunks),
                    'relevant_indices': list(relevant_indices),
                    'gold_rows': len(gold_answer),
                    'selections': selections,
                    'retrieval_seconds': time.perf_counter() - question_start,
                }
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                output.flush()
                print(f'{number}/{total} {split}: {len(chunks)} chunks, '
                      f'{len(relevant_indices)} relevant; gap selects {before_buffer}, '
                      f'buffered k={k}', flush=True)
    print(f'Saved {args.out} and {args.out.with_suffix(".config.json")}')


if __name__ == '__main__':
    main()
