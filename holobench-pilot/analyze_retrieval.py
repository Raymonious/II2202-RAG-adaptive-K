"""Retrieval-level analysis for the HoloBench pilot: does adaptive-k show
its expected tendency (fewer tokens, comparable/better gold-relevance
recall than fixed-k) in a larger-candidate-pool regime than QASPER?

Gold here is `relevant_indices`: which merged-context chunks came from the
table(s) the SQL query actually references (see load_context.py). This is an
item-level recall/precision/F1 (each chunk is already an atomic unit, unlike
QASPER's character-offset evidence coverage).
"""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path


def coverage(retrieved_ids, gold_ids):
    overlap = len(retrieved_ids & gold_ids)
    recall = overlap / len(gold_ids) if gold_ids else 0.0
    precision = overlap / len(retrieved_ids) if retrieved_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return {'relevant_recall': recall, 'relevant_precision': precision, 'relevant_f1': f1}


def selections(row):
    ranking = row['ranked_chunk_ids']
    for name, entry in row['selections'].items():
        yield name, entry['chunk_ids']
    yield 'gap_without_buffer_diagnostic', ranking[:row['k_before_buffer']]
    yield 'full_context', ranking


def mean_ci(differences):
    rng = random.Random(2026)
    n = len(differences)
    samples = sorted(statistics.mean(rng.choices(differences, k=n)) for _ in range(2000))
    return {'mean_difference': statistics.mean(differences),
            'paired_bootstrap_95_percent_interval': [samples[49], samples[1949]]}


def analyze(rows):
    output = []
    for row in rows:
        gold = set(row['relevant_indices'])
        for name, ids in selections(row):
            chosen = set(ids)
            scores = coverage(chosen, gold)
            output.append({'question_id': row['question_id'], 'split': row['split'],
                           'method': name, 'k': len(ids),
                           'tokens': sum(row['chunk_tokens'][i] for i in ids),
                           'full_selection': len(ids) == row['num_chunks'], **scores})
    method_names = [name for name, _ in selections(rows[0])]
    grouped = {name: [r for r in output if r['method'] == name] for name in method_names}
    summary = {}
    for name, values in grouped.items():
        summary[name] = {key: statistics.mean(r[key] for r in values) for key in
                         ['k', 'tokens', 'relevant_recall', 'relevant_precision', 'relevant_f1']}
        summary[name]['full_selection_count'] = sum(r['full_selection'] for r in values)
        summary[name]['zero_relevant_overlap_count'] = sum(r['relevant_recall'] == 0 for r in values)
        summary[name]['at_least_90_percent_recall_count'] = sum(r['relevant_recall'] >= .9 for r in values)
    paired = {}
    for name in ('fixed_k_5', 'fixed_k_10'):
        if name not in grouped:
            continue
        paired[name] = {metric: mean_ci([a[metric] - b[metric] for a, b in zip(grouped['adaptive_k'], grouped[name])])
                        for metric in ('relevant_recall', 'relevant_f1', 'tokens')}
    return output, {'questions': len(rows), 'summary': summary, 'adaptive_minus_baseline': paired,
                    'retrieval_seconds': sum(r['retrieval_seconds'] for r in rows),
                    'chunks_min_median_max': [min(r['num_chunks'] for r in rows),
                                              statistics.median(r['num_chunks'] for r in rows),
                                              max(r['num_chunks'] for r in rows)],
                    'gap_rank_1_count': sum(r['k_before_buffer'] == 1 for r in rows),
                    'metric': 'Item-level overlap against relevant_indices (query-referenced table rows).',
                    'interval_caveat': 'Exploratory paired bootstrap, 2000 resamples; 18 questions per split, 5 splits.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    results, summary = analyze(rows)
    args.out_prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    with args.out_prefix.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
