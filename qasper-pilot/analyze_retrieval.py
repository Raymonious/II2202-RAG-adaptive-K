"""Descriptive feasibility analysis; character coverage is NOT QASPER Evidence F1."""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from run_qasper import load_papers, select_questions


def positions(text, intervals):
    return {i for start, end in intervals for i in range(start, end) if not text[i].isspace()}


def evidence_positions(body, evidence):
    intervals = []
    for passage in evidence:
        start = body.find(passage)
        if start < 0:
            raise ValueError('Evidence not found verbatim')
        if body.find(passage, start + 1) >= 0:
            raise ValueError('Ambiguous duplicated evidence paragraph')
        intervals.append((start, start + len(passage)))
    return positions(body, intervals)


def coverage(retrieved, gold):
    overlap = len(retrieved & gold)
    recall = overlap / len(gold) if gold else 0.0
    precision = overlap / len(retrieved) if retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return {'evidence_character_recall': recall, 'evidence_character_precision': precision,
            'evidence_character_f1': f1}


def selections(row):
    ranking = row['ranked_chunk_ids']
    yield 'adaptive_k', row['selections']['adaptive_k']['chunk_ids']
    yield 'gap_without_buffer_diagnostic', ranking[:row['k_before_buffer']]
    for k in (1, 3, 5, 10, 23, 36):
        yield f'fixed_k_{k}', ranking[:k]
    for budget in (512, 1024, 1847, 2048, 2883):
        chosen, used = [], 0
        for i in ranking:
            tokens = row['chunks'][i]['embedding_tokens']
            if used + tokens > budget:
                break
            chosen.append(i)
            used += tokens
        yield f'fixed_tokens_{budget}', chosen
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
        references = [evidence_positions(row['body'], a['evidence']) for a in row['reference_annotations']]
        for name, ids in selections(row):
            chosen = positions(row['body'], [(row['chunks'][i]['start_char'], row['chunks'][i]['end_char']) for i in ids])
            # Keep annotations separate; choose the best F1 reference, then
            # report P/R from that same reference. Never union alternate answers.
            scores = max((coverage(chosen, gold) for gold in references), key=lambda x: x['evidence_character_f1'])
            output.append({'question_id': row['question_id'], 'paper_id': row['paper_id'],
                           'method': name, 'k': len(ids),
                           'tokens': sum(row['chunks'][i]['embedding_tokens'] for i in ids),
                           'full_selection': len(ids) == len(row['chunks']), **scores})
    grouped = {name: [r for r in output if r['method'] == name] for name, _ in selections(rows[0])}
    summary = {}
    for name, values in grouped.items():
        summary[name] = {key: statistics.mean(r[key] for r in values) for key in
                         ['k', 'tokens', 'evidence_character_recall', 'evidence_character_precision', 'evidence_character_f1']}
        summary[name]['full_selection_count'] = sum(r['full_selection'] for r in values)
        summary[name]['zero_evidence_overlap_count'] = sum(r['evidence_character_recall'] == 0 for r in values)
        summary[name]['at_least_90_percent_evidence_count'] = sum(r['evidence_character_recall'] >= .9 for r in values)
    paired = {}

    for name in (
    'fixed_k_5',
    'fixed_k_10',
    'fixed_k_23',
    'fixed_k_36',
    'fixed_tokens_1024',):
        paired[name] = {metric: mean_ci([a[metric] - b[metric] for a, b in zip(grouped['adaptive_k'], grouped[name])])
                        for metric in ('evidence_character_recall', 'evidence_character_f1', 'tokens')}
    return output, {'questions': len(rows), 'summary': summary, 'adaptive_minus_baseline': paired,
                    'retrieval_seconds': sum(r['retrieval_seconds'] for r in rows),
                    'chunks_min_median_max': [min(len(r['chunks']) for r in rows), statistics.median(len(r['chunks']) for r in rows), max(len(r['chunks']) for r in rows)],
                    'gap_rank_1_count': sum(r['k_before_buffer'] == 1 for r in rows),
                    'metric': 'Non-whitespace character overlap at source offsets, against annotated evidence paragraphs; best F1 annotation. Not official Evidence F1.',
                    'budget_rule': 'Largest ranked prefix whose summed BGE-token counts do not exceed the budget.',
                    'interval_caveat': 'Exploratory paired bootstrap, 2000 resamples; one question per paper. No multiplicity adjustment.'}


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
