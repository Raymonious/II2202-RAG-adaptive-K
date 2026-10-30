"""Summarize run_answers.py output: per-method answer F1 and context tokens.

Token counts for adaptive_k/fixed_k_5/fixed_k_10 are joined from the source
retrieval file (BGE-tokenizer embedding_tokens, as in analyze_retrieval.py).
gold_evidence is reported with F1 only: it is a diagnostic upper-bound context
built from reference annotations, not a retrieval method with a token budget.
"""
import argparse
import json
import statistics
from pathlib import Path

from analyze_retrieval import mean_ci

TOKEN_METHODS = ('adaptive_k', 'fixed_k_5', 'fixed_k_10')


def context_tokens(row, method):
    chunks = row['chunks']
    if method == 'adaptive_k':
        return row['selections']['adaptive_k']['retrieved_embedding_tokens']
    k = {'fixed_k_5': 5, 'fixed_k_10': 10}[method]
    return sum(chunks[i]['embedding_tokens'] for i in row['ranked_chunk_ids'][:k])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('answers', type=Path)
    parser.add_argument('retrieval', type=Path)
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()

    retrieval_by_qid = {}
    for line in args.retrieval.read_text().splitlines():
        row = json.loads(line)
        retrieval_by_qid[row['question_id']] = row

    answers = [json.loads(line) for line in args.answers.read_text().splitlines()]
    grouped_all = {}
    for record in answers:
        grouped_all.setdefault(record['method'], []).append(record)
    skipped_counts = {method: sum(1 for r in records if r.get('skipped'))
                       for method, records in grouped_all.items()}
    grouped = {method: [r for r in records if not r.get('skipped')]
               for method, records in grouped_all.items()}

    summary = {}
    for method, records in grouped.items():
        entry = {'questions': len(records), 'skipped': skipped_counts.get(method, 0),
                  'mean_answer_f1': statistics.mean(r['answer_f1'] for r in records) if records else None}
        if method in TOKEN_METHODS and records:
            tokens = [context_tokens(retrieval_by_qid[r['question_id']], method) for r in records]
            entry['mean_context_tokens'] = statistics.mean(tokens)
        summary[method] = entry

    paired = {}
    if 'adaptive_k' in grouped:
        adaptive_by_qid = {r['question_id']: r for r in grouped['adaptive_k']}
        for baseline in ('fixed_k_5', 'fixed_k_10'):
            if baseline not in grouped:
                continue
            baseline_by_qid = {r['question_id']: r for r in grouped[baseline]}
            common = sorted(set(adaptive_by_qid) & set(baseline_by_qid))
            f1_diff = [adaptive_by_qid[q]['answer_f1'] - baseline_by_qid[q]['answer_f1'] for q in common]
            token_diff = [context_tokens(retrieval_by_qid[q], 'adaptive_k') -
                          context_tokens(retrieval_by_qid[q], baseline) for q in common]
            paired[f'adaptive_minus_{baseline}'] = {
                'questions': len(common),
                'answer_f1': mean_ci(f1_diff),
                'context_tokens': mean_ci(token_diff),
            }

    result = {'answers_file': str(args.answers), 'retrieval_file': str(args.retrieval),
              'summary': summary, 'paired_diffs': paired,
              'interval_caveat': 'Exploratory paired bootstrap, 2000 resamples; one question per paper.'}

    args.out_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')

    lines = ['# Answer-level pilot summary', '',
             f'Source answers: `{args.answers}`  ', f'Source retrieval: `{args.retrieval}`', '',
             '| Method | Questions | Skipped (context overflow) | Mean answer F1 | Mean context tokens |',
             '|---|---:|---:|---:|---:|']
    for method in ('gold_evidence', 'adaptive_k', 'fixed_k_5', 'fixed_k_10'):
        if method not in summary:
            continue
        entry = summary[method]
        tokens = f"{entry['mean_context_tokens']:.1f}" if 'mean_context_tokens' in entry else 'n/a'
        f1 = f"{entry['mean_answer_f1']:.3f}" if entry['mean_answer_f1'] is not None else 'n/a'
        lines.append(f"| {method} | {entry['questions']} | {entry['skipped']} | {f1} | {tokens} |")
    lines.append('')
    for name, diff in paired.items():
        f1 = diff['answer_f1']
        tok = diff['context_tokens']
        lines.append(f"**{name}** (n={diff['questions']}): "
                     f"answer F1 mean diff {f1['mean_difference']:+.3f} "
                     f"(95% CI [{f1['paired_bootstrap_95_percent_interval'][0]:+.3f}, "
                     f"{f1['paired_bootstrap_95_percent_interval'][1]:+.3f}]); "
                     f"context tokens mean diff {tok['mean_difference']:+.1f} "
                     f"(95% CI [{tok['paired_bootstrap_95_percent_interval'][0]:+.1f}, "
                     f"{tok['paired_bootstrap_95_percent_interval'][1]:+.1f}])  ")
    args.out_prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
