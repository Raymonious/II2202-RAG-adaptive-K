"""Small local Ollama reader check on precomputed QASPER retrieval outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import string
import time
from urllib.request import Request, urlopen


def call(path, payload=None):
    request = Request('http://127.0.0.1:11434' + path,
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=600) as response:
        return json.load(response)


def answer_text(annotation):
    if annotation.get('unanswerable'):
        return 'Unanswerable'
    if annotation.get('extractive_spans'):
        return ', '.join(annotation['extractive_spans'])
    if annotation.get('free_form_answer'):
        return annotation['free_form_answer']
    if annotation.get('yes_no') is not None:
        return 'Yes' if annotation['yes_no'] else 'No'
    raise ValueError('Missing reference answer')


def words(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\b(a|an|the)\b', ' ', text).split()


def answer_f1(prediction, reference):
    p, r = words(prediction), words(reference)
    overlap = sum((Counter(p) & Counter(r)).values())
    return 2 * overlap / (len(p) + len(r)) if overlap else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('retrieval', type=Path)
    parser.add_argument('--questions', type=int, default=10)
    parser.add_argument('--model', default='llama3.2:latest')
    parser.add_argument('--num-ctx', type=int, default=16384)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.retrieval.read_text().splitlines()][:args.questions]
    config = {'model': args.model, 'questions': len(rows), 'model_details': call('/api/show', {'model': args.model}),
              'available_models': call('/api/tags'), 'server_version': call('/api/version'),
              'options': {'temperature': 0, 'seed': 42, 'num_ctx': args.num_ctx, 'num_predict': 256},
              'system': 'Answer the question using only the supplied paper excerpts. Give only a concise answer, without a preamble or citations. If the excerpts do not contain enough information, say Unanswerable.',
              'gold_policy': 'Union of unique evidence paragraphs from all reference annotations, in original source order; diagnostic only.',
              'answer_metric': 'Maximum reference normalized token F1, following QASPER evaluator conventions.',
              'scope': 'Exploratory reader feasibility; first questions of seeded training sample. No final test claims.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        args.out.with_suffix('.config.json').write_text(json.dumps(config, indent=2))
        for index, row in enumerate(rows, 1):
            gold = set(e for a in row['reference_annotations'] for e in a['evidence'])
            contexts = {
                'gold_evidence': '\n\n'.join(sorted(gold, key=row['body'].index)),
                'adaptive_k': row['selections']['adaptive_k']['context'],
                'fixed_k_5': '\n\n'.join(row['chunks'][i]['text'] for i in row['ranked_chunk_ids'][:5]),
                'fixed_k_10': '\n\n'.join(row['chunks'][i]['text'] for i in row['ranked_chunk_ids'][:10]),
            }
            methods = ['adaptive_k']
            random.Random(42 + index).shuffle(methods)
            for method in methods:
                user = f"Paper excerpts:\n{contexts[method]}\n\nQuestion: {row['question']}\nAnswer:"
                # Byte-level BPE cannot use more ordinary tokens than UTF-8
                # bytes. Reserve another 1024 tokens for the chat template and
                # 256 for output; skip (never silently truncate) on overflow.
                bound = len((config['system'] + user).encode('utf-8')) + 1024 + 256
                if bound > config['options']['num_ctx']:
                    record = {'question_id': row['question_id'], 'paper_id': row['paper_id'],
                              'question': row['question'], 'method': method, 'prediction': None,
                              'references': None, 'answer_f1': None, 'context': contexts[method],
                              'skipped': True, 'skip_reason': 'context_bound_exceeded',
                              'conservative_bound_bytes': bound, 'wall_seconds': 0.0, 'ollama': None}
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                    output.flush()
                    print(f'{index}/{len(rows)} {method}: SKIPPED (context bound {bound} > {config["options"]["num_ctx"]})', flush=True)
                    continue
                start = time.perf_counter()
                result = call('/api/chat', {
                    'model': args.model, 'stream': False, 'keep_alive': '10m',
                    'messages': [{'role': 'system', 'content': config['system']},
                                 {'role': 'user', 'content': user}], 'options': config['options'],
                })
                prediction = result['message']['content']
                references = [answer_text(a) for a in row['reference_annotations']]
                record = {'question_id': row['question_id'], 'paper_id': row['paper_id'],
                          'question': row['question'], 'method': method, 'prediction': prediction,
                          'references': references, 'answer_f1': max(answer_f1(prediction, ref) for ref in references),
                          'context': contexts[method], 'wall_seconds': time.perf_counter() - start,
                          'ollama': {k: v for k, v in result.items() if k != 'message'}}
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                output.flush()
                print(f'{index}/{len(rows)} {method}: F1={record["answer_f1"]:.3f}, '
                      f'{record["wall_seconds"]:.1f}s, reason={result.get("done_reason")}', flush=True)


if __name__ == '__main__':
    main()
