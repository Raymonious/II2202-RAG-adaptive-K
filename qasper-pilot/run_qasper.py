"""Local QASPER retrieval pilot; no answer generation or paid API calls.

Largest-gap selection follows the local Adaptive-k implementation's default
integer buffer and tail exclusion. This is a standalone adaptation, not the
authors' full experimental pipeline. See README.md for scope and provenance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tarfile
import time

from semantic_chunking import semantic_chunk_body


def load_papers(path, split):
    """Read original QASPER v0.3 JSON, directly or inside its release archive."""
    if path.name.endswith('.json'):
        return json.loads(path.read_text())
    filename = f'qasper-{split}-v0.3.json'
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if Path(member.name).name == filename and member.isfile():
                with archive.extractfile(member) as source:
                    return json.load(source)
    raise ValueError(f'{filename} not found in {path}')


def select_questions(papers, count, seed):
    """One eligible question per paper, for a small, reproducible pilot.

    Require every annotation to be answerable and have text-only evidence that
    can be found verbatim in the body. This deliberately restricted subset is
    NOT representative of the whole QASPER benchmark.
    """
    rng = random.Random(seed)
    paper_ids = sorted(papers)
    rng.shuffle(paper_ids)
    selected = []
    for paper_id in paper_ids:
        paper = papers[paper_id]
        body = '\n\n'.join(
            paragraph for section in paper['full_text']
            for paragraph in section['paragraphs'] if paragraph.strip()
        )
        eligible = []
        for qa in paper['qas']:
            annotations = [item['answer'] for item in qa['answers']]
            if annotations and all(
                not a.get('unanswerable', False)
                and a.get('evidence')
                and all(e.strip() and not e.startswith('FLOAT SELECTED')
                        and e in body for e in a['evidence'])
                for a in annotations
            ):
                eligible.append(qa)
        if eligible:
            selected.append((paper_id, paper, body, rng.choice(eligible)))
        if len(selected) == count:
            break
    return selected


def choose_k(sorted_scores, buffer=5, tail_fraction=0.1):
    """Return chunk counts before/after buffer (not zero-based gap indices).

    Match retriever.py: exclude floor((N-1)*tail_fraction) final gaps;
    select the first largest remaining drop, then add five chunks and clamp.
    A singleton returns one chunk; the upstream code rejects that case.
    """
    n = len(sorted_scores)
    if not n:
        return 0, 0
    if n == 1:
        return 1, 1
    drops = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
    excluded = int((n - 1) * tail_fraction)
    searchable = len(drops) - excluded
    gap_index = max(range(searchable), key=drops.__getitem__)
    before_buffer = gap_index + 1
    return before_buffer, min(n, before_buffer + buffer)


def chunk_body(body, tokenizer, size):
    """Non-overlapping windows of embedding-tokenizer tokens over body text.

    Retain source offsets so evidence can later be evaluated independently of
    chunk IDs. Section headings/title/abstract are not separately prepended.
    """
    # Tokenize the whole body only to find chunk offsets; it is never embedded
    # as one sequence. Actual chunk lengths are checked before model.encode.
    offsets = tokenizer(body, add_special_tokens=False,
                        return_offsets_mapping=True, truncation=False,
                        verbose=False)['offset_mapping']
    chunks = []
    for start in range(0, len(offsets), size):
        window = offsets[start:start + size]
        begin, end = window[0][0], window[-1][1]
        text = body[begin:end]
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        chunks.append({'chunk_id': len(chunks), 'start_char': begin,
                       'end_char': end, 'text': text, 'embedding_tokens': tokens})
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='train')
    parser.add_argument('--questions', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inspect', action='store_true', help='Inspect data only; no models required')
    parser.add_argument('--model', default='BAAI/bge-large-en-v1.5')
    parser.add_argument('--model-revision', default='main')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--chunk-tokens', type=int, default=128)
    parser.add_argument('--chunking', choices=['fixed', 'semantic'], default='fixed')
    parser.add_argument('--semantic-percentile', type=float, default=70.0,
                        help='Percentile threshold for breakpoint semantic chunking')
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--fixed-k', type=int, default=5)
    parser.add_argument('--buffer', type=int, default=5)
    parser.add_argument('--out', type=Path, default=Path('results/pilot.jsonl'))
    args = parser.parse_args()
    if min(args.questions, args.chunk_tokens, args.batch_size, args.fixed_k) < 1 or args.buffer < 0:
        parser.error('Counts must be positive; buffer must be nonnegative.')
    papers = load_papers(args.data, args.split)
    samples = select_questions(papers, args.questions, args.seed)
    print(f'Loaded {len(papers)} papers; selected {len(samples)} eligible questions.')
    if len(samples) < args.questions:
        raise ValueError('Not enough eligible papers for requested question count.')
    if args.inspect:
        for paper_id, paper, body, qa in samples:
            print(json.dumps({'paper_id': paper_id, 'title': paper['title'],
                              'body_characters': len(body), 'question': qa['question']}, ensure_ascii=False))
        return

    os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent / 'data' / 'hf-cache'))
    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; rerun with --device cpu.')
    model = SentenceTransformer(args.model, revision=args.model_revision, device=args.device)
    tokenizer = model.tokenizer
    nlp = None
    if args.chunking == 'semantic':
        try:
            import spacy
            nlp = spacy.load(args.spacy_model)
        except (ImportError, OSError) as error:
            raise RuntimeError(
                'Semantic chunking requires spaCy and an English model. '
                'Install with: pip install spacy && python -m spacy download en_core_web_sm'
            ) from error
    max_payload = model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    if args.chunk_tokens > max_payload:
        raise ValueError(f'Chunk size exceeds embedding model payload limit ({max_payload}).')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Prevent an accidental replacement of completed experimental outputs.
    with args.out.open('x') as output:
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update({
            'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
            'tail_fraction': 0.1,
            'query_instruction': '',
            'token_count_basis': 'embedding tokenizer; excludes special tokens and prompt overhead',
            'resolved_model_commit': getattr(model[0].auto_model.config, '_commit_hash', None),
            'versions': {p: importlib.metadata.version(p) for p in
                         ['sentence-transformers', 'transformers', 'torch']},
        })
        args.out.with_suffix('.config.json').write_text(json.dumps(config, indent=2))
        for number, (paper_id, paper, body, qa) in enumerate(samples, 1):
            question_start = time.perf_counter()
            semantic_info = None
            if args.chunking == 'fixed':
                chunks = chunk_body(body, tokenizer, args.chunk_tokens)
            else:
                chunks, semantic_info = semantic_chunk_body(
                    body, tokenizer, model, nlp, args.semantic_percentile,
                    args.batch_size, max_tokens=max_payload
                )
            if not chunks:
                raise ValueError(f'No chunks for {paper_id}')
            if any(c['embedding_tokens'] > max_payload for c in chunks):
                if args.chunking == 'fixed':
                    raise ValueError('Fixed chunk is too long; reduce --chunk-tokens.')
                raise RuntimeError('Semantic max-length safeguard failed.')
            if len(tokenizer.encode(qa['question'], add_special_tokens=False)) > max_payload:
                raise ValueError('Question exceeds embedding limit; do not silently truncate.')
            embeddings = model.encode([c['text'] for c in chunks], batch_size=args.batch_size,
                                      normalize_embeddings=True, convert_to_numpy=True)
            query = model.encode([qa['question']], normalize_embeddings=True, convert_to_numpy=True)[0]
            scores = (embeddings @ query).tolist()
            ranking = sorted(range(len(chunks)), key=lambda i: -scores[i])
            sorted_scores = [scores[i] for i in ranking]
            before_buffer, k = choose_k(sorted_scores, args.buffer)
            selections = {}
            for name, ids in [('adaptive_k', ranking[:k]), ('fixed_k', ranking[:args.fixed_k])]:
                selections[name] = {
                    'k': len(ids), 'chunk_ids': ids,
                    'retrieved_embedding_tokens': sum(chunks[i]['embedding_tokens'] for i in ids),
                    'context': '\n\n'.join(chunks[i]['text'] for i in ids),
                }
            record = {
                'paper_id': paper_id, 'title': paper['title'],
                'chunking_method': args.chunking,
                'semantic_chunking': semantic_info,
                'question_id': qa['question_id'], 'question': qa['question'],
                'reference_annotations': [a['answer'] for a in qa['answers']],
                'body': body, 'chunks': chunks, 'ranked_chunk_ids': ranking,
                'sorted_scores': sorted_scores, 'k_before_buffer': before_buffer,
                'retrieves_everything': k == len(chunks), 'selections': selections,
                'retrieval_seconds': time.perf_counter() - question_start,
            }
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            print(f'{number}/{len(samples)} {paper_id}: {args.chunking}, {len(chunks)} chunks; '
                  f'gap selects {before_buffer}, buffered k={k}', flush=True)
    print(f'Saved {args.out} and {args.out.with_suffix(".config.json")}')


if __name__ == '__main__':
    main()
