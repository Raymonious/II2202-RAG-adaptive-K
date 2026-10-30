"""Simple breakpoint-based semantic chunking for the QASPER pilot.

The implementation follows the breakpoint idea evaluated by Qu et al. (2025):
1. split a document into sentences;
2. embed every sentence;
3. compute cosine distance between each pair of consecutive sentences;
4. insert a breakpoint when the distance is above a chosen percentile threshold.

It intentionally implements only the percentile threshold variant.  This keeps
it small enough to inspect and explain while still matching one of the methods
studied in the paper.
"""
from __future__ import annotations

import numpy as np


def sentence_spans(body, nlp):
    """Return non-empty sentence spans as dictionaries with source offsets."""
    doc = nlp(body)
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue

        # Remove leading/trailing whitespace while keeping offsets relative to
        # the original document.  The offsets are useful for evidence checks.
        leading = len(sent.text) - len(sent.text.lstrip())
        trailing = len(sent.text) - len(sent.text.rstrip())
        start = sent.start_char + leading
        end = sent.end_char - trailing
        if start < end:
            sentences.append({'start_char': start, 'end_char': end,
                              'text': body[start:end]})
    return sentences


def split_oversized_chunks(chunks, body, tokenizer, max_tokens):
    """Split only chunks that exceed the embedding model's token limit.

    Token offset mappings are used so every new piece keeps exact character
    offsets into the original QASPER document.  This matters for the evidence
    overlap evaluation later in the pipeline.
    """
    if max_tokens is None:
        return chunks, 0

    result = []
    forced_split_chunks = 0

    for chunk in chunks:
        if chunk['embedding_tokens'] <= max_tokens:
            result.append(chunk)
            continue

        forced_split_chunks += 1
        encoded = tokenizer(
            chunk['text'],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get('offset_mapping')
        if offsets is None:
            raise ValueError('Max-length safeguard requires a fast tokenizer with offset mappings.')

        token_start = 0
        while token_start < len(offsets):
            token_end = min(token_start + max_tokens, len(offsets))

            # Retokenizing a substring can occasionally change the token count
            # slightly at its boundary.  If so, shrink the piece until it fits.
            while True:
                rel_start = offsets[token_start][0]
                rel_end = offsets[token_end - 1][1]
                start_char = chunk['start_char'] + rel_start
                end_char = chunk['start_char'] + rel_end

                while start_char < end_char and body[start_char].isspace():
                    start_char += 1
                while end_char > start_char and body[end_char - 1].isspace():
                    end_char -= 1

                text = body[start_char:end_char]
                token_count = len(tokenizer.encode(text, add_special_tokens=False))
                if token_count <= max_tokens or token_end == token_start + 1:
                    break
                token_end -= 1

            if text:
                result.append({
                    'chunk_id': -1,  # reassigned below
                    'start_char': start_char,
                    'end_char': end_char,
                    'text': text,
                    'embedding_tokens': token_count,
                })
            token_start = token_end

    for i, chunk in enumerate(result):
        chunk['chunk_id'] = i

    return result, forced_split_chunks


def semantic_chunk_body(body, tokenizer, model, nlp, percentile=70.0,
                        batch_size=8, max_tokens=None):
    """Split ``body`` at large semantic changes between adjacent sentences.

    Sentence embeddings are L2-normalized by SentenceTransformer, so the dot
    product is cosine similarity.  Semantic distance is ``1 - similarity``.
    A breakpoint is placed after sentence i when its distance to sentence i+1
    is greater than the selected percentile of all adjacent distances in that
    document.

    Returns ``(chunks, info)``.  ``chunks`` has the same fields as the existing
    fixed-size chunker so the rest of the QASPER pipeline can stay unchanged.
    ``info`` stores the threshold and sentence distances for inspection.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError('percentile must be between 0 and 100')

    sentences = sentence_spans(body, nlp)
    if not sentences:
        return [], {'sentence_count': 0, 'threshold': None,
                    'distances': [], 'break_after_sentence': []}

    # A one-sentence document has no possible breakpoint.
    if len(sentences) == 1:
        text = sentences[0]['text']
        chunk = {
            'chunk_id': 0,
            'start_char': sentences[0]['start_char'],
            'end_char': sentences[0]['end_char'],
            'text': text,
            'embedding_tokens': len(tokenizer.encode(text, add_special_tokens=False)),
        }
        return [chunk], {'sentence_count': 1, 'threshold': None,
                         'distances': [], 'break_after_sentence': []}

    sentence_embeddings = model.encode(
        [s['text'] for s in sentences],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    similarities = np.sum(sentence_embeddings[:-1] * sentence_embeddings[1:], axis=1)
    # Numerical noise can push normalized dot products slightly outside [-1, 1].
    similarities = np.clip(similarities, -1.0, 1.0)
    distances = 1.0 - similarities
    threshold = float(np.percentile(distances, percentile))

    # Split after sentence i if the local semantic distance is unusually large.
    # Strict '>' avoids turning every equal-distance boundary into a split.
    break_after = {i for i, distance in enumerate(distances) if distance > threshold}

    chunks = []
    chunk_start = 0
    for i in range(len(sentences)):
        is_last = i == len(sentences) - 1
        if i in break_after or is_last:
            first = sentences[chunk_start]
            last = sentences[i]
            start_char, end_char = first['start_char'], last['end_char']
            text = body[start_char:end_char]
            chunks.append({
                'chunk_id': len(chunks),
                'start_char': start_char,
                'end_char': end_char,
                'text': text,
                'embedding_tokens': len(tokenizer.encode(text, add_special_tokens=False)),
            })
            chunk_start = i + 1

    chunk_count_before_safeguard = len(chunks)
    chunks, forced_split_chunks = split_oversized_chunks(
        chunks, body, tokenizer, max_tokens
    )

    info = {
        'sentence_count': len(sentences),
        'percentile': percentile,
        'threshold': threshold,
        'distances': [float(x) for x in distances],
        'break_after_sentence': sorted(break_after),
        'max_chunk_tokens': max_tokens,
        'chunk_count_before_max_length_safeguard': chunk_count_before_safeguard,
        'chunk_count_after_max_length_safeguard': len(chunks),
        'forced_max_length_split_count': forced_split_chunks,
    }
    return chunks, info
