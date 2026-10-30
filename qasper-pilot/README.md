# Run Adaptive-k retrieval on QASPER locally

This is a small **retrieval-only pilot** for the II2202 project. It implements
the largest-gap selection rule from Taguchi et al. and compares it with fixed-k
on identical fixed-size chunks. It does not yet implement semantic chunking,
answer generation, evidence metrics, or the complete four-condition experiment.
There are no API charges. Initial model downloads require internet and disk space.

## 1. Create an isolated Python environment

In Terminal:

```sh
cd /Users/raymond/Desktop/II2202/qasper-pilot
/Users/raymond/miniconda3/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.12 exists at that path on this Mac. The system `python3` is 3.9.6.
The dependency versions follow the original repository's relevant pins. This
environment avoids changing your existing conda packages. Do not install the
entire upstream repository just to run this pilot: FAISS and its other benchmark
dependencies are unnecessary here.

## 2. Obtain the data

The public v0.3 train/development archive was downloaded during setup and is
available as `data/qasper-train-dev-v0.3.tgz`. For a fresh setup, obtain it with:

```sh
mkdir -p data
curl -fL https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz -o data/qasper-train-dev-v0.3.tgz
```

The runner reads JSON directly from the archive without extracting files. The
URL and format come from the [QASPER loader](https://huggingface.co/datasets/allenai/qasper/blob/main/qasper.py).

Inspect three sampled questions without loading any model:

```sh
python run_qasper.py --data data/qasper-train-dev-v0.3.tgz --questions 3 --inspect
```

It should report 888 training papers and three selected questions. Sampling
uses seed 42, one question per paper. All reference annotations must indicate an
answerable question and supply exclusively textual evidence found verbatim in
the body. This restricted pilot subset is not the complete QASPER benchmark;
report the restriction if retained in the study. Reference answers never enter
the embeddings or retrieval scores.

## 3. Run one question first

```sh
python run_qasper.py --data data/qasper-train-dev-v0.3.tgz --questions 1 --device mps --out results/one-question.jsonl
```

The first run downloads `BAAI/bge-large-en-v1.5`, then uses the Mac GPU through
PyTorch MPS. The script defaults to caching the model in `data/hf-cache`.
The model is now downloaded and both example outputs already exist: choose a
new output filename when rerunning. If MPS is unavailable or errors,
rerun with `--device cpu` and a fresh output filename.

What happens for each question:

1. Join the paper's body paragraphs in source order, retaining source offsets.
2. Split into non-overlapping windows of 128 BGE-tokenizer tokens. This is an
   initial pilot choice, not an optimized setting. Titles and section headings
   are not prepended, nor is the abstract separately added.
3. Embed all chunks and the question and normalize their vectors.
4. Compute cosine similarities and sort descending.
5. Find the largest adjacent score drop, excluding the final
   `floor((number_of_chunks - 1) * 0.1)` gaps.
6. Select chunks through that gap and add five more, capped at the total count.
7. Save both Adaptive-k and fixed-k=5 selections from the same ranking.

For example, a gap after rank 3 selects 3 + 5 = 8 chunks if enough exist.
The code uses no query instruction, matching the upstream SentenceTransformer
path inspected locally. Adding an instruction later changes the configuration.

## 4. Inspect the actual retrieved passages

```sh
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('results/one-question.jsonl').read_text().splitlines()[0])
print('Question:', r['question'])
print('Available chunks:', len(r['chunks']))
print('k before buffer:', r['k_before_buffer'])
print('All chunks retrieved:', r['retrieves_everything'])
for method, result in r['selections'].items():
    print('\nMethod:', method)
    print('k:', result['k'])
    print('BGE tokens:', result['retrieved_embedding_tokens'])
    print('Context:', result['context'])
print('\nReference annotations:', r['reference_annotations'])
PY
```

Check that retrieved text is relevant to the question. Do not interpret
similarity scores as probabilities. The reference annotations allow manual
inspection; no retrieval or answer accuracy has been computed yet.

The JSONL includes source text, chunk offsets, all ranked scores, pre-buffer k,
both selections, and references. The adjacent `.config.json` records settings,
package versions, the data hash, and the resolved model commit when available.
Token counts use the **embedding tokenizer** and exclude prompt overhead; use
your eventual answer model's tokenizer for generation context and cost metrics.

## 5. Scale to ten questions

```sh
python run_qasper.py --data data/qasper-train-dev-v0.3.tgz --questions 10 --device mps --out results/pilot-10.jsonl
python -m pip freeze > results/environment.txt
```

Each completed question is immediately saved. Existing result files are not
overwritten. This pilot has no automatic resume or persistent embedding cache;
use a fresh filename to rerun. For another Terminal session, activate `.venv`
again before running commands.

Inspect how often Adaptive-k retrieves everything before scaling further.
Use training papers for configuration decisions and `--split dev` for a later
held-out evaluation. Do not tune on the evaluation results. Fixed-k=5 is only a
starter comparator; the study needs justified baselines, including a token-budget
comparison if making claims about context efficiency.

## 6. Add answer generation after retrieval works

Pass each record's question and `selections.adaptive_k.context` to your local
answer model. Use exactly the same prompt/model/decoding settings for the
`fixed_k` context. Keep reference annotations out of those prompts.

Before generating, count the full prompt with the answer model's tokenizer,
reserve space for the answer, and define an explicit overflow policy. Silently
truncating a selected context changes the retrieval condition. Save answers and
model settings, then use QASPER's answer evaluator. This runner intentionally
does not claim to provide that generation/evaluation stage.

## Verification and scientific scope

```sh
python check_selector.py
```

This checks selection against the actual local upstream threshold method on
203 score arrays, plus empty/singleton and no-buffer cases. The singleton policy
is a documented extension: upstream rejects fewer than two scores.

Execution verified on September 14, 2026: the isolated environment is installed,
the BGE model is cached, and both the one-question and ten-question runs completed
successfully on MPS. The selector comparison and `pip check` passed. The ten
records were checked for unique questions, finite descending similarity scores,
and valid selections. See `results/pilot-summary.md`, `results/summary.json`,
and `results/environment.txt`. Answer generation and formal accuracy evaluation
remain unimplemented. Retrieval timings exclude model download and loading.

This script is a standalone adaptation, not a reproduction of the entire paper:
the corpus, chunking, candidate pool and baselines differ. The original checkout
was left unchanged (commit `fc86f12d7786cc68e7bd21885457c06e07c7a38c`).

Sources:
- [Adaptive-k paper](https://aclanthology.org/2025.emnlp-main.1017/)
- [Original implementation](https://github.com/megagonlabs/adaptive-k-retrieval)
- [QASPER data](https://huggingface.co/datasets/allenai/qasper)
- [BGE model documentation](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Sentence Transformers quickstart](https://www.sbert.net/docs/quickstart.html)

Record this AI coding assistance in your project log. Review and verify the
implementation yourself before using it to produce research results.


## 7. Breakpoint-based semantic chunking

The pilot now also supports a simple breakpoint-based semantic chunker inspired
by Qu et al. (2025).  It follows the paper's basic procedure: split the paper
into sentences, embed every sentence, calculate cosine distance between each
pair of consecutive sentences, and create a new chunk when the distance is
above a percentile threshold.

The paper used spaCy's `en_core_web_sm` model for sentence splitting.  Install
spaCy and the English model once inside the virtual environment:

```sh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run a small semantic-chunking pilot with the same sampled questions as the
fixed-size run:

```sh
python run_qasper.py \
  --data data/qasper-train-dev-v0.3.tgz \
  --questions 10 \
  --device mps \
  --chunking semantic \
  --semantic-percentile 70 \
  --out results/semantic-10.jsonl
```

`70` is only a starting value, not a claimed optimum from the paper.  The
threshold is configurable because Qu et al. tested several percentile values.
For the final experiment, choose the value using training/pilot data rather
than changing it after looking at held-out evaluation results.  One reasonable
choice is to select a percentile that gives semantic chunks of roughly similar
average token length to the fixed-size baseline, reducing chunk-size as a
confound.

The semantic chunker is implemented in `semantic_chunking.py`.  The remainder
of the retrieval pipeline is unchanged, so `run_answers.py` and
`analyze_retrieval.py` can consume semantic-run JSONL files in the same way as
fixed-size runs.  Run fixed and semantic chunking into separate result files to
obtain the four conditions:

- fixed chunks + fixed-k
- fixed chunks + Adaptive-k
- semantic chunks + fixed-k
- semantic chunks + Adaptive-k

The implementation uses the same BGE model for sentence embeddings and chunk
retrieval embeddings.  Qu et al. evaluated BGE-large among their embedding
models, so this keeps the experiment simple while avoiding another model as a
confounding implementation choice.
