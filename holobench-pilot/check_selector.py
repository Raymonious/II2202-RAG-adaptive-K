"""Check the extended (head+tail exclusion) selector against upstream.

Same technique as qasper-pilot/check_selector.py: AST-extract the actual
find_threshold_largest_gap method from the local upstream checkout so no
faiss/config.json/torch-heavy import is needed, and compare against
run_holobench.py's choose_k across many score arrays, this time exercising
ignore_extreme > 0 (HoloBench's repo default) as well as the tail-only case
already verified for QASPER.

Run with Python that has torch installed. No models or network required.
"""
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from run_holobench import choose_k


source = Path(__file__).resolve().parents[1] / 'adaptive-k-retrieval/adaptive-k-retrieval/retriever.py'
tree = ast.parse(source.read_text())
retriever = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Retriever')
method = next(n for n in retriever.body if isinstance(n, ast.FunctionDef)
              and n.name == 'find_threshold_largest_gap')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
original = namespace['find_threshold_largest_gap']

rng = random.Random(42)
cases = [[0.9, 0.8, 0.3, 0.2], [0.5] * 12,
         [1 - i / 100 for i in range(19)] + [-1.0]]
cases += [sorted([rng.uniform(-1, 1) for _ in range(n)], reverse=True)
          for n in range(2, 202)]

checked = 0
for values in cases:
    scores = torch.tensor(values, dtype=torch.float64)
    for extreme in (0.0, 0.05, 0.1, 0.2):
        settings = SimpleNamespace(ignore_below_median=False, ignore_extreme=extreme,
                                   ignore_extreme_tail=0.1)
        gap_index = int(original(settings, scores))
        assert choose_k(values, extreme_fraction=extreme) == (gap_index + 1, min(len(values), gap_index + 6)), \
            f'mismatch at extreme={extreme}, n={len(values)}'
        checked += 1

assert choose_k([]) == (0, 0)
assert choose_k([0.5]) == (1, 1)
assert choose_k([0.9, 0.8, 0.3, 0.2], buffer=0) == (2, 2)
print(f'Passed {checked} comparisons against upstream (incl. ignore_extreme > 0), plus boundary checks.')
