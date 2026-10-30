"""Check the standalone selector against the actual local upstream method.

Run with Python that has torch installed. No models or network required.
"""
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from run_qasper import choose_k


source = Path(__file__).resolve().parents[1] / 'adaptive-k-retrieval/adaptive-k-retrieval/retriever.py'
tree = ast.parse(source.read_text())
retriever = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Retriever')
method = next(n for n in retriever.body if isinstance(n, ast.FunctionDef)
              and n.name == 'find_threshold_largest_gap')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
original = namespace['find_threshold_largest_gap']
settings = SimpleNamespace(ignore_below_median=False, ignore_extreme=0.0, ignore_extreme_tail=0.1)
rng = random.Random(42)
cases = [[0.9, 0.8, 0.3, 0.2], [0.5] * 12,
         [1 - i / 100 for i in range(19)] + [-1.0]]
cases += [sorted([rng.uniform(-1, 1) for _ in range(n)], reverse=True)
          for n in range(2, 202)]
for values in cases:
    scores = torch.tensor(values, dtype=torch.float64)
    gap_index = int(original(settings, scores))
    assert choose_k(values) == (gap_index + 1, min(len(values), gap_index + 6))
assert choose_k([]) == (0, 0)
assert choose_k([0.5]) == (1, 1)
assert choose_k([0.9, 0.8, 0.3, 0.2], buffer=0) == (2, 2)
print(f'Passed {len(cases)} comparisons against upstream, plus boundary checks.')
