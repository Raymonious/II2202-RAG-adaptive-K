from analyze_retrieval import coverage, evidence_positions, positions, selections

text = 'abc def ghi'
gold = evidence_positions(text, ['def ghi'])
assert len(gold) == 6
assert coverage(positions(text, [(4, 7)]), gold)['evidence_character_recall'] == .5
assert coverage(positions(text, [(0, 3)]), gold)['evidence_character_recall'] == 0
assert coverage(positions(text, [(4, 6), (6, 11)]), gold)['evidence_character_recall'] == 1
assert coverage(positions(text, [(0, 11)]), gold)['evidence_character_precision'] == 2 / 3
assert len(positions(text, [(4, 10), (5, 11)])) == 6
assert coverage(set(), gold)['evidence_character_f1'] == 0
try:
    evidence_positions('abc abc', ['abc'])
except ValueError:
    pass
else:
    raise AssertionError('Duplicated evidence must not be silently mapped')
print('Coverage checks passed: partial/disjoint/full/overlapping spans and ambiguous evidence.')
