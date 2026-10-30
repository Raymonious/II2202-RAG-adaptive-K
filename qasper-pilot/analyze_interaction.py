import argparse
import json
from pathlib import Path

from analyze_retrieval import mean_ci


def load_answers(path):
    records = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]

    by_method = {}

    for record in records:
        if record.get("skipped"):
            continue

        method = record["method"]
        qid = record["question_id"]

        by_method.setdefault(method, {})[qid] = record

    return by_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed_answers", type=Path)
    parser.add_argument("semantic_answers", type=Path)
    args = parser.parse_args()

    fixed = load_answers(args.fixed_answers)
    semantic = load_answers(args.semantic_answers)

    # A = fixed chunks + fixed-k=5
    A = fixed["fixed_k_5"]

    # B = fixed chunks + Adaptive-k
    B = fixed["adaptive_k"]

    # C = semantic chunks + fixed-k=5
    C = semantic["fixed_k_5"]

    # D = semantic chunks + Adaptive-k
    D = semantic["adaptive_k"]

    # Only questions that exist in ALL FOUR conditions
    common = sorted(
        set(A)
        & set(B)
        & set(C)
        & set(D)
    )

    ba = []
    dc = []
    interaction = []

    for qid in common:
        a = A[qid]["answer_f1"]
        b = B[qid]["answer_f1"]
        c = C[qid]["answer_f1"]
        d = D[qid]["answer_f1"]

        effect_fixed = b - a
        effect_semantic = d - c

        ba.append(effect_fixed)
        dc.append(effect_semantic)

        interaction.append(
            effect_semantic - effect_fixed
        )

    ba_result = mean_ci(ba)
    dc_result = mean_ci(dc)
    interaction_result = mean_ci(interaction)

    print(f"Common questions: {len(common)}")
    print()

    print("B - A: Adaptive-k effect with fixed chunks")
    print(json.dumps(ba_result, indent=2))
    print()

    print("D - C: Adaptive-k effect with semantic chunks")
    print(json.dumps(dc_result, indent=2))
    print()

    print("(D-C) - (B-A): interaction")
    print(json.dumps(interaction_result, indent=2))


if __name__ == "__main__":
    main()