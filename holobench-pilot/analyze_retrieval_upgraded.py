"""Stronger HoloBench retrieval analysis using matched token-budget baselines.

Reads an existing run_holobench.py JSONL file. No re-embedding or retrieval rerun is
required because ranked_chunk_ids and per-chunk token counts are already saved.

Adds exact BGE-tokenizer budget baselines at 1k, 5k, and 10k tokens. These are
internal matched-cost baselines; they are not a literal reproduction of the
authors' fixed-token implementation, which uses an approximate char/token ratio.
"""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

TOKEN_BUDGETS = (1000, 5000, 10000)


def coverage(retrieved_ids, gold_ids):
    retrieved = set(retrieved_ids)
    gold = set(gold_ids)
    overlap = len(retrieved & gold)
    recall = overlap / len(gold) if gold else 0.0
    precision = overlap / len(retrieved) if retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return recall, precision, f1


def ids_within_token_budget(row, budget):
    """Take ranked chunks until adding the next chunk would exceed budget."""
    chosen = []
    total = 0
    for idx in row["ranked_chunk_ids"]:
        t = row["chunk_tokens"][idx]
        if total + t > budget:
            break
        chosen.append(idx)
        total += t
    return chosen


def selections(row):
    # Existing experimental selections
    for name, entry in row["selections"].items():
        yield name, entry["chunk_ids"]

    # Stronger fixed token-budget baselines
    for budget in TOKEN_BUDGETS:
        yield f"fixed_tokens_{budget}", ids_within_token_budget(row, budget)

    # Diagnostics / upper retrieval boundary
    yield "gap_without_buffer_diagnostic", row["ranked_chunk_ids"][:row["k_before_buffer"]]
    yield "full_context", row["ranked_chunk_ids"]


def mean_ci(differences):
    rng = random.Random(2026)
    n = len(differences)
    samples = sorted(
        statistics.mean(rng.choices(differences, k=n))
        for _ in range(2000)
    )
    return {
        "mean_difference": statistics.mean(differences),
        "paired_bootstrap_95_percent_interval": [samples[49], samples[1949]],
    }


def summarize(values):
    return {
        key: statistics.mean(r[key] for r in values)
        for key in ("k", "tokens", "relevant_recall", "relevant_precision", "relevant_f1")
    }


def analyze(rows):
    output = []

    for row in rows:
        gold = row["relevant_indices"]
        for name, ids in selections(row):
            recall, precision, f1 = coverage(ids, gold)
            output.append({
                "question_id": row["question_id"],
                "split": row["split"],
                "method": name,
                "k": len(ids),
                "tokens": sum(row["chunk_tokens"][i] for i in ids),
                "relevant_recall": recall,
                "relevant_precision": precision,
                "relevant_f1": f1,
                "full_selection": len(ids) == row["num_chunks"],
            })

    method_names = list(dict.fromkeys(r["method"] for r in output))
    grouped = {
        name: [r for r in output if r["method"] == name]
        for name in method_names
    }

    summary = {}
    for name, values in grouped.items():
        summary[name] = summarize(values)
        summary[name]["full_selection_count"] = sum(r["full_selection"] for r in values)
        summary[name]["zero_relevant_overlap_count"] = sum(
            r["relevant_recall"] == 0 for r in values
        )
        summary[name]["at_least_90_percent_recall_count"] = sum(
            r["relevant_recall"] >= 0.9 for r in values
        )

    # Paired Adaptive-k comparisons against token budgets and full context.
    paired = {}
    adaptive = grouped["adaptive_k"]
    for baseline in [f"fixed_tokens_{b}" for b in TOKEN_BUDGETS] + ["full_context"]:
        base = grouped[baseline]
        paired[baseline] = {
            metric: mean_ci([
                a[metric] - b[metric]
                for a, b in zip(adaptive, base)
            ])
            for metric in ("relevant_recall", "relevant_f1", "tokens")
        }

    # Per-domain/split check for the closest-cost fixed baseline (5k).
    by_split = {}
    splits = sorted({r["split"] for r in output})
    for split in splits:
        by_split[split] = {}
        for method in ("adaptive_k", "fixed_tokens_5000"):
            vals = [r for r in grouped[method] if r["split"] == split]
            by_split[split][method] = summarize(vals)

    return output, {
        "questions": len(rows),
        "summary": summary,
        "adaptive_minus_baseline": paired,
        "by_split_adaptive_vs_fixed_tokens_5000": by_split,
        "metric": "Item-level overlap against relevant_indices.",
        "token_budget_definition": (
            "Exact BGE-tokenizer chunk counts from the saved pilot; ranked chunks are "
            "added until the next chunk would exceed the stated budget."
        ),
        "interval_caveat": (
            "Exploratory paired bootstrap, 2000 resamples; same questions for every method."
        ),
    }


def fmt_ci(x, digits=3):
    lo, hi = x["paired_bootstrap_95_percent_interval"]
    return (
        f'{x["mean_difference"]:+.{digits}f} '
        f'[{lo:+.{digits}f}, {hi:+.{digits}f}]'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]

    results, report = analyze(rows)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    with args.out_prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    methods = [
        "adaptive_k",
        "fixed_tokens_1000",
        "fixed_tokens_5000",
        "fixed_tokens_10000",
        "full_context",
    ]

    lines = [
        "# Upgraded HoloBench retrieval analysis",
        "",
        f"Questions: {report['questions']}",
        "",
        "| Method | Mean k | Mean tokens | Recall | Precision | Relevant F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method in methods:
        s = report["summary"][method]
        lines.append(
            f"| {method} | {s['k']:.1f} | {s['tokens']:.1f} | "
            f"{s['relevant_recall']:.3f} | {s['relevant_precision']:.3f} | "
            f"{s['relevant_f1']:.3f} |"
        )

    lines += [
        "",
        "## Paired Adaptive-k minus token-budget baselines",
        "",
        "| Baseline | Δ Recall (95% CI) | Δ Relevant F1 (95% CI) | Δ Tokens (95% CI) |",
        "|---|---:|---:|---:|",
    ]

    for baseline in ("fixed_tokens_1000", "fixed_tokens_5000", "fixed_tokens_10000"):
        p = report["adaptive_minus_baseline"][baseline]
        lines.append(
            f"| {baseline} | {fmt_ci(p['relevant_recall'])} | "
            f"{fmt_ci(p['relevant_f1'])} | {fmt_ci(p['tokens'], 1)} |"
        )

    lines += [
        "",
        "## Per-split check: Adaptive-k vs fixed 5k-token budget",
        "",
        "| Split | Adaptive recall | 5k recall | Adaptive F1 | 5k F1 | Adaptive tokens | 5k tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for split, d in report["by_split_adaptive_vs_fixed_tokens_5000"].items():
        a = d["adaptive_k"]
        b = d["fixed_tokens_5000"]
        lines.append(
            f"| {split} | {a['relevant_recall']:.3f} | {b['relevant_recall']:.3f} | "
            f"{a['relevant_f1']:.3f} | {b['relevant_f1']:.3f} | "
            f"{a['tokens']:.0f} | {b['tokens']:.0f} |"
        )

    lines += [
        "",
        "**Important:** these fixed token budgets use the exact BGE tokenizer counts "
        "stored by the local pilot. They are intended as fair matched-cost baselines "
        "inside this study, not as a literal reproduction of the paper's approximate "
        "fixed-token implementation.",
    ]

    md = "\n".join(lines) + "\n"
    args.out_prefix.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
