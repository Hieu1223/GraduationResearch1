"""
confusion_matrix.py — compute and display confusion matrix + metrics for each trait.

Usage:
    python confusion_matrix.py                  # all traits
    python confusion_matrix.py --traits O C E   # selected traits
    python confusion_matrix.py --results-root results/
"""

import argparse
import json
import os
from pathlib import Path


# ------------------------------------------------------------------
# Config — must match run_all.py
# ------------------------------------------------------------------

TRAIT_CONFIG = {
    "O": {"trait": "Openness",          "results_dir": "results/Openness"},
    "C": {"trait": "Conscientiousness", "results_dir": "results/Conscientiousness"},
    "E": {"trait": "Extraversion",      "results_dir": "results/Extraversion"},
    "A": {"trait": "Agreeableness",     "results_dir": "results/Agreeableness"},
    "N": {"trait": "Neuroticism",       "results_dir": "results/Neuroticism"},
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_results(results_dir: str) -> list[dict]:
    """Load all per-sample JSON files from a results directory."""
    records = []
    path = Path(results_dir)
    if not path.exists():
        return records
    for f in sorted(path.glob("*.json"), key=lambda p: int(p.stem)):
        with open(f, encoding="utf-8") as fp:
            records.append(json.load(fp))
    return records


def compute_metrics(records: list[dict]) -> dict:
    """
    Compute confusion matrix counts and derived metrics.
    Positive class = HIGH, Negative class = LOW.
    """
    tp = fp = tn = fn = unknown = 0

    for i,r in enumerate(records):
        pred = r.get("prediction", "UNKNOWN").upper()
        gt   = r.get("ground_truth", "UNKNOWN").upper()

        if pred == "UNKNOWN":
            print(f"Sample {i}")
            unknown += 1
            continue

        if pred == "HIGH" and gt == "HIGH":
            tp += 1
        elif pred == "HIGH" and gt == "LOW":
            fp += 1
        elif pred == "LOW" and gt == "LOW":
            tn += 1
        elif pred == "LOW" and gt == "HIGH":
            fn += 1

    total = tp + fp + tn + fn

    precision_high = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_high    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_high        = (
        2 * precision_high * recall_high / (precision_high + recall_high)
        if (precision_high + recall_high) > 0 else 0.0
    )

    precision_low = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_low    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_low        = (
        2 * precision_low * recall_low / (precision_low + recall_low)
        if (precision_low + recall_low) > 0 else 0.0
    )

    accuracy = (tp + tn) / total if total > 0 else 0.0
    macro_f1 = (f1_high + f1_low) / 2

    return {
        "total":          total,
        "unknown":        unknown,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision_high": precision_high,
        "recall_high":    recall_high,
        "f1_high":        f1_high,
        "precision_low":  precision_low,
        "recall_low":     recall_low,
        "f1_low":         f1_low,
        "accuracy":       accuracy,
        "macro_f1":       macro_f1,
    }


def print_confusion_matrix(trait_name: str, m: dict) -> None:
    w = 54
    sep = "=" * w

    print(f"\n{sep}")
    print(f"  {trait_name.upper()}")
    print(sep)

    if m["total"] == 0:
        print("  No results found.")
        return

    # Confusion matrix table
    print(f"  {'':20s} {'Predicted HIGH':>14}  {'Predicted LOW':>13}")
    print(f"  {'-'*50}")
    print(f"  {'Actual HIGH':20s} {'TP':>6} = {m['tp']:<6}  {'FN':>6} = {m['fn']:<6}")
    print(f"  {'Actual LOW':20s} {'FP':>6} = {m['fp']:<6}  {'TN':>6} = {m['tn']:<6}")
    print(f"  {'-'*50}")

    # Metrics
    print(f"\n  {'Metric':<22} {'HIGH':>8}  {'LOW':>8}")
    print(f"  {'-'*40}")
    print(f"  {'Precision':<22} {m['precision_high']:>8.3f}  {m['precision_low']:>8.3f}")
    print(f"  {'Recall':<22} {m['recall_high']:>8.3f}  {m['recall_low']:>8.3f}")
    print(f"  {'F1':<22} {m['f1_high']:>8.3f}  {m['f1_low']:>8.3f}")
    print(f"  {'-'*40}")
    print(f"  {'Accuracy':<22} {m['accuracy']:>8.3f}")
    print(f"  {'Macro F1':<22} {m['macro_f1']:>8.3f}")
    print(f"  {'Total samples':<22} {m['total']:>8}")

    if m["unknown"] > 0:
        print(f"  {'UNKNOWN predictions':<22} {m['unknown']:>8}  (excluded above)")

    print(sep)


def print_summary_table(results: dict[str, dict]) -> None:
    """Print a compact cross-trait summary table."""
    w = 70
    print(f"\n{'='*w}")
    print("  SUMMARY — ALL TRAITS")
    print(f"{'='*w}")
    print(f"  {'Trait':<20} {'Acc':>6}  {'F1-H':>6}  {'F1-L':>6}  {'MacroF1':>8}  {'N':>6}")
    print(f"  {'-'*60}")
    for key, m in results.items():
        name = TRAIT_CONFIG[key]["trait"]
        if m["total"] == 0:
            print(f"  {name:<20} {'—':>6}  {'—':>6}  {'—':>6}  {'—':>8}  {'—':>6}")
        else:
            print(
                f"  {name:<20} "
                f"{m['accuracy']:>6.3f}  "
                f"{m['f1_high']:>6.3f}  "
                f"{m['f1_low']:>6.3f}  "
                f"{m['macro_f1']:>8.3f}  "
                f"{m['total']:>6}"
            )
    print(f"{'='*w}\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Confusion matrix for OCEAN traits.")
    parser.add_argument(
        "--traits",
        nargs="+",
        choices=list(TRAIT_CONFIG.keys()),
        default=list(TRAIT_CONFIG.keys()),
        help="Traits to evaluate (default: all).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Override root directory for results (overrides per-trait paths).",
    )
    args = parser.parse_args()

    all_metrics = {}

    for key in args.traits:
        cfg = TRAIT_CONFIG[key]
        results_dir = (
            os.path.join(args.results_root, cfg["trait"].lower())
            if args.results_root
            else cfg["results_dir"]
        )

        records = load_results(results_dir)
        m = compute_metrics(records)
        all_metrics[key] = m

        print_confusion_matrix(cfg["trait"], m)

    if len(args.traits) > 1:
        print_summary_table(all_metrics)


if __name__ == "__main__":
    main()