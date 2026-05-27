"""
probability_mass_analysis.py  —  Sprint 3 (Part 2)
====================================================
Builds on token_rank_analysis.py with deeper probability analysis.

Three questions:
1. How much probability does DPO destroy? P(correct) per stage
2. Does DPO make the model more uncertain? Entropy per stage
3. What does DPO put in place of correct tokens? Replacement semantics

Usage (no GPU needed — runs on sprint3 CSV output):
    python scripts/probability_mass_analysis.py \
        --rank_results outputs/sprint3/token_rank_full.csv \
        --output_dir   outputs/sprint3
"""

import csv
import json
import os
import argparse
from collections import defaultdict


def load_rank_results(csv_path: str) -> list[dict]:
    results = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row["correct_token_rank"] = int(row["correct_token_rank"])
            row["correct_token_prob"] = float(row["correct_token_prob"])
            row["entropy"] = float(row["entropy"])
            row["judge_score"] = int(row["judge_score"])
            row["suppressed"] = row["suppressed"] == "True"
            row["forgotten"] = row["forgotten"] == "True"
            row["correctly_generated"] = row["correctly_generated"] == "True"
            results.append(row)
    return results


def prob_mass_by_stage(results: list[dict]) -> dict:
    """Per-stage probability statistics."""
    by_stage = defaultdict(list)
    for r in results:
        by_stage[r["model_stage"]].append(r)

    out = {}
    for stage, rows in sorted(by_stage.items()):
        probs = sorted(r["correct_token_prob"] for r in rows)
        entropies = [r["entropy"] for r in rows]
        passing = [r for r in rows if r["judge_score"] >= 2]
        failing = [r for r in rows if r["judge_score"] <= 1]

        out[stage] = {
            "n": len(rows),
            "mean_prob": round(sum(probs) / len(probs), 4),
            "median_prob": round(probs[len(probs) // 2], 4),
            "frac_prob_gt_01": round(sum(1 for p in probs if p > 0.1) / len(probs), 3),
            "frac_prob_gt_03": round(sum(1 for p in probs if p > 0.3) / len(probs), 3),
            "mean_prob_passing": round(
                sum(r["correct_token_prob"] for r in passing) / max(len(passing), 1), 4
            ),
            "mean_prob_failing": round(
                sum(r["correct_token_prob"] for r in failing) / max(len(failing), 1), 4
            ),
            "mean_entropy": round(sum(entropies) / len(entropies), 3),
            "median_entropy": round(sorted(entropies)[len(entropies) // 2], 3),
        }
    return out


def dpo_prob_destruction(results: list[dict]) -> dict:
    """Compare P(correct) between SFT and DPO stages."""
    lookup = defaultdict(dict)
    for r in results:
        lookup[r["prompt_id"]][r["model_stage"]] = r

    comparisons = [
        ("sft_3ep", "dpo_3ep", "SFT3→DPO3"),
        ("sft_5ep", "dpo_5ep", "SFT5→DPO5"),
        ("base", "sft_3ep", "Base→SFT3"),
    ]

    out = {}
    for stage_a, stage_b, label in comparisons:
        drops = []
        for pid, stages in lookup.items():
            ra = stages.get(stage_a)
            rb = stages.get(stage_b)
            if ra and rb:
                drops.append(ra["correct_token_prob"] - rb["correct_token_prob"])

        if drops:
            drops.sort()
            out[label] = {
                "mean_drop": round(sum(drops) / len(drops), 4),
                "median_drop": round(drops[len(drops) // 2], 4),
                "frac_dropped": round(sum(1 for d in drops if d > 0) / len(drops), 3),
                "frac_improved": round(sum(1 for d in drops if d < 0) / len(drops), 3),
            }
    return out


def replacement_token_semantics(results: list[dict]) -> dict:
    """For suppressed cases: what tokens replace the correct answer?"""
    GENERIC_TOKENS = {
        "i", "the", "a", "an", "this", "that", "it", "there",
        "you", "we", "to", "is", "are", "was", "be", "in", "of",
        "think", "believe", "would", "could", "may", "might",
        "sure", "certainly", "yes", "no", "however", "but",
    }

    by_stage = defaultdict(list)
    for r in results:
        if r["suppressed"]:
            by_stage[r["model_stage"]].append(r)

    out = {}
    for stage, rows in sorted(by_stage.items()):
        generic_count = 0
        top1_counter = defaultdict(int)
        for r in rows:
            top1 = r["top1_token"].lower().strip()
            top1_counter[r["top1_token"]] += 1
            if top1 in GENERIC_TOKENS:
                generic_count += 1

        n = len(rows)
        out[stage] = {
            "n_suppressed": n,
            "frac_generic": round(generic_count / max(n, 1), 3),
            "top_replacements": sorted(
                top1_counter.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }
    return out


def print_report(prob_stats: dict, destruction: dict, replacements: dict):
    print("\n" + "=" * 70)
    print("PROBABILITY MASS ANALYSIS")
    print("=" * 70)

    print(f"\n{'Stage':<14} {'Mean P(✓)':>10} {'Med P(✓)':>9} "
          f"{'P>0.1':>7} {'P>0.3':>7} {'Entropy':>9}")
    print("-" * 59)
    for stage, s in prob_stats.items():
        print(f"{stage:<14} {s['mean_prob']:>10.4f} {s['median_prob']:>9.4f} "
              f"{s['frac_prob_gt_01']:>7.1%} {s['frac_prob_gt_03']:>7.1%} "
              f"{s['mean_entropy']:>9.3f}")

    print(f"\nDPO probability destruction:")
    for label, s in destruction.items():
        print(f"  {label}: mean drop={s['mean_drop']:+.4f}, "
              f"dropped={s['frac_dropped']:.1%}, improved={s['frac_improved']:.1%}")

    print(f"\nReplacement tokens (suppressed cases):")
    for stage, s in replacements.items():
        if s["n_suppressed"] == 0:
            continue
        top3 = ", ".join(f"'{t}'({c})" for t, c in s["top_replacements"][:3])
        print(f"  {stage}: {s['n_suppressed']} cases, "
              f"generic={s['frac_generic']:.1%}, top: {top3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_results", default="outputs/sprint3/token_rank_full.csv")
    parser.add_argument("--output_dir", default="outputs/sprint3")
    args = parser.parse_args()

    print(f"Loading results from {args.rank_results}...")
    results = load_rank_results(args.rank_results)
    print(f"  Loaded {len(results)} results")

    prob_stats = prob_mass_by_stage(results)
    destruction = dpo_prob_destruction(results)
    replacements = replacement_token_semantics(results)

    print_report(prob_stats, destruction, replacements)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "probability_mass_analysis.json")
    with open(out_path, "w") as f:
        json.dump({
            "prob_stats": prob_stats,
            "destruction": destruction,
            "replacements": {
                stage: {k: v for k, v in s.items()}
                for stage, s in replacements.items()
            },
        }, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
