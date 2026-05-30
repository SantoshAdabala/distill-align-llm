"""
Human Annotation Tool
=====================

Interactive terminal tool for human evaluation of model responses.
Loads a results CSV, samples responses stratified by judge score,
and presents each (question, response) pair for human rating.

Usage:
    # Start fresh annotation session
    python scripts/human_annotation.py \
        --results_csv outputs/eval/model_results.csv \
        --n_samples 100 \
        --output_dir outputs/human_annotation

    # Resume a previous session
    python scripts/human_annotation.py \
        --results_csv outputs/eval/model_results.csv \
        --output_dir outputs/human_annotation \
        --resume

    # Compute agreement metrics only (no annotation)
    python scripts/human_annotation.py \
        --results_csv outputs/eval/model_results.csv \
        --output_dir outputs/human_annotation \
        --compute_only

Arguments:
    --results_csv   Path to results CSV with columns: prompt, response, judge_score or judge_norm
    --n_samples     Number of samples to annotate (default: 100)
    --output_dir    Directory to save annotations and agreement metrics
                    (default: outputs/human_annotation)
    --resume        Resume annotation from where you left off
    --compute_only  Only compute agreement metrics from existing annotations

Rating Scale:
    0 = Poor (incorrect, harmful, or irrelevant)
    1 = Below average (partially correct but significant issues)
    2 = Good (mostly correct, minor issues)
    3 = Excellent (fully correct, helpful, well-structured)

Outputs:
    - {output_dir}/annotations.json: All human annotations with metadata
    - {output_dir}/agreement.json: Inter-annotator agreement metrics
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Human annotation tool for evaluating model responses"
    )
    parser.add_argument(
        "--results_csv", type=str, required=True,
        help="Path to results CSV (must have columns: prompt, response, and judge_score or judge_norm)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of samples to annotate (default: 100)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/human_annotation",
        help="Directory to save annotations (default: outputs/human_annotation)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume annotation from where you left off"
    )
    parser.add_argument(
        "--compute_only", action="store_true",
        help="Only compute agreement metrics from existing annotations (no annotation)"
    )
    return parser.parse_args()


def get_judge_score_column(df: pd.DataFrame) -> str:
    """Determine which column contains the judge score."""
    if "judge_score" in df.columns:
        return "judge_score"
    elif "judge_norm" in df.columns:
        return "judge_norm"
    else:
        raise ValueError(
            "Results CSV must contain either 'judge_score' or 'judge_norm' column. "
            f"Found columns: {list(df.columns)}"
        )


def stratified_sample(df: pd.DataFrame, score_col: str, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample n_samples rows stratified by judge score.

    Bins scores into quartiles and samples proportionally from each bin.
    """
    rng = np.random.default_rng(seed)

    # Create score bins (quartiles)
    df = df.copy()
    df["_score_bin"] = pd.qcut(df[score_col], q=4, labels=False, duplicates="drop")

    # Calculate samples per bin (proportional)
    bin_counts = df["_score_bin"].value_counts()
    bin_proportions = bin_counts / len(df)
    samples_per_bin = (bin_proportions * n_samples).round().astype(int)

    # Adjust to hit exact n_samples
    diff = n_samples - samples_per_bin.sum()
    if diff > 0:
        # Add to largest bin
        largest_bin = samples_per_bin.idxmax()
        samples_per_bin[largest_bin] += diff
    elif diff < 0:
        # Remove from largest bin
        largest_bin = samples_per_bin.idxmax()
        samples_per_bin[largest_bin] += diff

    # Sample from each bin
    sampled_dfs = []
    for bin_label, n in samples_per_bin.items():
        bin_df = df[df["_score_bin"] == bin_label]
        n = min(n, len(bin_df))
        if n > 0:
            indices = rng.choice(len(bin_df), size=n, replace=False)
            sampled_dfs.append(bin_df.iloc[indices])

    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.drop(columns=["_score_bin"])

    # Shuffle the final sample
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    return result


def annotate_samples(samples: pd.DataFrame, score_col: str, output_dir: str, resume: bool = False):
    """
    Present samples to the annotator in the terminal and collect ratings.

    Supports resuming from a previous session.
    """
    annotations_path = os.path.join(output_dir, "annotations.json")

    # Load existing annotations if resuming
    annotations = []
    start_idx = 0
    if resume and os.path.exists(annotations_path):
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        start_idx = len(annotations)
        print(f"Resuming from sample {start_idx + 1}/{len(samples)}")
        print()

    if start_idx >= len(samples):
        print("All samples already annotated!")
        return annotations

    print("=" * 60)
    print("HUMAN ANNOTATION SESSION")
    print("=" * 60)
    print()
    print("Rate each response on a scale of 0-3:")
    print("  0 = Poor (incorrect, harmful, or irrelevant)")
    print("  1 = Below average (partially correct but significant issues)")
    print("  2 = Good (mostly correct, minor issues)")
    print("  3 = Excellent (fully correct, helpful, well-structured)")
    print()
    print("Type 'q' to quit and save progress.")
    print("=" * 60)
    print()

    for idx in range(start_idx, len(samples)):
        row = samples.iloc[idx]
        prompt = row["prompt"]
        response = row["response"]
        judge_score = row[score_col]

        print(f"--- Sample {idx + 1}/{len(samples)} ---")
        print()
        print(f"[PROMPT]")
        print(prompt)
        print()
        print(f"[RESPONSE]")
        print(response)
        print()

        # Get human rating
        while True:
            user_input = input("Your rating (0/1/2/3, or 'q' to quit): ").strip()

            if user_input.lower() == "q":
                # Save progress and exit
                _save_annotations(annotations, annotations_path)
                print(f"\nProgress saved ({len(annotations)}/{len(samples)} annotated).")
                return annotations

            if user_input in ("0", "1", "2", "3"):
                human_score = int(user_input)
                break
            else:
                print("  Invalid input. Please enter 0, 1, 2, 3, or 'q'.")

        annotations.append({
            "index": idx,
            "prompt": prompt,
            "response": response,
            "judge_score": float(judge_score),
            "human_score": human_score,
        })

        # Save after each annotation (in case of crash)
        _save_annotations(annotations, annotations_path)
        print()

    print(f"\nAnnotation complete! ({len(annotations)}/{len(samples)} samples)")
    return annotations


def _save_annotations(annotations: list, path: str):
    """Save annotations to JSON file."""
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def compute_agreement(annotations: list) -> dict:
    """
    Compute agreement metrics between human scores and judge scores.

    Metrics:
        - Cohen's kappa (weighted, quadratic)
        - Exact agreement (proportion of exact matches)
        - Within-1 agreement (proportion within 1 point)
        - Pearson correlation
    """
    if len(annotations) == 0:
        return {"error": "No annotations to compute agreement from"}

    human_scores = np.array([a["human_score"] for a in annotations])
    judge_scores = np.array([a["judge_score"] for a in annotations])

    # Round judge scores to nearest integer for kappa/agreement
    judge_rounded = np.round(judge_scores).astype(int)
    judge_rounded = np.clip(judge_rounded, 0, 3)

    # Exact agreement
    exact_agreement = float(np.mean(human_scores == judge_rounded))

    # Within-1 agreement
    within_1 = float(np.mean(np.abs(human_scores - judge_rounded) <= 1))

    # Pearson correlation
    if np.std(human_scores) > 0 and np.std(judge_scores) > 0:
        pearson_r = float(np.corrcoef(human_scores, judge_scores)[0, 1])
    else:
        pearson_r = 0.0

    # Cohen's kappa (quadratic weighted)
    kappa = _cohens_kappa_quadratic(human_scores, judge_rounded, n_categories=4)

    return {
        "n_annotations": len(annotations),
        "cohens_kappa_quadratic": round(kappa, 4),
        "exact_agreement": round(exact_agreement, 4),
        "within_1_agreement": round(within_1, 4),
        "pearson_correlation": round(pearson_r, 4),
        "human_score_mean": round(float(np.mean(human_scores)), 4),
        "human_score_std": round(float(np.std(human_scores)), 4),
        "judge_score_mean": round(float(np.mean(judge_scores)), 4),
        "judge_score_std": round(float(np.std(judge_scores)), 4),
    }


def _cohens_kappa_quadratic(y1: np.ndarray, y2: np.ndarray, n_categories: int = 4) -> float:
    """
    Compute quadratic-weighted Cohen's kappa.

    This measures agreement between two raters, accounting for chance
    agreement and weighting disagreements by their squared distance.
    """
    # Build confusion matrix
    confusion = np.zeros((n_categories, n_categories), dtype=float)
    for a, b in zip(y1, y2):
        a_int = int(np.clip(a, 0, n_categories - 1))
        b_int = int(np.clip(b, 0, n_categories - 1))
        confusion[a_int, b_int] += 1

    n = confusion.sum()
    if n == 0:
        return 0.0

    # Normalize
    confusion = confusion / n

    # Marginals
    row_marginals = confusion.sum(axis=1)
    col_marginals = confusion.sum(axis=0)

    # Expected matrix (outer product of marginals)
    expected = np.outer(row_marginals, col_marginals)

    # Quadratic weight matrix
    weights = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            weights[i, j] = ((i - j) ** 2) / ((n_categories - 1) ** 2)

    # Weighted observed and expected agreement
    observed_disagreement = np.sum(weights * confusion)
    expected_disagreement = np.sum(weights * expected)

    if expected_disagreement == 0:
        return 1.0

    kappa = 1.0 - (observed_disagreement / expected_disagreement)
    return kappa


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotations_path = os.path.join(args.output_dir, "annotations.json")
    agreement_path = os.path.join(args.output_dir, "agreement.json")

    # Compute-only mode
    if args.compute_only:
        if not os.path.exists(annotations_path):
            print(f"Error: No annotations found at {annotations_path}")
            sys.exit(1)

        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        print(f"Computing agreement metrics from {len(annotations)} annotations...")
        agreement = compute_agreement(annotations)

        with open(agreement_path, "w") as f:
            json.dump(agreement, f, indent=2)

        print()
        print("=== Agreement Metrics ===")
        for key, value in agreement.items():
            print(f"  {key}: {value}")
        print(f"\nSaved to: {agreement_path}")
        return

    # Load results CSV
    print(f"Loading results from: {args.results_csv}")
    df = pd.read_csv(args.results_csv)
    score_col = get_judge_score_column(df)
    print(f"  Rows: {len(df)}, Score column: {score_col}")

    # Validate required columns
    required_cols = {"prompt", "response", score_col}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)

    # Sample
    n_samples = min(args.n_samples, len(df))
    print(f"  Sampling {n_samples} responses (stratified by {score_col})...")
    samples = stratified_sample(df, score_col, n_samples)

    # If resuming, load existing sample order (to maintain consistency)
    sample_path = os.path.join(args.output_dir, "sampled_indices.json")
    if args.resume and os.path.exists(sample_path):
        # Reload the same sample used previously
        with open(sample_path, "r") as f:
            saved_indices = json.load(f)
        samples = df.iloc[saved_indices].reset_index(drop=True)
        print(f"  Loaded previous sample of {len(samples)} items")
    else:
        # Save sample indices for reproducibility
        with open(sample_path, "w") as f:
            json.dump(samples.index.tolist(), f)

    # Run annotation
    annotations = annotate_samples(samples, score_col, args.output_dir, resume=args.resume)

    # Compute agreement if we have annotations
    if annotations:
        print()
        print("Computing agreement metrics...")
        agreement = compute_agreement(annotations)

        with open(agreement_path, "w") as f:
            json.dump(agreement, f, indent=2)

        print()
        print("=== Agreement Metrics ===")
        for key, value in agreement.items():
            print(f"  {key}: {value}")
        print(f"\nSaved to: {agreement_path}")


if __name__ == "__main__":
    main()
