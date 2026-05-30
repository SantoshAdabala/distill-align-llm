"""
TruthfulQA MC1 Evaluation Script
=================================

Evaluates a language model on the TruthfulQA multiple-choice (MC1) benchmark.
For each question, computes the normalized log-probability of each answer choice
conditioned on the prompt "Q: {question}\nA:" and selects the highest-scoring one.

Usage:
    python scripts/eval_truthfulqa.py \
        --model_path path/to/model \
        --label "my-model" \
        --max_samples 200 \
        --output_dir outputs/truthfulqa

Arguments:
    --model_path    Path to the HuggingFace model (local or hub ID)
    --label         Label for this evaluation run (used in output filenames)
    --max_samples   Maximum number of samples to evaluate (default: all)
    --output_dir    Directory to save results (default: outputs/truthfulqa)

Outputs:
    - {output_dir}/{label}_summary.json: Overall accuracy and metadata
    - {output_dir}/{label}_details.json: Per-question predictions and scores
"""

import argparse
import json
import os
import time

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on TruthfulQA MC1 (multiple choice, single correct answer)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the HuggingFace model (local directory or hub ID)"
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="Label for this evaluation run (used in output filenames)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/truthfulqa",
        help="Directory to save results (default: outputs/truthfulqa)"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    """Load model with 4-bit NF4 quantization and its tokenizer."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def compute_completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """
    Compute the normalized log-probability of a completion given a prompt.

    Returns the average log-probability per token of the completion,
    conditioned on the prompt tokens.
    """
    # Tokenize prompt and full sequence
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    full_text = prompt + " " + completion
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)

    prompt_len = prompt_ids.shape[1]
    completion_len = full_ids.shape[1] - prompt_len

    if completion_len <= 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log-probabilities for completion tokens
    # logits[0, i] predicts token at position i+1
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

    total_logprob = 0.0
    for i in range(prompt_len - 1, full_ids.shape[1] - 1):
        token_id = full_ids[0, i + 1].item()
        total_logprob += log_probs[i, token_id].item()

    # Only count completion tokens (normalize by completion length)
    completion_logprob = 0.0
    for i in range(prompt_len - 1, full_ids.shape[1] - 1):
        if i >= prompt_len - 1:
            token_id = full_ids[0, i + 1].item()
            completion_logprob += log_probs[i, token_id].item()

    # Normalize by number of completion tokens
    normalized_logprob = completion_logprob / completion_len

    return normalized_logprob


def evaluate_mc1(model, tokenizer, dataset, max_samples=None):
    """
    Evaluate model on TruthfulQA MC1 task.

    For each question, score all choices by normalized log-probability
    and pick the highest-scoring one. MC1 has exactly one correct answer
    at index 0 in mc1_targets.
    """
    results = []
    correct = 0
    total = 0

    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    for idx, example in enumerate(samples):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]

        # The correct answer has label=1
        correct_idx = labels.index(1)

        prompt = f"Q: {question}\nA:"

        # Score each choice
        scores = []
        for choice in choices:
            score = compute_completion_logprob(model, tokenizer, prompt, choice)
            scores.append(score)

        # Pick highest-scoring choice
        predicted_idx = int(np.argmax(scores))
        is_correct = predicted_idx == correct_idx

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "index": idx,
            "question": question,
            "choices": choices,
            "correct_idx": correct_idx,
            "predicted_idx": predicted_idx,
            "scores": scores,
            "correct": is_correct,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(samples)}] Running accuracy: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def main():
    args = parse_args()

    print(f"=== TruthfulQA MC1 Evaluation ===")
    print(f"Model: {args.model_path}")
    print(f"Label: {args.label}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print()

    # Load dataset
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    print(f"  Dataset size: {len(dataset)} questions")

    # Load model
    print("Loading model with 4-bit NF4 quantization...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    print("  Model loaded successfully")
    print()

    # Evaluate
    print("Running MC1 evaluation...")
    start_time = time.time()
    accuracy, details = evaluate_mc1(model, tokenizer, dataset, args.max_samples)
    elapsed = time.time() - start_time

    print()
    print(f"=== Results ===")
    print(f"  Accuracy: {accuracy:.4f} ({sum(d['correct'] for d in details)}/{len(details)})")
    print(f"  Time: {elapsed:.1f}s")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "model_path": args.model_path,
        "label": args.label,
        "task": "truthfulqa_mc1",
        "accuracy": accuracy,
        "num_samples": len(details),
        "num_correct": sum(d["correct"] for d in details),
        "elapsed_seconds": round(elapsed, 1),
    }

    summary_path = os.path.join(args.output_dir, f"{args.label}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

    details_path = os.path.join(args.output_dir, f"{args.label}_details.json")
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)
    print(f"  Details saved to: {details_path}")


if __name__ == "__main__":
    main()
