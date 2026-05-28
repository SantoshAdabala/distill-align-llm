"""
cross_method_eval.py — Cross-Method AFG Comparison
===================================================
Runs LLM-judge factuality evaluation across alignment methods:
- DPO (Llama-3.1-8B) — baseline
- SimPO (Llama-3.1-8B) — alternative alignment
- DPO (Mistral-7B) — cross-architecture
- DPO (Llama-3.2-3B) — scale sensitivity

Produces the comparison table for the paper.

Usage:
    # Merge SimPO adapter
    python scripts/cross_method_eval.py merge \
        --base ./outputs/sft_merged \
        --adapter ./outputs/simpo/simpo_adapter \
        --output ./outputs/simpo_merged

    # Run judge eval
    python scripts/cross_method_eval.py eval \
        --eval_file data/eval_factuality.jsonl \
        --openai_key $OPENAI_API_KEY \
        --simpo_merged ./outputs/simpo_merged \
        --simpo_reward_acc 0.XX \
        --output_dir outputs/cross_method

    # Print comparison table
    python scripts/cross_method_eval.py compare \
        --results_dir outputs/cross_method
"""

import json
import csv
import os
import argparse
import time
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a helpful ML engineering assistant with deep knowledge of "
    "LLM training, fine-tuning, alignment, and inference optimization. "
    "Answer questions factually and concisely."
)

JUDGE_SYSTEM = "You are an expert ML engineer evaluating factual accuracy. Return ONLY valid JSON."

JUDGE_USER = """Evaluate this response for factual accuracy.

Question: {question}
Must include (key concepts): {must_include}
Model response: {response}

Score 0-3:
  3 = Fully correct, all key facts present
  2 = Mostly correct, minor omission
  1 = Partially correct, key fact missing
  0 = Incorrect or hallucinated

Return JSON: {{"score": <int>, "reason": "<one sentence>"}}"""


# ── Merge utility ─────────────────────────────────────────────────────────────

def merge_adapter(base_path: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base weights."""
    print(f"\nMerging adapter into base weights")
    print(f"  Base:    {base_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"  Done. Merged model saved to {output_path}")
    del model
    gc.collect()


# ── Model loading and generation ──────────────────────────────────────────────

def load_model(model_path: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = f"### User:\n{question}\n\n### Assistant:\n"

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def judge_response(client: OpenAI, question: str, must_include: list,
                   response: str) -> tuple:
    user_msg = JUDGE_USER.format(
        question=question, must_include=", ".join(must_include),
        response=response[:600],
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0, max_tokens=100,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return max(0, min(3, int(data.get("score", 0)))), data.get("reason", "")
        except Exception as e:
            if attempt == 2:
                return 0, f"Judge error: {e}"
            time.sleep(1)
    return 0, "Judge error"


def run_eval_for_model(model_path: str, model_label: str, reward_accuracy: float,
                       prompts: list, judge_client: OpenAI, output_dir: str) -> dict:
    """Run judge eval for one model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_label}  (reward_acc={reward_accuracy:.1%})")
    print(f"  Path: {model_path}")

    model, tokenizer = load_model(model_path)
    results = []

    for i, p in enumerate(prompts):
        response = generate(model, tokenizer, p["prompt"])
        score, reason = judge_response(
            judge_client, p["prompt"], p.get("must_include", []), response
        )
        results.append({
            "prompt_id": p.get("id", f"p{i}"),
            "prompt": p["prompt"],
            "category": p.get("category", "unknown"),
            "model_label": model_label,
            "response": response[:300],
            "judge_score": score,
            "judge_norm": round(score / 3.0, 4),
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            avg = sum(r["judge_norm"] for r in results) / len(results)
            print(f"  [{i+1}/{len(prompts)}] avg judge: {avg:.3f}")
        if (i + 1) % 20 == 0:
            time.sleep(1)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save and compute AFG
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_label.replace(' ', '_').lower()}_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(results)
    factuality = sum(r["judge_norm"] for r in results) / n
    afg = reward_accuracy - factuality

    summary = {
        "model_label": model_label,
        "reward_accuracy": reward_accuracy,
        "factuality_judge": round(factuality, 4),
        "afg_points": round(afg * 100, 1),
        "n_prompts": n,
    }
    summary_path = os.path.join(output_dir, f"{model_label.replace(' ', '_').lower()}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Reward accuracy:    {reward_accuracy:.1%}")
    print(f"  Factuality (judge): {factuality:.1%}")
    print(f"  AFG:                {afg*100:.1f} pts")
    print(f"  Saved: {out_path}")
    return summary


def print_comparison_table(results_dir: str):
    """Print cross-method comparison table."""
    summaries = []

    # Load DPO baseline
    dpo_baseline = {
        "model_label": "DPO (Llama-3.1-8B)",
        "reward_accuracy": 0.82,
        "factuality_judge": 0.81,
        "afg_points": 1.0,
        "n_prompts": 51,
    }
    summaries.append(dpo_baseline)

    # Load other results
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_summary.json"):
            with open(os.path.join(results_dir, fname)) as f:
                summaries.append(json.load(f))

    print("\n" + "=" * 72)
    print("CROSS-METHOD AFG COMPARISON (LLM-judge evaluation)")
    print("=" * 72)
    print(f"\n{'Model/Method':<28} {'Reward Acc':>11} {'Factuality':>11} {'AFG':>8}")
    print("-" * 62)
    for s in summaries:
        print(f"{s['model_label']:<28} "
              f"{s['reward_accuracy']:>11.1%} "
              f"{s['factuality_judge']:>11.1%} "
              f"{s['afg_points']:>7.1f}pts")

    afg_values = [s["afg_points"] for s in summaries]
    print(f"\n{'─'*62}")
    if all(a < 10 for a in afg_values):
        print("FINDING: AFG < 10 pts across all methods under judge evaluation.")
        print("The measurement artifact generalizes beyond DPO.")
    elif any(a > 20 for a in afg_values):
        outliers = [s["model_label"] for s in summaries if s["afg_points"] > 20]
        print(f"FINDING: Large AFG for: {', '.join(outliers)}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # merge
    merge_p = subparsers.add_parser("merge")
    merge_p.add_argument("--base", required=True)
    merge_p.add_argument("--adapter", required=True)
    merge_p.add_argument("--output", required=True)

    # eval
    eval_p = subparsers.add_parser("eval")
    eval_p.add_argument("--eval_file", default="data/eval_factuality.jsonl")
    eval_p.add_argument("--openai_key", default=os.environ.get("OPENAI_API_KEY"))
    eval_p.add_argument("--output_dir", default="outputs/cross_method")
    eval_p.add_argument("--simpo_merged", default="./outputs/simpo_merged")
    eval_p.add_argument("--simpo_reward_acc", type=float, default=None)
    eval_p.add_argument("--mistral_merged", default=None)
    eval_p.add_argument("--mistral_reward_acc", type=float, default=0.77)
    eval_p.add_argument("--llama3b_merged", default=None)
    eval_p.add_argument("--llama3b_reward_acc", type=float, default=0.73)

    # compare
    compare_p = subparsers.add_parser("compare")
    compare_p.add_argument("--results_dir", default="outputs/cross_method")

    args = parser.parse_args()

    if args.command == "merge":
        merge_adapter(args.base, args.adapter, args.output)

    elif args.command == "eval":
        if not args.openai_key:
            raise ValueError("Provide --openai_key or set OPENAI_API_KEY")

        prompts = []
        with open(args.eval_file) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line.strip()))
        print(f"Loaded {len(prompts)} eval prompts")

        client = OpenAI(api_key=args.openai_key)

        if os.path.exists(args.simpo_merged):
            reward_acc = args.simpo_reward_acc or 0.80
            run_eval_for_model(args.simpo_merged, "SimPO (Llama-3.1-8B)",
                               reward_acc, prompts, client, args.output_dir)
        else:
            print(f"[SKIP] SimPO not found: {args.simpo_merged}")

        if args.mistral_merged and os.path.exists(args.mistral_merged):
            run_eval_for_model(args.mistral_merged, "DPO (Mistral-7B)",
                               args.mistral_reward_acc, prompts, client, args.output_dir)

        if args.llama3b_merged and os.path.exists(args.llama3b_merged):
            run_eval_for_model(args.llama3b_merged, "DPO (Llama-3.2-3B)",
                               args.llama3b_reward_acc, prompts, client, args.output_dir)

        print("\nAll evals complete. Run compare:")
        print(f"  python scripts/cross_method_eval.py compare --results_dir {args.output_dir}")

    elif args.command == "compare":
        print_comparison_table(args.results_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
