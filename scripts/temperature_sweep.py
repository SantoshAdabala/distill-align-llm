"""
temperature_sweep.py — Sprint 3 validation experiment
======================================================
Tests whether lowering temperature recovers factuality from DPO models.

If the AFG is caused by entropy/verbosity (not forgetting), then lower
temperature should concentrate probability mass on the correct tokens
and improve factuality scores without any retraining.

This is the "zero-cost inference-time fix" for the paper.

Usage:
    python scripts/temperature_sweep.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --dpo_adapter ./outputs/llama_8b_5ep/dpo/dpo_adapter \
        --dpo_base ./outputs/sft_merged \
        --eval_file data/eval_factuality.jsonl \
        --output_dir outputs/sprint3

    # Also test SFT model for comparison:
    python scripts/temperature_sweep.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --sft_adapter ./outputs/llama_8b_5ep/sft/final_adapter \
        --eval_file data/eval_factuality.jsonl \
        --output_dir outputs/sprint3 \
        --label sft_5ep

Cost: ~$1 on A100 (forward passes only, ~1 hour)
"""

import json
import argparse
import os
import gc
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0]


def load_model(base_model: str, adapter_path: str = None, dpo_base: str = None):
    """Load model with optional adapter."""
    model_path = dpo_base if dpo_base and os.path.exists(dpo_base) else base_model
    print(f"  Loading: {model_path}" + (f" + {adapter_path}" if adapter_path else ""))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model if not dpo_base else model_path
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, temperature: float,
                      max_new_tokens: int = 200) -> str:
    """Generate a response at a specific temperature."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)

    if temperature <= 0.01:
        # Greedy decoding
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    else:
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_response(response: str, must_include: list, must_not_include: list) -> dict:
    """Keyword-based factuality scoring (same as eval_factuality_all.py)."""
    resp_lower = response.lower()
    included = [t for t in must_include if t.lower() in resp_lower]
    missing = [t for t in must_include if t.lower() not in resp_lower]
    hallucinated = [t for t in must_not_include if t.lower() in resp_lower]
    return {
        "passed": len(missing) == 0 and len(hallucinated) == 0,
        "included": included,
        "missing": missing,
        "hallucinated": hallucinated,
    }


def run_sweep(model, tokenizer, prompts: list[dict], temperatures: list[float],
              label: str) -> dict:
    """Run factuality eval at each temperature."""
    results = {}
    n = len(prompts)

    for temp in temperatures:
        passed = 0
        details = []
        print(f"\n  T={temp:.1f}: evaluating {n} prompts...")

        for i, p in enumerate(prompts):
            response = generate_response(model, tokenizer, p["prompt"], temp)
            score = evaluate_response(
                response, p["must_include"], p.get("must_not_include", [])
            )
            if score["passed"]:
                passed += 1
            details.append({
                "prompt": p["prompt"],
                "response": response[:300],
                **score,
            })

            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{n}] passed so far: {passed}")

        accuracy = passed / n
        results[f"T={temp}"] = {
            "temperature": temp,
            "passed": passed,
            "total": n,
            "accuracy": round(accuracy, 4),
            "details": details,
        }
        print(f"  T={temp:.1f}: {passed}/{n} ({accuracy:.1%})")

    return results


def print_report(results: dict, label: str):
    print("\n" + "=" * 60)
    print(f"TEMPERATURE SWEEP RESULTS — {label}")
    print("=" * 60)
    print(f"\n{'Temperature':>12} {'Passed':>8} {'Accuracy':>10}")
    print("-" * 33)

    best_temp = None
    best_acc = 0
    for key, r in sorted(results.items(), key=lambda x: x[1]["temperature"]):
        acc = r["accuracy"]
        marker = " ← best" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_temp = r["temperature"]
        print(f"{r['temperature']:>12.1f} {r['passed']:>8}/{r['total']}  "
              f"{acc:>9.1%}{marker}")

    print(f"\nBest temperature: {best_temp}")
    print(f"Best accuracy:    {best_acc:.1%}")

    # Compare to T=1.0 (standard)
    t1_acc = results.get("T=1.0", {}).get("accuracy", 0)
    if best_acc > t1_acc:
        improvement = best_acc - t1_acc
        print(f"\nImprovement over T=1.0: +{improvement:.1%} ({improvement*100:.1f}pp)")
        print("→ Confirms entropy hypothesis: lower temperature recovers factuality")
        print("→ Paper claim: 'Temperature reduction is a zero-cost inference-time")
        print("  fix for the Alignment-Factuality Gap'")
    else:
        print("\nNo improvement from temperature reduction.")
        print("→ AFG may not be purely an entropy/verbosity issue")


def main():
    parser = argparse.ArgumentParser(description="Temperature sweep for AFG diagnosis")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dpo_adapter", default="./outputs/llama_8b_5ep/dpo/dpo_adapter")
    parser.add_argument("--dpo_base", default="./outputs/sft_merged")
    parser.add_argument("--sft_adapter", default=None,
                        help="If set, test SFT model instead of DPO")
    parser.add_argument("--eval_file", default="data/eval_factuality.jsonl")
    parser.add_argument("--output_dir", default="outputs/sprint3")
    parser.add_argument("--label", default="dpo_5ep",
                        help="Label for this run (used in output filenames)")
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=TEMPERATURES)
    parser.add_argument("--max_prompts", type=int, default=None)
    args = parser.parse_args()

    # Load prompts
    print(f"\nLoading prompts from {args.eval_file}...")
    prompts = []
    with open(args.eval_file) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"  Loaded {len(prompts)} prompts")
    print(f"  Temperatures: {args.temperatures}")

    # Load model
    print(f"\nLoading model...")
    if args.sft_adapter:
        model, tokenizer = load_model(args.base_model, adapter_path=args.sft_adapter)
    else:
        model, tokenizer = load_model(
            args.base_model, adapter_path=args.dpo_adapter, dpo_base=args.dpo_base
        )

    # Run sweep
    results = run_sweep(model, tokenizer, prompts, args.temperatures, args.label)

    # Report
    print_report(results, args.label)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"temperature_sweep_{args.label}.json")
    # Strip details for the summary file
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "details"}
               for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Also save full details
    full_path = os.path.join(args.output_dir, f"temperature_sweep_{args.label}_full.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full:  {full_path}")


if __name__ == "__main__":
    main()
