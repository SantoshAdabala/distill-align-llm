"""
attention_shift_analysis.py  —  Sprint 3 (Part 3, optional)
============================================================
Checks whether DPO shifts attention away from technical tokens
toward conversational framing tokens at factual answer positions.

Usage (GPU required):
    python scripts/attention_shift_analysis.py \
        --eval_file  data/eval_factuality.jsonl \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --sft_adapter ./outputs/sft/final_adapter \
        --dpo_adapter ./outputs/dpo/dpo_adapter \
        --dpo_base    ./outputs/sft_merged \
        --output_dir  outputs/sprint3 \
        --n_examples  30

Cost: ~$0.50 on A100 (very fast — 30 examples × 2 models)
"""

import json
import os
import csv
import argparse
import gc
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Technical tokens the model should attend to for factual answers
TECHNICAL_STEMS = [
    "lora", "dpo", "qlora", "rlhf", "sft", "grpo", "peft", "trl",
    "quant", "rank", "adapt", "gradient", "attention", "token",
    "layer", "weight", "model", "train", "fine", "tune", "embed",
    "param", "loss", "optim", "learn", "rate", "batch", "epoch",
    "norm", "head", "proj", "linear", "hidden", "kv", "cache",
    "fp16", "bf16", "nf4", "int8", "bits", "byte", "flash",
]

HEDGE_STEMS = [
    "i", "think", "believe", "would", "could", "should", "may",
    "might", "sure", "certain", "great", "question", "help",
    "assist", "provide", "explain", "note", "also", "however",
]


def classify_token(token_str: str) -> str:
    t = token_str.lower().strip().lstrip("▁").lstrip(" ")
    for stem in TECHNICAL_STEMS:
        if stem in t:
            return "technical"
    for stem in HEDGE_STEMS:
        if t == stem or t.startswith(stem):
            return "hedge"
    return "other"


def load_model(base_model: str, adapter_path: str = None, use_eager_attn: bool = True):
    """Load model with attention output enabled."""
    print(f"  Loading: {base_model}" + (f" + {adapter_path}" if adapter_path else ""))
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kwargs = {
        "quantization_config": bnb,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "output_attentions": True,
    }
    if use_eager_attn:
        kwargs["attn_implementation"] = "eager"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"User: {question}\nAssistant:"


@torch.no_grad()
def extract_attention_profile(model, tokenizer, question: str) -> dict:
    """Extract attention distribution from last position to all input tokens."""
    formatted = build_prompt(tokenizer, question)
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    input_ids = inputs["input_ids"][0]

    outputs = model(**inputs, output_attentions=True)

    # Average attention from last position across last 4 layers and all heads
    n_layers = len(outputs.attentions)
    last_4 = outputs.attentions[max(0, n_layers - 4):]
    attn_stack = torch.stack([layer[0] for layer in last_4])  # (L, H, S, S)
    attn_last_pos = attn_stack[:, :, -1, :]  # (L, H, S)
    attn_mean = attn_last_pos.mean(dim=(0, 1)).float().cpu()  # (S,)

    # Classify tokens
    tokens = [tokenizer.decode([tid]) for tid in input_ids.tolist()]
    token_types = [classify_token(t) for t in tokens]

    # Sum attention by type
    attn_by_type = defaultdict(float)
    for attn_val, ttype in zip(attn_mean.tolist(), token_types):
        attn_by_type[ttype] += attn_val

    total = sum(attn_by_type.values()) or 1.0
    return {
        "technical_attn": round(attn_by_type["technical"] / total, 4),
        "hedge_attn": round(attn_by_type["hedge"] / total, 4),
        "other_attn": round(attn_by_type["other"] / total, 4),
        "n_technical_tokens": token_types.count("technical"),
        "n_hedge_tokens": token_types.count("hedge"),
    }


def run_analysis(model, tokenizer, prompts: list[dict], stage_name: str) -> list[dict]:
    """Run attention analysis for one model stage."""
    results = []
    print(f"  Analyzing {stage_name} ({len(prompts)} prompts)")
    for i, p in enumerate(prompts):
        try:
            profile = extract_attention_profile(model, tokenizer, p["prompt"])
            results.append({
                "prompt_id": p.get("id", f"p_{i}"),
                "category": p.get("category", ""),
                "model_stage": stage_name,
                **profile,
            })
        except Exception as e:
            if i < 3:
                print(f"    [WARN] {p.get('id', i)}: {e}")
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(prompts)}]")
    return results


def print_report(sft_results: list[dict], dpo_results: list[dict]):
    def avg(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 70)
    print("ATTENTION SHIFT ANALYSIS: SFT vs DPO")
    print("=" * 70)

    print(f"\n{'Stage':<12} {'Tech.Attn':>10} {'Hedge.Attn':>11} {'Other.Attn':>11}")
    print("-" * 47)
    print(f"{'SFT':<12} {avg(sft_results,'technical_attn'):>10.4f} "
          f"{avg(sft_results,'hedge_attn'):>11.4f} "
          f"{avg(sft_results,'other_attn'):>11.4f}")
    print(f"{'DPO':<12} {avg(dpo_results,'technical_attn'):>10.4f} "
          f"{avg(dpo_results,'hedge_attn'):>11.4f} "
          f"{avg(dpo_results,'other_attn'):>11.4f}")

    shift = avg(dpo_results, 'technical_attn') - avg(sft_results, 'technical_attn')
    print(f"\nTechnical attention shift (DPO - SFT): {shift:+.4f}")
    if shift < -0.03:
        print("→ DPO reduces attention to technical tokens — supports suppression")
    elif shift > 0.03:
        print("→ DPO increases attention to technical tokens (unexpected)")
    else:
        print("→ No significant shift (< 3pp)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default="data/eval_factuality.jsonl")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sft_adapter", default="./outputs/sft/final_adapter")
    parser.add_argument("--dpo_adapter", default="./outputs/dpo/dpo_adapter")
    parser.add_argument("--dpo_base", default="./outputs/sft_merged")
    parser.add_argument("--output_dir", default="outputs/sprint3")
    parser.add_argument("--n_examples", type=int, default=30)
    args = parser.parse_args()

    # Load prompts
    print(f"\nLoading prompts from {args.eval_file}...")
    all_prompts = []
    with open(args.eval_file) as f:
        for i, line in enumerate(f):
            d = json.loads(line.strip())
            d["id"] = d.get("id", f"p_{i:03d}")
            all_prompts.append(d)
    prompts = all_prompts[:args.n_examples]
    print(f"  Using {len(prompts)} prompts")

    # SFT model
    print(f"\n[1/2] SFT model")
    model, tokenizer = load_model(args.base_model, args.sft_adapter)
    sft_results = run_analysis(model, tokenizer, prompts, "sft")
    del model; gc.collect(); torch.cuda.empty_cache()

    # DPO model
    print(f"\n[2/2] DPO model")
    dpo_base = args.dpo_base if os.path.exists(args.dpo_base) else args.base_model
    model, tokenizer = load_model(dpo_base, args.dpo_adapter)
    dpo_results = run_analysis(model, tokenizer, prompts, "dpo")
    del model; gc.collect(); torch.cuda.empty_cache()

    # Report
    print_report(sft_results, dpo_results)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = sft_results + dpo_results
    out_csv = os.path.join(args.output_dir, "attention_shift_results.csv")
    if all_results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Saved: {out_csv}")

    print("\nSprint 3 Analysis 3 complete.")


if __name__ == "__main__":
    main()
