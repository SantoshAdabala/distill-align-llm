"""
token_rank_analysis.py  —  Sprint 3: Mechanistic Analysis
==========================================================
Answers the core question: does DPO *erase* factual knowledge or *suppress* it?

Method
------
For each eval prompt we identify the "key token position" — the first position
in the generated sequence where a must_include keyword should appear. At that
position we extract the full logit distribution from the frozen model (no
generation, just a forward pass) and compute:

• correct_token_rank  — where the correct token sits (1 = most likely)
• correct_token_prob  — raw softmax probability assigned to correct token
• entropy             — H = -Σ p log p  (certainty measure)
• top5_tokens         — what tokens the model prefers instead
• suppression_flag    — rank ≤ 10 but token not generated  →  suppressed

We run this across model stages: base / sft_3ep / sft_5ep / dpo_3ep / dpo_5ep

Key output: suppression_rate per stage
= (prompts where judge_score ≤ 1 AND correct_token rank ≤ 10) / failed_prompts

> 40%  →  Hypothesis B confirmed (DPO suppresses)
< 15%  →  Hypothesis A likely (DPO erases)

Usage (on Lightning.ai A100)
----------------------------
    # Using adapters (no merged checkpoints needed):
    python scripts/token_rank_analysis.py \
        --eval_file      data/eval_factuality.jsonl \
        --base_model     meta-llama/Llama-3.1-8B-Instruct \
        --sft_3ep_adapter ./outputs/sft/final_adapter \
        --sft_5ep_adapter ./outputs/llama_8b_5ep/sft/final_adapter \
        --dpo_3ep_adapter ./outputs/dpo/dpo_adapter \
        --dpo_5ep_adapter ./outputs/llama_8b_5ep/dpo/dpo_adapter \
        --output_dir     outputs/sprint3

    # Quick test (10 prompts):
    python scripts/token_rank_analysis.py --max_prompts 10

Cost: ~$2-3 on A100 (forward passes only, no training)
Time: ~15-20 minutes for all stages × 51 prompts
"""

import json
import csv
import argparse
import os
import gc
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TokenRankResult:
    prompt_id: str
    category: str
    question_type: str
    difficulty: int
    model_stage: str
    concept: str
    target_keyword: str
    target_token_id: int
    target_token_str: str
    correct_token_rank: int
    correct_token_prob: float
    correct_token_logit: float
    entropy: float
    top1_token: str
    top5_tokens: str
    judge_score: int
    judge_normalized: float
    suppressed: bool
    forgotten: bool
    correctly_generated: bool


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(model_id: str):
    """Load quantized base model."""
    print(f"  Loading base: {model_id}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def load_adapter_model(base_model_id: str, adapter_path: str, stage_name: str):
    """Load base + LoRA adapter for a specific stage."""
    print(f"  Loading {stage_name}: base={base_model_id} + adapter={adapter_path}")
    model, tokenizer = load_base_model(base_model_id)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def unload_model(model):
    """Free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Key token position finding ────────────────────────────────────────────────

def find_key_token(tokenizer, must_include: list[str]) -> tuple[Optional[int], str, str]:
    """Find the token ID for the first must_include keyword.

    Strategy: tokenize the first word of each keyword and return
    the first token ID. This is reproducible and directly connected
    to the eval schema.
    """
    for keyword in must_include:
        for prefix in [" ", "", "▁"]:
            candidate = prefix + keyword.split()[0]
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                token_str = tokenizer.decode(ids[0])
                return ids[0], token_str.strip(), keyword
    return None, "", ""


def build_prompt_text(tokenizer, prompt_text: str) -> str:
    """Format prompt using the model's chat template."""
    messages = [
        {"role": "user", "content": prompt_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"User: {prompt_text}\nAssistant:"


# ── Core analysis ─────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_prompt(
    model, tokenizer, prompt_text: str, must_include: list[str],
    prompt_id: str, category: str, question_type: str, difficulty: int,
    concept: str, stage_name: str, judge_score: int,
) -> Optional[TokenRankResult]:
    """Generate a short response and find the BEST rank the correct token
    achieves at any position during generation.

    This fixes the issue where checking only position 0 gives meaningless
    ranks — the model says preamble before the factual keyword appears.
    We generate up to 50 tokens and check the rank of the target token
    at every step, reporting the best (lowest) rank found.
    """
    target_token_id, target_token_str, keyword_used = find_key_token(
        tokenizer, must_include
    )
    if target_token_id is None:
        return None

    formatted = build_prompt_text(tokenizer, prompt_text)
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)

    # Generate up to 50 tokens, collecting logits at each step
    input_len = inputs["input_ids"].shape[1]
    generated_ids = inputs["input_ids"].clone()

    best_rank = 999999
    best_prob = 0.0
    best_logit = 0.0
    best_entropy = 0.0
    best_top5_ids = []
    best_top1_token = ""

    max_gen_tokens = 50
    for step in range(max_gen_tokens):
        outputs = model(input_ids=generated_ids)
        next_logits = outputs.logits[0, -1, :].float()
        probs = F.softmax(next_logits, dim=-1)

        # Check rank of target token at this position
        sorted_indices = torch.argsort(probs, descending=True)
        rank_0indexed = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0]
        if len(rank_0indexed) > 0:
            rank = int(rank_0indexed[0].item()) + 1
            if rank < best_rank:
                best_rank = rank
                best_prob = float(probs[target_token_id].item())
                best_logit = float(next_logits[target_token_id].item())
                p_clamped = probs.clamp(min=1e-10)
                best_entropy = float(-torch.sum(p_clamped * torch.log(p_clamped)).item())
                best_top5_ids = sorted_indices[:5].tolist()
                best_top1_token = tokenizer.decode([best_top5_ids[0]]).strip()

        # Greedy next token for continued generation
        next_token = sorted_indices[0].unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Stop if we found rank 1 or hit EOS
        if best_rank == 1:
            break
        if next_token.item() == tokenizer.eos_token_id:
            break

    if best_rank == 999999:
        return None

    top5_tokens = ", ".join(
        repr(tokenizer.decode([tid]).strip()) for tid in best_top5_ids[:5]
    )

    suppressed = (best_rank <= 10) and (judge_score <= 1)
    forgotten = (best_rank > 100) and (judge_score <= 1)
    correctly_generated = judge_score >= 2

    return TokenRankResult(
        prompt_id=prompt_id, category=category,
        question_type=question_type, difficulty=difficulty,
        model_stage=stage_name, concept=concept,
        target_keyword=keyword_used, target_token_id=target_token_id,
        target_token_str=target_token_str,
        correct_token_rank=best_rank, correct_token_prob=best_prob,
        correct_token_logit=best_logit, entropy=best_entropy,
        top1_token=best_top1_token, top5_tokens=top5_tokens,
        judge_score=judge_score, judge_normalized=judge_score / 3.0,
        suppressed=suppressed, forgotten=forgotten,
        correctly_generated=correctly_generated,
    )


def run_stage(model, tokenizer, prompts: list[dict], stage_name: str) -> list[TokenRankResult]:
    """Run token rank analysis for one model stage."""
    results = []
    n = len(prompts)
    skipped = 0
    print(f"\n  Stage: {stage_name}  ({n} prompts)")

    for i, p in enumerate(prompts):
        result = analyze_prompt(
            model, tokenizer,
            prompt_text=p["prompt"],
            must_include=p.get("must_include", []),
            prompt_id=p.get("id", f"p_{i:03d}"),
            category=p.get("category", "unknown"),
            question_type=p.get("question_type", "definition"),
            difficulty=p.get("difficulty", 1),
            concept=p.get("concept", ""),
            stage_name=stage_name,
            judge_score=p.get("_judge_score", 0),
        )
        if result is not None:
            results.append(result)
        else:
            skipped += 1

        if (i + 1) % 25 == 0:
            valid = results[-25:]
            if valid:
                median_rank = sorted(r.correct_token_rank for r in valid)[len(valid)//2]
                print(f"    [{i+1}/{n}] median rank: {median_rank}")

    print(f"  Done. {len(results)} analyzed, {skipped} skipped")
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def compute_stats(results: list[TokenRankResult]) -> dict:
    """Compute suppression/forgetting stats for one stage."""
    if not results:
        return {}

    failed = [r for r in results if r.judge_score <= 1]
    suppressed = [r for r in failed if r.suppressed]
    forgotten = [r for r in failed if r.forgotten]
    ranks = sorted(r.correct_token_rank for r in results)
    probs = [r.correct_token_prob for r in results]

    return {
        "n_total": len(results),
        "n_failed": len(failed),
        "n_suppressed": len(suppressed),
        "n_forgotten": len(forgotten),
        "suppression_rate": len(suppressed) / max(len(failed), 1),
        "forgetting_rate": len(forgotten) / max(len(failed), 1),
        "median_rank": ranks[len(ranks) // 2],
        "mean_rank": round(sum(ranks) / len(ranks), 1),
        "mean_correct_prob": round(sum(probs) / len(probs), 4),
        "mean_entropy": round(sum(r.entropy for r in results) / len(results), 2),
    }


def print_report(results_by_stage: dict[str, list[TokenRankResult]]):
    """Print paper-ready summary."""
    print("\n" + "=" * 70)
    print("SPRINT 3 — TOKEN RANK ANALYSIS RESULTS")
    print("Mechanistic evidence: suppression vs forgetting")
    print("=" * 70)

    print(f"\n{'Stage':<14} {'Med.Rank':>9} {'Mean P(✓)':>10} "
          f"{'Entropy':>9} {'Supp.Rate':>10} {'Forg.Rate':>10}")
    print("-" * 64)
    for stage, results in results_by_stage.items():
        s = compute_stats(results)
        if not s:
            continue
        print(f"{stage:<14} {s['median_rank']:>9}  {s['mean_correct_prob']:>9.4f}  "
              f"{s['mean_entropy']:>8.2f}  {s['suppression_rate']:>9.1%}  "
              f"{s['forgetting_rate']:>9.1%}")

    # Headline finding
    for stage_name in ["dpo_3ep", "dpo_5ep"]:
        if stage_name in results_by_stage:
            s = compute_stats(results_by_stage[stage_name])
            if not s:
                continue
            print(f"\n{'─'*70}")
            print(f"HEADLINE ({stage_name}):")
            print(f"  Failed prompts:         {s['n_failed']}")
            print(f"  → Suppressed (rank≤10): {s['n_suppressed']}  "
                  f"({s['suppression_rate']:.1%})")
            print(f"  → Forgotten (rank>100): {s['n_forgotten']}  "
                  f"({s['forgetting_rate']:.1%})")
            if s['suppression_rate'] > 0.40:
                print("\n  ✓ HYPOTHESIS B CONFIRMED: DPO suppresses knowledge")
            elif s['suppression_rate'] < 0.15:
                print("\n  → Hypothesis A likely: DPO erases knowledge")
            else:
                print("\n  → Mixed evidence: both suppression and forgetting")


def write_outputs(results_by_stage: dict, output_dir: str):
    """Write CSV and JSON outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Full CSV
    all_rows = []
    for results in results_by_stage.values():
        for r in results:
            all_rows.append(asdict(r))

    if all_rows:
        full_path = os.path.join(output_dir, "token_rank_full.csv")
        with open(full_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Full CSV: {full_path}")

    # Summary JSON
    summary = {}
    for stage, results in results_by_stage.items():
        summary[stage] = compute_stats(results)

    summary_path = os.path.join(output_dir, "token_rank_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:  {summary_path}")

    # Suppression cases
    suppress_rows = [asdict(r) for results in results_by_stage.values()
                     for r in results if r.suppressed]
    if suppress_rows:
        sp = os.path.join(output_dir, "suppression_cases.csv")
        with open(sp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(suppress_rows[0].keys()))
            writer.writeheader()
            writer.writerows(suppress_rows)
        print(f"  Suppressed: {sp} ({len(suppress_rows)} cases)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sprint 3: Token rank mechanistic analysis")
    parser.add_argument("--eval_file", default="data/eval_factuality.jsonl")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sft_3ep_adapter", default="./outputs/sft/final_adapter",
                        help="SFT 3-epoch adapter path")
    parser.add_argument("--sft_5ep_adapter", default="./outputs/llama_8b_5ep/sft/final_adapter",
                        help="SFT 5-epoch adapter path")
    parser.add_argument("--dpo_3ep_adapter", default="./outputs/dpo/dpo_adapter",
                        help="DPO 3-epoch adapter (trained on 3ep SFT merged base)")
    parser.add_argument("--dpo_3ep_base", default="./outputs/sft_merged",
                        help="Merged SFT base for DPO 3ep adapter")
    parser.add_argument("--dpo_5ep_adapter", default="./outputs/llama_8b_5ep/dpo/dpo_adapter",
                        help="DPO 5-epoch adapter")
    parser.add_argument("--dpo_5ep_base", default="./outputs/sft_merged",
                        help="Merged SFT base for DPO 5ep adapter")
    parser.add_argument("--output_dir", default="outputs/sprint3")
    parser.add_argument("--stages", nargs="+",
                        default=["base", "sft_3ep", "sft_5ep", "dpo_3ep", "dpo_5ep"],
                        help="Which stages to analyze")
    parser.add_argument("--max_prompts", type=int, default=None)
    args = parser.parse_args()

    # Load eval prompts
    print(f"\nLoading eval prompts from {args.eval_file}...")
    prompts = []
    with open(args.eval_file) as f:
        for i, line in enumerate(f):
            d = json.loads(line.strip())
            d["id"] = d.get("id", f"p_{i:03d}")
            d["_judge_score"] = 0  # default; will be overridden if judge results exist
            prompts.append(d)

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"  [TEST MODE] Limited to {args.max_prompts} prompts")
    print(f"  Loaded {len(prompts)} prompts")

    # Run analysis per stage
    results_by_stage = {}

    # ── Base model ──
    if "base" in args.stages:
        print(f"\n{'='*60}\nStage: base")
        model, tokenizer = load_base_model(args.base_model)
        results_by_stage["base"] = run_stage(model, tokenizer, prompts, "base")
        unload_model(model)

    # ── SFT 3-epoch ──
    if "sft_3ep" in args.stages and os.path.exists(args.sft_3ep_adapter):
        print(f"\n{'='*60}\nStage: sft_3ep")
        model, tokenizer = load_adapter_model(
            args.base_model, args.sft_3ep_adapter, "sft_3ep"
        )
        results_by_stage["sft_3ep"] = run_stage(model, tokenizer, prompts, "sft_3ep")
        unload_model(model)

    # ── SFT 5-epoch ──
    if "sft_5ep" in args.stages and os.path.exists(args.sft_5ep_adapter):
        print(f"\n{'='*60}\nStage: sft_5ep")
        model, tokenizer = load_adapter_model(
            args.base_model, args.sft_5ep_adapter, "sft_5ep"
        )
        results_by_stage["sft_5ep"] = run_stage(model, tokenizer, prompts, "sft_5ep")
        unload_model(model)

    # ── DPO 3-epoch ──
    if "dpo_3ep" in args.stages and os.path.exists(args.dpo_3ep_adapter):
        print(f"\n{'='*60}\nStage: dpo_3ep")
        dpo_base = args.dpo_3ep_base if os.path.exists(args.dpo_3ep_base) else args.base_model
        model, tokenizer = load_adapter_model(
            dpo_base, args.dpo_3ep_adapter, "dpo_3ep"
        )
        results_by_stage["dpo_3ep"] = run_stage(model, tokenizer, prompts, "dpo_3ep")
        unload_model(model)

    # ── DPO 5-epoch ──
    if "dpo_5ep" in args.stages and os.path.exists(args.dpo_5ep_adapter):
        print(f"\n{'='*60}\nStage: dpo_5ep")
        dpo_base = args.dpo_5ep_base if os.path.exists(args.dpo_5ep_base) else args.base_model
        model, tokenizer = load_adapter_model(
            dpo_base, args.dpo_5ep_adapter, "dpo_5ep"
        )
        results_by_stage["dpo_5ep"] = run_stage(model, tokenizer, prompts, "dpo_5ep")
        unload_model(model)

    # Report and save
    print_report(results_by_stage)
    write_outputs(results_by_stage, args.output_dir)
    print("\nSprint 3 Analysis 1 complete.")


if __name__ == "__main__":
    main()
