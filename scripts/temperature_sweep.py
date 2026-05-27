"""
temperature_sweep.py  —  Sprint 3 (Part 4)
==========================================
Tests whether inference-time temperature reduction closes the AFG.

Hypothesis: DPO flattens the logit distribution (entropy 1.76→2.17).
If true, lowering temperature sharpens the distribution and surfaces
the correct high-P tokens that DPO suppresses at T=1.0.

Cost estimates (A100 SXM):
  51 prompts  × 3 temps = ~18 min  ~$0.45   quick sanity check
  500 prompts × 3 temps = ~3 hrs   ~$4.50   paper-quality (recommended)
  500 prompts × 5 temps = ~5 hrs   ~$7.50   full sweep

Usage
-----
    # Recommended: 500 prompts × 3 temperatures
    python scripts/temperature_sweep.py \
        --dpo_base    ./outputs/sft_merged \
        --dpo_adapter ./outputs/llama_8b_5ep/dpo/dpo_adapter \
        --eval_file   data/eval_factuality_v2.jsonl \
        --temperatures 0.1 0.5 1.0 \
        --label        dpo_5ep

    # Full sweep: 5 temperatures
    python scripts/temperature_sweep.py \
        --eval_file   data/eval_factuality_v2.jsonl \
        --temperatures 0.1 0.3 0.5 0.7 1.0 \
        --label        dpo_5ep

    # Quick sanity check (original 51-prompt file)
    python scripts/temperature_sweep.py \
        --eval_file    data/eval_factuality.jsonl \
        --temperatures 0.1 0.5 1.0 \
        --label        dpo_5ep_quick
"""

import json
import csv
import os
import time
import argparse
import gc
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a helpful ML engineering assistant with deep knowledge of "
    "LLM training, fine-tuning, alignment, and inference optimization. "
    "Answer questions factually and concisely."
)
MAX_NEW_TOKENS = 200
GREEDY_THRESHOLD = 0.05


@dataclass
class EvalPrompt:
    id: str
    prompt: str
    reference_answer: str
    must_include: list
    must_not_include: list
    category: str
    question_type: str
    difficulty: int
    concept: str = ""


@dataclass
class SweepResult:
    prompt_id: str
    category: str
    question_type: str
    difficulty: int
    temperature: float
    model_label: str
    response: str
    keyword_pass: bool
    keyword_score: float
    n_must_include: int
    n_keywords_found: int
    mean_entropy: float
    mean_top1_prob: float
    mean_correct_prob: float


def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(base_path: str, adapter_path: Optional[str] = None):
    print(f"  Loading base: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        print(f"  Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"### User:\n{question}\n\n### Assistant:\n"


def score_keywords(response, must_include, must_not_include):
    resp_lower = response.lower()
    found = sum(1 for kw in must_include if kw.lower() in resp_lower)
    bad = sum(1 for kw in must_not_include if kw.lower() in resp_lower)
    score = found / max(len(must_include), 1)
    passed = (found >= max(1, len(must_include) // 2)) and (bad == 0)
    return passed, score, found


def get_target_token_ids(tokenizer, must_include: list) -> list:
    ids = set()
    for keyword in must_include:
        first_word = keyword.split()[0].rstrip(",-.")
        for prefix in [" ", "", "▁"]:
            cids = tokenizer.encode(prefix + first_word, add_special_tokens=False)
            real = [i for i in cids
                    if i not in (tokenizer.bos_token_id, tokenizer.eos_token_id)
                    and i < tokenizer.vocab_size]
            if real:
                ids.add(real[0])
                break
    return list(ids)


@torch.no_grad()
def generate_with_metrics(model, tokenizer, prompt, temperature, target_ids):
    formatted = build_prompt(tokenizer, prompt.prompt)
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    is_greedy = temperature < GREEDY_THRESHOLD
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if is_greedy:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = 1.0
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    outputs = model.generate(**inputs, **gen_kwargs)
    gen_ids = outputs.sequences[0][prompt_len:].tolist()
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    entropies, top1_probs, correct_probs = [], [], []
    for step_scores in outputs.scores:
        logits = step_scores[0].float()
        probs = F.softmax(logits, dim=-1)
        p_clamp = probs.clamp(min=1e-12)
        entropy = float(-torch.sum(p_clamp * torch.log(p_clamp)).item())
        entropies.append(entropy)
        top1_probs.append(float(probs.max().item()))
        if target_ids:
            t = torch.tensor(
                [i for i in target_ids if i < probs.shape[0]],
                device=probs.device, dtype=torch.long,
            )
            correct_probs.append(
                float(probs[t].sum().item()) if len(t) > 0 else 0.0
            )
        else:
            correct_probs.append(0.0)

    n = max(len(entropies), 1)
    return (
        response,
        sum(entropies) / n,
        sum(top1_probs) / n,
        sum(correct_probs) / n,
    )


def run_sweep(model, tokenizer, prompts, temperatures, model_label):
    results = []
    total = len(prompts) * len(temperatures)
    done, t0 = 0, time.time()
    print(f"\n  {len(prompts)} prompts × {len(temperatures)} temps = {total} generations")

    for temp in sorted(temperatures):
        kw_passes = 0
        print(f"\n  [T={temp:.2f}] Starting...")

        for i, prompt in enumerate(prompts):
            target_ids = get_target_token_ids(tokenizer, prompt.must_include)
            response, mean_ent, mean_top1, mean_corr = generate_with_metrics(
                model, tokenizer, prompt, temp, target_ids
            )
            kw_pass, kw_score, n_found = score_keywords(
                response, prompt.must_include, prompt.must_not_include
            )
            if kw_pass:
                kw_passes += 1

            results.append(SweepResult(
                prompt_id=prompt.id,
                category=prompt.category,
                question_type=prompt.question_type,
                difficulty=prompt.difficulty,
                temperature=temp,
                model_label=model_label,
                response=response[:300],
                keyword_pass=kw_pass,
                keyword_score=round(kw_score, 4),
                n_must_include=len(prompt.must_include),
                n_keywords_found=n_found,
                mean_entropy=round(mean_ent, 4),
                mean_top1_prob=round(mean_top1, 4),
                mean_correct_prob=round(mean_corr, 6),
            ))
            done += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(prompts):
                elapsed = time.time() - t0
                per_gen = elapsed / max(done, 1)
                remaining = (total - done) * per_gen
                print(f"    [{i+1}/{len(prompts)}] pass={kw_passes/(i+1):.1%}  "
                      f"ETA {remaining/60:.1f} min")

    return results


def compute_stats(results):
    by_temp = defaultdict(list)
    for r in results:
        by_temp[r.temperature].append(r)

    stats = {}
    for temp in sorted(by_temp.keys()):
        rows = by_temp[temp]
        n = len(rows)
        passes = sum(1 for r in rows if r.keyword_pass)
        stats[temp] = {
            "n": n,
            "keyword_pass_rate": round(passes / n, 4),
            "mean_keyword_score": round(sum(r.keyword_score for r in rows) / n, 4),
            "mean_entropy": round(sum(r.mean_entropy for r in rows) / n, 4),
            "mean_top1_prob": round(sum(r.mean_top1_prob for r in rows) / n, 4),
            "mean_correct_prob": round(sum(r.mean_correct_prob for r in rows) / n, 6),
            "by_category": {},
            "by_difficulty": {},
        }
        # Category breakdown
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r.category].append(r)
        for cat, cat_rows in sorted(by_cat.items()):
            cn = len(cat_rows)
            cp = sum(1 for r in cat_rows if r.keyword_pass)
            stats[temp]["by_category"][cat] = {
                "n": cn, "pass_rate": round(cp / cn, 4),
            }
        # Difficulty breakdown
        by_diff = defaultdict(list)
        for r in rows:
            by_diff[r.difficulty].append(r)
        for diff, diff_rows in sorted(by_diff.items()):
            dn = len(diff_rows)
            dp = sum(1 for r in diff_rows if r.keyword_pass)
            stats[temp]["by_difficulty"][str(diff)] = {
                "n": dn, "pass_rate": round(dp / dn, 4),
            }
    return stats


def find_optimal(stats):
    best = max(stats.keys(), key=lambda t: stats[t]["keyword_pass_rate"])
    base_rate = stats.get(1.0, stats[max(stats.keys())])["keyword_pass_rate"]
    best_rate = stats[best]["keyword_pass_rate"]
    delta = (best_rate - base_rate) * 100
    if delta > 5:
        verdict = f"CONFIRMED (+{delta:.1f}pp): temperature reduction helps"
    elif delta > 2:
        verdict = f"PARTIAL (+{delta:.1f}pp): modest effect"
    elif delta < -2:
        verdict = f"REJECTED: temperature reduction hurts factuality"
    else:
        verdict = f"NEUTRAL (<2pp difference)"
    return best, verdict


def print_report(stats, model_label):
    print("\n" + "=" * 70)
    print(f"TEMPERATURE SWEEP — {model_label}")
    print("Hypothesis: lower T surfaces suppressed factual tokens")
    print("=" * 70)

    base_rate = stats.get(1.0, stats[max(stats.keys())])["keyword_pass_rate"]
    print(f"\n{'Temp':>6}  {'Pass%':>7}  {'KwScore':>8}  {'Entropy':>9}  "
          f"{'Top1P':>7}  {'Delta':>8}")
    print("-" * 58)
    for temp in sorted(stats.keys()):
        s = stats[temp]
        delta = (s["keyword_pass_rate"] - base_rate) * 100
        tag = f"{delta:+.1f}pp" if temp != 1.0 else "baseline"
        print(f"{temp:>6.2f}  {s['keyword_pass_rate']:>7.1%}  "
              f"{s['mean_keyword_score']:>8.4f}  "
              f"{s['mean_entropy']:>9.4f}  "
              f"{s['mean_top1_prob']:>7.4f}  "
              f"{tag:>8}")

    best_temp, verdict = find_optimal(stats)
    print(f"\nFINDING: {verdict}")

    # Entropy trend
    print(f"\nEntropy trend:")
    for temp in sorted(stats.keys()):
        bar = "█" * int(stats[temp]["mean_entropy"] * 4)
        print(f"  T={temp:.2f}  {stats[temp]['mean_entropy']:.4f}  {bar}")

    # Category breakdown
    best_s = stats[best_temp]
    base_s = stats.get(1.0, stats[max(stats.keys())])
    if best_temp != 1.0 and best_s["by_category"]:
        print(f"\nPer-category: T={best_temp} vs T=1.0")
        print(f"  {'Category':<35} {'BestT':>7}  {'T=1.0':>7}  {'Δ':>7}")
        print("  " + "-" * 58)
        for cat in sorted(best_s["by_category"].keys()):
            br = best_s["by_category"][cat]["pass_rate"]
            dr = base_s["by_category"].get(cat, {}).get("pass_rate", 0)
            d = (br - dr) * 100
            arrow = "↑" if d > 2 else ("↓" if d < -2 else "→")
            print(f"  {cat:<35} {br:>7.1%}  {dr:>7.1%}  {arrow}{d:>+6.1f}pp")

    # Paper sentence
    best_rate = best_s["keyword_pass_rate"]
    print(f"\nPaper-ready sentence:")
    if best_rate > base_rate + 0.05:
        print(f'  "Temperature reduction from T=1.0 to T={best_temp} improves DPO')
        print(f'   factuality from {base_rate:.1%} to {best_rate:.1%} '
              f'(+{(best_rate-base_rate)*100:.1f}pp),')
        print(f'   confirming the AFG as distributional flattening."')
    else:
        print(f'  "Temperature had minimal effect on DPO factuality '
              f'(max +{(best_rate-base_rate)*100:.1f}pp),')
        print(f'   suggesting suppression at the training-time objective level."')


def write_outputs(results, stats, output_dir, model_label):
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"temp_sweep_{model_label}")

    # Full CSV
    if results:
        with open(prefix + "_full.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            w.writerows([asdict(r) for r in results])
        print(f"\n  Full CSV:    {prefix}_full.csv")

    # Summary JSON
    best_temp, verdict = find_optimal(stats)
    with open(prefix + "_summary.json", "w") as f:
        json.dump({
            "model_label": model_label,
            "best_temp": best_temp,
            "verdict": verdict,
            "stats": {str(k): v for k, v in stats.items()},
        }, f, indent=2)
    print(f"  Summary:     {prefix}_summary.json")

    # Paper table CSV
    base_rate = stats.get(1.0, stats[max(stats.keys())])["keyword_pass_rate"]
    with open(prefix + "_paper_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "factuality_%", "mean_entropy",
                    "mean_top1_prob", "delta_vs_T1_pp"])
        for temp in sorted(stats.keys()):
            s = stats[temp]
            w.writerow([
                temp,
                round(s["keyword_pass_rate"] * 100, 1),
                round(s["mean_entropy"], 4),
                round(s["mean_top1_prob"], 4),
                round((s["keyword_pass_rate"] - base_rate) * 100, 1),
            ])
    print(f"  Paper table: {prefix}_paper_table.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dpo_adapter", default="./outputs/llama_8b_5ep/dpo/dpo_adapter")
    parser.add_argument("--dpo_base", default="./outputs/sft_merged")
    parser.add_argument("--eval_file", default="data/eval_factuality_v2.jsonl")
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/sprint3")
    parser.add_argument("--label", default="dpo_5ep")
    args = parser.parse_args()

    n_est = args.max_prompts or 500
    est_min = n_est * len(args.temperatures) * 0.3 / 60
    print(f"\nTemperature Sweep")
    print(f"  Prompts:      {n_est}")
    print(f"  Temperatures: {sorted(args.temperatures)}")
    print(f"  Est. time:    {est_min:.0f} min  (~${est_min/60*1.50:.2f} on A100)")

    print(f"\nLoading prompts from {args.eval_file}...")
    prompts = []
    with open(args.eval_file) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line.strip())
            prompts.append(EvalPrompt(
                id=d.get("id", f"p{len(prompts)}"),
                prompt=d["prompt"],
                reference_answer=d.get("reference_answer", ""),
                must_include=d.get("must_include", []),
                must_not_include=d.get("must_not_include", []),
                category=d.get("category", "unknown"),
                question_type=d.get("question_type", "definition"),
                difficulty=d.get("difficulty", 1),
                concept=d.get("concept", ""),
            ))

    if args.max_prompts:
        # Stratified sampling
        by_cat = defaultdict(list)
        for p in prompts:
            by_cat[p.category].append(p)
        per_cat = max(1, args.max_prompts // max(len(by_cat), 1))
        prompts = []
        for cp in by_cat.values():
            prompts.extend(cp[:per_cat])
        prompts = prompts[:args.max_prompts]
        print(f"  [LIMITED] {len(prompts)} prompts (stratified)")
    else:
        print(f"  Loaded {len(prompts)} prompts")

    print(f"\nLoading DPO model...")
    model, tokenizer = load_model(args.dpo_base, args.dpo_adapter)

    t_start = time.time()
    results = run_sweep(model, tokenizer, prompts, args.temperatures, args.label)
    elapsed = time.time() - t_start

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nComplete in {elapsed/60:.1f} min")

    stats = compute_stats(results)
    print_report(stats, args.label)
    write_outputs(results, stats, args.output_dir, args.label)


if __name__ == "__main__":
    main()
