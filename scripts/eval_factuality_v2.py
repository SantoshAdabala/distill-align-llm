"""
eval_factuality_v2.py
=====================
Replaces eval_factuality_all.py with:
  - 500-prompt eval set (eval_factuality_v2.jsonl)
  - LLM-as-judge scoring (GPT-4o-mini, 0-3 scale)
  - Per-category breakdown
  - Continuous AFG score (not binary pass/fail)
  - Full CSV output for paper tables

Usage:
  python scripts/eval_factuality_v2.py \
      --base_model meta-llama/Llama-3.1-8B-Instruct \
      --sft_adapter ./outputs/sft/final_adapter \
      --dpo_adapter ./outputs/dpo/dpo_adapter \
      --dpo_base    ./outputs/sft_merged \
      --eval_file   data/eval_factuality_v2.jsonl \
      --output_dir  outputs/eval_v2 \
      --openai_key  $OPENAI_API_KEY
"""

import json
import os
import csv
import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI

# ── Data classes ──────────────────────────────────────────────────────────────

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
    source: str = ""

@dataclass
class EvalResult:
    prompt_id: str
    category: str
    question_type: str
    difficulty: int
    model_stage: str
    response: str
    # Keyword-based score (backward compat with v1)
    keyword_pass: bool
    keyword_score: float        # fraction of must_include keywords present
    # LLM-judge score
    judge_score: int            # 0-3
    judge_reason: str
    judge_normalized: float     # judge_score / 3.0

# ── Prompts ───────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert ML engineer evaluating factual accuracy of language model responses.
You are strict but fair. You care about whether the key technical facts are present and correct.
Return ONLY valid JSON with no markdown, no preamble."""

JUDGE_USER = """Evaluate this language model response for factual accuracy.

Question: {question}
Reference answer: {reference}
Must include (at least 2 of these): {must_include}
Model response: {response}

Score the response 0-3:
  3 = Fully correct. All key facts present and accurate.
  2 = Mostly correct. Minor omission or imprecise wording, but core fact is right.
  1 = Partially correct. Key fact is missing or confused with something else.
  0 = Incorrect or hallucinated. Wrong facts stated confidently.

Return JSON: {{"score": <int 0-3>, "reason": "<one sentence explanation>"}}"""

INFERENCE_SYSTEM = """You are a helpful ML engineering assistant with deep knowledge of
LLM training, fine-tuning, alignment, and inference optimization.
Answer questions factually and concisely."""

# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(model_id: str, device: str = "auto"):
    """Load quantized base model."""
    print(f"  Loading base model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer

def load_sft_model(base_model_id: str, adapter_path: str):
    """Load base + SFT LoRA adapter."""
    print(f"  Loading SFT model from adapter: {adapter_path}")
    model, tokenizer = load_base_model(base_model_id)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def load_dpo_model(dpo_base_path: str, dpo_adapter_path: str):
    """Load merged-SFT base + DPO LoRA adapter."""
    print(f"  Loading DPO model: base={dpo_base_path}, adapter={dpo_adapter_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(dpo_base_path)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        dpo_base_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, dpo_adapter_path)
    model.eval()
    return model, tokenizer

# ── Inference ─────────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate a response at temperature=0 (deterministic)."""
    messages = [
        {"role": "system", "content": INFERENCE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # temperature=0
            temperature=1.0,       # required when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── Scoring ───────────────────────────────────────────────────────────────────

def keyword_score(response: str, must_include: list, must_not_include: list) -> tuple[bool, float]:
    """v1-compatible keyword scoring for backward compatibility."""
    resp_lower = response.lower()
    hits = sum(1 for kw in must_include if kw.lower() in resp_lower)
    bads = sum(1 for kw in must_not_include if kw.lower() in resp_lower)
    score = hits / max(len(must_include), 1)
    passed = hits >= max(1, len(must_include) // 2) and bads == 0
    return passed, score

def judge_score(
    client: OpenAI,
    question: str,
    reference: str,
    must_include: list,
    response: str,
    retries: int = 3,
) -> tuple[int, str]:
    """Call GPT-4o-mini to score a response 0-3."""
    # Fallback: if no reference, use must_include as the reference
    ref = reference if reference.strip() else f"Must mention: {', '.join(must_include)}"
    user_msg = JUDGE_USER.format(
        question=question,
        reference=ref,
        must_include=", ".join(must_include),
        response=response[:800],  # truncate to save tokens
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            score = int(data.get("score", 0))
            reason = data.get("reason", "")
            return max(0, min(3, score)), reason
        except Exception as e:
            if attempt == retries - 1:
                print(f"    [WARN] Judge failed after {retries} attempts: {e}")
                return 0, f"Judge error: {e}"
            time.sleep(1)
    return 0, "Judge error"

# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_eval(
    model,
    tokenizer,
    prompts: list[EvalPrompt],
    stage_name: str,
    judge_client: OpenAI,
    max_new_tokens: int = 200,
) -> list[EvalResult]:
    """Run full eval for one model stage."""
    results = []
    n = len(prompts)
    print(f"\n  Evaluating {stage_name} on {n} prompts...")

    for i, ep in enumerate(prompts):
        # Generate response
        response = generate_response(model, tokenizer, ep.prompt, max_new_tokens)

        # Keyword score (v1 compat)
        kw_pass, kw_score = keyword_score(response, ep.must_include, ep.must_not_include)

        # LLM judge score
        j_score, j_reason = judge_score(
            judge_client, ep.prompt, ep.reference_answer, ep.must_include, response
        )

        results.append(EvalResult(
            prompt_id=ep.id,
            category=ep.category,
            question_type=ep.question_type,
            difficulty=ep.difficulty,
            model_stage=stage_name,
            response=response,
            keyword_pass=kw_pass,
            keyword_score=kw_score,
            judge_score=j_score,
            judge_reason=j_reason,
            judge_normalized=j_score / 3.0,
        ))

        if (i + 1) % 50 == 0:
            so_far = results[-50:]
            avg_judge = sum(r.judge_normalized for r in so_far) / len(so_far)
            avg_kw = sum(r.keyword_score for r in so_far) / len(so_far)
            print(f"    [{i+1}/{n}] Judge avg: {avg_judge:.3f}  Keyword avg: {avg_kw:.3f}")

        # Gentle rate limiting for judge API
        if (i + 1) % 20 == 0:
            time.sleep(1)

    return results

# ── Reporting ─────────────────────────────────────────────────────────────────

def compute_afg(results_by_stage: dict[str, list[EvalResult]]) -> dict:
    """
    Compute the Alignment-Factuality Gap.

    AFG = reward_accuracy(DPO) - factuality_score(DPO)

    Both metrics normalized to [0, 1].
    Factuality score = mean judge_normalized across all eval prompts.
    Reward accuracy is read from DPO training logs (passed in separately).
    """
    afg_data = {}
    for stage, results in results_by_stage.items():
        avg_judge = sum(r.judge_normalized for r in results) / len(results)
        avg_kw = sum(1 for r in results if r.keyword_pass) / len(results)
        afg_data[stage] = {
            "n_prompts": len(results),
            "factuality_judge": round(avg_judge, 4),
            "factuality_keyword": round(avg_kw, 4),
            "factuality_by_category": {},
            "factuality_by_difficulty": {},
            "factuality_by_qtype": {},
        }
        # Category breakdown
        by_cat = defaultdict(list)
        for r in results:
            by_cat[r.category].append(r.judge_normalized)
        afg_data[stage]["factuality_by_category"] = {
            cat: round(sum(scores) / len(scores), 4)
            for cat, scores in by_cat.items()
        }
        # Difficulty breakdown
        by_diff = defaultdict(list)
        for r in results:
            by_diff[r.difficulty].append(r.judge_normalized)
        afg_data[stage]["factuality_by_difficulty"] = {
            f"level_{d}": round(sum(scores) / len(scores), 4)
            for d, scores in by_diff.items()
        }
        # Question type breakdown
        by_qt = defaultdict(list)
        for r in results:
            by_qt[r.question_type].append(r.judge_normalized)
        afg_data[stage]["factuality_by_qtype"] = {
            qt: round(sum(scores) / len(scores), 4)
            for qt, scores in by_qt.items()
        }
    return afg_data

def print_summary(afg_data: dict, dpo_reward_accuracy: float = None):
    """Print a paper-ready summary table."""
    print("\n" + "=" * 70)
    print("FACTUALITY EVALUATION RESULTS (v2 — LLM Judge)")
    print("=" * 70)

    stages = list(afg_data.keys())

    # Main table
    print(f"\n{'Stage':<20} {'Judge Score':>12} {'Keyword Score':>14} {'N':>6}")
    print("-" * 55)
    for stage in stages:
        d = afg_data[stage]
        print(f"{stage:<20} {d['factuality_judge']:>12.1%} "
              f"{d['factuality_keyword']:>14.1%} {d['n_prompts']:>6}")

    # AFG computation
    if "dpo" in afg_data and dpo_reward_accuracy is not None:
        dpo_fact = afg_data["dpo"]["factuality_judge"]
        afg = dpo_reward_accuracy - dpo_fact
        print(f"\n{'─'*55}")
        print(f"  DPO Reward Accuracy:        {dpo_reward_accuracy:.1%}")
        print(f"  DPO Domain Factuality:       {dpo_fact:.1%}")
        print(f"  Alignment-Factuality Gap:    {afg:.1%}  ({afg*100:.1f} points)")

    # Category table
    print("\nPer-category factuality (judge score):")
    all_cats = set()
    for d in afg_data.values():
        all_cats.update(d["factuality_by_category"].keys())

    header = f"{'Category':<30}" + "".join(f"{s:>12}" for s in stages)
    print(header)
    print("-" * (30 + 12 * len(stages)))
    for cat in sorted(all_cats):
        row = f"{cat:<30}"
        for stage in stages:
            val = afg_data[stage]["factuality_by_category"].get(cat, 0)
            row += f"{val:>12.1%}"
        print(row)

    # Difficulty table
    print("\nPer-difficulty factuality (judge score):")
    for stage in stages:
        print(f"\n  {stage}:")
        for diff_key, val in sorted(afg_data[stage]["factuality_by_difficulty"].items()):
            level = diff_key.replace("level_", "")
            labels = {"1": "Easy (definitions)", "2": "Medium (mechanism/numerical)",
                      "3": "Hard (application/causal)"}
            label = labels.get(str(level), "")
            print(f"    Level {level} {label:<35} {val:.1%}")

def write_csv(results_by_stage: dict, output_dir: str):
    """Write full results to CSV for paper tables."""
    os.makedirs(output_dir, exist_ok=True)

    # Full results CSV
    all_rows = []
    for stage, results in results_by_stage.items():
        for r in results:
            all_rows.append(asdict(r))

    full_path = os.path.join(output_dir, "eval_v2_full_results.csv")
    with open(full_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Full results: {full_path}")

    # Summary CSV (one row per stage × category)
    summary_path = os.path.join(output_dir, "eval_v2_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "category", "question_type", "difficulty",
                         "n", "judge_mean", "judge_std", "keyword_pass_rate"])
        for stage, results in results_by_stage.items():
            by_cat_qt = defaultdict(list)
            for r in results:
                key = (r.category, r.question_type, r.difficulty)
                by_cat_qt[key].append(r)
            for (cat, qt, diff), rlist in sorted(by_cat_qt.items()):
                scores = [r.judge_normalized for r in rlist]
                kw_pass = sum(1 for r in rlist if r.keyword_pass) / len(rlist)
                mean = sum(scores) / len(scores)
                std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
                writer.writerow([stage, cat, qt, diff, len(rlist),
                                  round(mean, 4), round(std, 4), round(kw_pass, 4)])
    print(f"  Summary CSV:  {summary_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sft_adapter", default="./outputs/sft/final_adapter")
    parser.add_argument("--dpo_adapter", default="./outputs/dpo/dpo_adapter")
    parser.add_argument("--dpo_base", default="./outputs/sft_merged")
    parser.add_argument("--eval_file", default="data/eval_factuality_v2.jsonl")
    parser.add_argument("--output_dir", default="outputs/eval_v2")
    parser.add_argument("--openai_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--dpo_reward_accuracy", type=float, default=0.82,
                        help="DPO reward accuracy from training logs (for AFG calc)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--stages", nargs="+", default=["base", "sft", "dpo"],
                        choices=["base", "sft", "dpo"],
                        help="Which model stages to evaluate")
    args = parser.parse_args()

    if not args.openai_key:
        raise ValueError("Provide --openai_key or set OPENAI_API_KEY")

    judge_client = OpenAI(api_key=args.openai_key)

    # Load eval prompts
    print(f"\nLoading eval prompts from {args.eval_file}...")
    prompts = []
    with open(args.eval_file) as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(EvalPrompt(
                id=data.get("id", ""),
                prompt=data["prompt"],
                reference_answer=data.get("reference_answer", ""),
                must_include=data.get("must_include", []),
                must_not_include=data.get("must_not_include", []),
                category=data.get("category", "unknown"),
                question_type=data.get("question_type", "definition"),
                difficulty=data.get("difficulty", 1),
                concept=data.get("concept", ""),
                source=data.get("source", ""),
            ))
    print(f"  Loaded {len(prompts)} prompts")

    results_by_stage = {}

    # ── Base model ──
    if "base" in args.stages:
        print("\n[Stage 1/3] Base model")
        model, tokenizer = load_base_model(args.base_model)
        results_by_stage["base"] = run_eval(
            model, tokenizer, prompts, "base", judge_client, args.max_new_tokens
        )
        del model
        torch.cuda.empty_cache()

    # ── SFT model ──
    if "sft" in args.stages:
        print("\n[Stage 2/3] SFT model")
        model, tokenizer = load_sft_model(args.base_model, args.sft_adapter)
        results_by_stage["sft"] = run_eval(
            model, tokenizer, prompts, "sft", judge_client, args.max_new_tokens
        )
        del model
        torch.cuda.empty_cache()

    # ── DPO model ──
    if "dpo" in args.stages:
        print("\n[Stage 3/3] DPO model")
        model, tokenizer = load_dpo_model(args.dpo_base, args.dpo_adapter)
        results_by_stage["dpo"] = run_eval(
            model, tokenizer, prompts, "dpo", judge_client, args.max_new_tokens
        )
        del model
        torch.cuda.empty_cache()

    # ── Reporting ──
    afg_data = compute_afg(results_by_stage)
    print_summary(afg_data, dpo_reward_accuracy=args.dpo_reward_accuracy)
    write_csv(results_by_stage, args.output_dir)

    # Save AFG data as JSON for dashboard
    afg_path = os.path.join(args.output_dir, "afg_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(afg_path, "w") as f:
        json.dump({
            "dpo_reward_accuracy": args.dpo_reward_accuracy,
            "stages": afg_data,
        }, f, indent=2)
    print(f"  AFG JSON:     {afg_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
