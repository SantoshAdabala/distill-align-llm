"""
Token Probability Analysis — Does the model KNOW but not GENERATE?

For each factuality prompt, measures the probability the model assigns to
the correct answer tokens across Base → SFT → DPO stages.

This reveals whether:
- The model has zero probability on correct tokens (doesn't know)
- The model has non-zero probability but doesn't generate them (knows but suppressed)

USAGE:
    python scripts/eval_token_probs.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --sft-adapter ./outputs/sft/final_adapter \
        --dpo-adapter ./outputs/dpo/dpo_adapter \
        --eval-file data/eval_factuality.jsonl

OUTPUT:
    outputs/token_probability_analysis.json
    - For each prompt: P(correct_tokens) for Base, SFT, DPO
    - Aggregated: avg probability, % where P > 0.01, % where P > 0.1
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("token_probs")


def measure_token_probability(model, tokenizer, prompt: str, target_text: str) -> dict:
    """Measure the probability the model assigns to target_text given prompt.

    Returns:
        dict with avg_log_prob, avg_prob, min_prob, max_prob, token_probs
    """
    import torch
    import torch.nn.functional as F

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize prompt + target together
    full_text = chat_text + target_text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    prompt_inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_len = inputs["input_ids"].shape[1]

    if full_len <= prompt_len:
        return {"avg_prob": 0.0, "avg_log_prob": -100.0, "min_prob": 0.0,
                "max_prob": 0.0, "token_probs": [], "num_target_tokens": 0}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Get probabilities for target tokens
    # logits[t] predicts token[t+1], so we look at logits[prompt_len-1 : full_len-1]
    target_token_ids = inputs["input_ids"][0, prompt_len:full_len]
    target_logits = logits[0, prompt_len - 1:full_len - 1, :]  # [target_len, vocab_size]

    # Convert to probabilities
    probs = F.softmax(target_logits, dim=-1)
    log_probs = F.log_softmax(target_logits, dim=-1)

    # Get probability of each target token
    token_probs = []
    for i, token_id in enumerate(target_token_ids):
        prob = probs[i, token_id].item()
        log_prob = log_probs[i, token_id].item()
        token_text = tokenizer.decode([token_id])
        token_probs.append({
            "token": token_text,
            "token_id": token_id.item(),
            "prob": prob,
            "log_prob": log_prob,
        })

    avg_prob = sum(t["prob"] for t in token_probs) / len(token_probs) if token_probs else 0.0
    avg_log_prob = sum(t["log_prob"] for t in token_probs) / len(token_probs) if token_probs else -100.0
    min_prob = min(t["prob"] for t in token_probs) if token_probs else 0.0
    max_prob = max(t["prob"] for t in token_probs) if token_probs else 0.0

    return {
        "avg_prob": avg_prob,
        "avg_log_prob": avg_log_prob,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "num_target_tokens": len(token_probs),
        "token_probs": token_probs[:20],  # Keep first 20 for inspection
    }


def main():
    parser = argparse.ArgumentParser(description="Token probability analysis")
    parser.add_argument("--eval-file", type=str, default="data/eval_factuality.jsonl")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sft-adapter", type=str, required=True)
    parser.add_argument("--dpo-adapter", type=str, required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/token_probability_analysis.json")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Load eval data
    eval_data = []
    with open(args.eval_file) as f:
        for line in f:
            eval_data.append(json.loads(line))
    if args.max_prompts:
        eval_data = eval_data[:args.max_prompts]
    logger.info(f"Loaded {len(eval_data)} factuality test cases")

    # For token probability, we need the target text
    # Use must_include terms joined as the "correct answer"
    for item in eval_data:
        item["target_text"] = " ".join(item["must_include"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    results = []

    # ─── Load Base Model ───
    logger.info("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Evaluate Base ───
    logger.info("Measuring token probabilities: Base Model")
    for i, item in enumerate(eval_data):
        result = {"prompt": item["prompt"], "target": item["target_text"]}
        result["base"] = measure_token_probability(base, tokenizer, item["prompt"], item["target_text"])
        results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  Base: {i+1}/{len(eval_data)}")

    # ─── Evaluate SFT ───
    logger.info("Loading SFT adapter...")
    sft_model = PeftModel.from_pretrained(base, args.sft_adapter)
    logger.info("Measuring token probabilities: SFT Model")
    for i, item in enumerate(eval_data):
        results[i]["sft"] = measure_token_probability(sft_model, tokenizer, item["prompt"], item["target_text"])
        if (i + 1) % 10 == 0:
            logger.info(f"  SFT: {i+1}/{len(eval_data)}")

    del sft_model
    torch.cuda.empty_cache()
    gc.collect()

    # ─── Evaluate DPO ───
    logger.info("Loading DPO adapter...")
    dpo_model = PeftModel.from_pretrained(base, args.dpo_adapter)
    logger.info("Measuring token probabilities: DPO Model")
    for i, item in enumerate(eval_data):
        results[i]["dpo"] = measure_token_probability(dpo_model, tokenizer, item["prompt"], item["target_text"])
        if (i + 1) % 10 == 0:
            logger.info(f"  DPO: {i+1}/{len(eval_data)}")

    del dpo_model, base
    torch.cuda.empty_cache()
    gc.collect()

    # ─── Aggregate Results ───
    summary = {}
    for stage in ["base", "sft", "dpo"]:
        probs = [r[stage]["avg_prob"] for r in results]
        summary[stage] = {
            "mean_prob": sum(probs) / len(probs),
            "pct_above_001": sum(1 for p in probs if p > 0.01) / len(probs),
            "pct_above_01": sum(1 for p in probs if p > 0.1) / len(probs),
            "pct_above_05": sum(1 for p in probs if p > 0.5) / len(probs),
        }

    # ─── Print Summary ───
    logger.info("\n" + "=" * 70)
    logger.info("TOKEN PROBABILITY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"{'Stage':<8} {'Mean P(correct)':<18} {'P>0.01':<10} {'P>0.1':<10} {'P>0.5':<10}")
    logger.info("-" * 70)
    for stage in ["base", "sft", "dpo"]:
        s = summary[stage]
        logger.info(f"{stage:<8} {s['mean_prob']:<18.4f} {s['pct_above_001']:<10.1%} "
                    f"{s['pct_above_01']:<10.1%} {s['pct_above_05']:<10.1%}")
    logger.info("=" * 70)

    interpretation = ""
    if summary["base"]["mean_prob"] < 0.01:
        interpretation = "Model does NOT know the answers (very low probability on correct tokens)"
    elif summary["base"]["mean_prob"] > 0.1 and summary["dpo"]["mean_prob"] < 0.05:
        interpretation = "Model KNOWS but DPO SUPPRESSES (probability drops after DPO)"
    elif summary["sft"]["mean_prob"] > summary["base"]["mean_prob"] * 2:
        interpretation = "SFT TEACHES knowledge (probability increases after SFT)"
    else:
        interpretation = "Knowledge is weakly present but not reliably generated"

    logger.info(f"\nInterpretation: {interpretation}")

    # ─── Save ───
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "summary": summary,
        "interpretation": interpretation,
        "num_prompts": len(eval_data),
        "details": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
