"""
Factuality Evaluation — Base vs SFT vs DPO.

Tests all three model stages on the same factuality prompts to isolate
where technical knowledge is retained vs lost.

Each model stage is loaded fresh (no shared base object) to avoid
adapter contamination between evaluations.

USAGE:
    # Standard (stacked adapter DPO):
    python scripts/eval_factuality_all.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --sft-adapter ./outputs/sft/final_adapter \
        --dpo-adapter ./outputs/dpo/dpo_adapter

    # Merged-SFT DPO (v4 — DPO trained on merged base):
    python scripts/eval_factuality_all.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --sft-adapter ./outputs/sft/final_adapter \
        --dpo-adapter ./outputs/dpo/dpo_adapter \
        --dpo-base ./outputs/sft_merged
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_factuality_all")


def evaluate_response(response: str, must_include: list[str], must_not_include: list[str]) -> dict:
    """Score a single response against factuality criteria."""
    response_lower = response.lower()
    included = [t for t in must_include if t.lower() in response_lower]
    missing = [t for t in must_include if t.lower() not in response_lower]
    hallucinated = [t for t in must_not_include if t.lower() in response_lower]
    return {
        "passed": len(missing) == 0 and len(hallucinated) == 0,
        "included": included,
        "missing": missing,
        "hallucinated": hallucinated,
    }


def generate_responses(model, tokenizer, prompts, max_tokens=200):
    """Generate responses with deterministic decoding (temperature=0)."""
    import torch

    responses = []
    model.eval()
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)
    return responses


def load_base_model(model_id: str):
    """Load a fresh base model with QLoRA quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def unload_model(model):
    """Fully unload a model and free GPU memory."""
    import torch
    del model
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_stage(stage_name: str, model, tokenizer, eval_data, prompts) -> dict:
    """Evaluate a single model stage and return results."""
    logger.info("=" * 60)
    logger.info(f"EVALUATING: {stage_name}")
    logger.info("=" * 60)

    responses = generate_responses(model, tokenizer, prompts)
    passed = 0
    details = []
    for item, resp in zip(eval_data, responses):
        r = evaluate_response(resp, item["must_include"], item["must_not_include"])
        if r["passed"]:
            passed += 1
        details.append({"prompt": item["prompt"], "response": resp[:500], **r})

    accuracy = passed / len(eval_data)
    logger.info(f"{stage_name}: {passed}/{len(eval_data)} ({accuracy:.1%})")
    return {"passed": passed, "total": len(eval_data), "accuracy": accuracy, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Factuality eval: Base vs SFT vs DPO")
    parser.add_argument("--eval-file", type=str, default="data/eval_factuality.jsonl")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sft-adapter", type=str, required=True)
    parser.add_argument("--dpo-adapter", type=str, required=True)
    parser.add_argument("--dpo-base", type=str, default=None,
                        help="Base model for DPO adapter. Use ./outputs/sft_merged for merged-SFT DPO (v4).")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--save-responses", action="store_true",
                        help="Save raw responses for semantic eval")
    args = parser.parse_args()

    from peft import PeftModel

    # Load eval data
    eval_data = []
    with open(args.eval_file) as f:
        for line in f:
            eval_data.append(json.loads(line))
    if args.max_prompts:
        eval_data = eval_data[:args.max_prompts]
    prompts = [item["prompt"] for item in eval_data]
    logger.info(f"Loaded {len(eval_data)} factuality test cases")

    results = {}
    all_responses = {}

    # ─── Stage 1: Base Model (fresh load) ───
    logger.info("\n[1/3] Loading fresh base model...")
    base_model, tokenizer = load_base_model(args.base_model)
    results["base"] = evaluate_stage("Base Model", base_model, tokenizer, eval_data, prompts)
    if args.save_responses:
        all_responses["base"] = results["base"]["details"]
    unload_model(base_model)

    # ─── Stage 2: SFT Model (fresh base + SFT adapter) ───
    logger.info("\n[2/3] Loading fresh base + SFT adapter...")
    sft_base, tokenizer = load_base_model(args.base_model)
    sft_model = PeftModel.from_pretrained(sft_base, args.sft_adapter)
    results["sft"] = evaluate_stage("SFT Model", sft_model, tokenizer, eval_data, prompts)
    if args.save_responses:
        all_responses["sft"] = results["sft"]["details"]
    unload_model(sft_model)
    unload_model(sft_base)

    # ─── Stage 3: DPO Model (fresh base + DPO adapter) ───
    # For merged-SFT DPO (v4): load the merged model as base
    # For stacked DPO (v1-v3): load original base
    dpo_base_path = args.dpo_base if args.dpo_base else args.base_model
    logger.info(f"\n[3/3] Loading fresh DPO base ({dpo_base_path}) + DPO adapter...")
    dpo_base, tokenizer = load_base_model(dpo_base_path)
    dpo_model = PeftModel.from_pretrained(dpo_base, args.dpo_adapter)
    results["dpo"] = evaluate_stage("DPO Model", dpo_model, tokenizer, eval_data, prompts)
    if args.save_responses:
        all_responses["dpo"] = results["dpo"]["details"]
    unload_model(dpo_model)
    unload_model(dpo_base)

    # ─── Summary ───
    logger.info("\n" + "=" * 60)
    logger.info("FACTUALITY COMPARISON")
    logger.info("=" * 60)
    for stage in ["base", "sft", "dpo"]:
        r = results[stage]
        logger.info(f"  {stage.upper():<5}: {r['passed']}/{r['total']} ({r['accuracy']:.1%})")
    logger.info("=" * 60)

    if args.dpo_base:
        logger.info(f"  Note: DPO evaluated on merged base ({args.dpo_base})")
    else:
        logger.info("  Note: DPO evaluated on original base (stacked adapter mode)")

    # ─── Save results ───
    output_path = Path("outputs/factuality_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip details for summary file (keep it small)
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "details"} for k, v in results.items()}
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {output_path}")

    # Save responses for semantic eval if requested
    if args.save_responses:
        resp_path = Path("outputs/factuality_responses.json")
        with open(resp_path, "w") as f:
            json.dump(all_responses, f, indent=2)
        logger.info(f"Responses saved to: {resp_path}")


if __name__ == "__main__":
    main()
