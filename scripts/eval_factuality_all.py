"""
Factuality Evaluation — Base vs SFT vs DPO.

Tests all three model stages on the same factuality prompts to isolate
where technical knowledge is retained vs lost.

USAGE:
    python scripts/eval_factuality_all.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --sft-adapter ./outputs/sft/final_adapter \
        --dpo-adapter ./outputs/dpo/dpo_adapter
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


def generate_responses(model, tokenizer, prompts, temperature=0.0, max_tokens=200):
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
                do_sample=False,  # Deterministic — no randomness
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)
    return responses


def main():
    parser = argparse.ArgumentParser(description="Factuality eval: Base vs SFT vs DPO")
    parser.add_argument("--eval-file", type=str, default="data/eval_factuality.jsonl")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sft-adapter", type=str, required=True)
    parser.add_argument("--dpo-adapter", type=str, required=True)
    parser.add_argument("--dpo-base", type=str, default=None,
                        help="Base model for DPO adapter (use ./outputs/sft_merged for merged-SFT DPO)")
    parser.add_argument("--max-prompts", type=int, default=None)
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
    prompts = [item["prompt"] for item in eval_data]
    logger.info(f"Loaded {len(eval_data)} factuality test cases")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    results = {}

    # ─── Evaluate Base Model ───
    logger.info("=" * 60)
    logger.info("EVALUATING: Base Model")
    logger.info("=" * 60)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_responses = generate_responses(base, tokenizer, prompts)
    base_passed = 0
    for item, resp in zip(eval_data, base_responses):
        r = evaluate_response(resp, item["must_include"], item["must_not_include"])
        if r["passed"]:
            base_passed += 1
    results["base"] = {"passed": base_passed, "total": len(eval_data), "accuracy": base_passed / len(eval_data)}
    logger.info(f"Base: {base_passed}/{len(eval_data)} ({results['base']['accuracy']:.1%})")

    # ─── Evaluate SFT Model ───
    logger.info("=" * 60)
    logger.info("EVALUATING: SFT Model")
    logger.info("=" * 60)
    sft_model = PeftModel.from_pretrained(base, args.sft_adapter)
    sft_responses = generate_responses(sft_model, tokenizer, prompts)
    sft_passed = 0
    for item, resp in zip(eval_data, sft_responses):
        r = evaluate_response(resp, item["must_include"], item["must_not_include"])
        if r["passed"]:
            sft_passed += 1
    results["sft"] = {"passed": sft_passed, "total": len(eval_data), "accuracy": sft_passed / len(eval_data)}
    logger.info(f"SFT: {sft_passed}/{len(eval_data)} ({results['sft']['accuracy']:.1%})")

    del sft_model
    torch.cuda.empty_cache()
    gc.collect()

    # ─── Evaluate DPO Model ───
    logger.info("=" * 60)
    logger.info("EVALUATING: DPO Model")
    logger.info("=" * 60)

    # If --dpo-base is specified, load DPO adapter on that base instead
    # This is needed for merged-SFT DPO (v4) where the adapter was trained
    # on the merged SFT model, not the original base.
    if args.dpo_base:
        logger.info(f"Loading DPO base model from: {args.dpo_base}")
        del base
        torch.cuda.empty_cache()
        gc.collect()
        dpo_base = AutoModelForCausalLM.from_pretrained(
            args.dpo_base, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
        )
        dpo_model = PeftModel.from_pretrained(dpo_base, args.dpo_adapter)
    else:
        dpo_model = PeftModel.from_pretrained(base, args.dpo_adapter)
    dpo_responses = generate_responses(dpo_model, tokenizer, prompts)
    dpo_passed = 0
    for item, resp in zip(eval_data, dpo_responses):
        r = evaluate_response(resp, item["must_include"], item["must_not_include"])
        if r["passed"]:
            dpo_passed += 1
    results["dpo"] = {"passed": dpo_passed, "total": len(eval_data), "accuracy": dpo_passed / len(eval_data)}
    logger.info(f"DPO: {dpo_passed}/{len(eval_data)} ({results['dpo']['accuracy']:.1%})")

    del dpo_model
    if args.dpo_base:
        del dpo_base
    else:
        del base
    torch.cuda.empty_cache()
    gc.collect()

    # ─── Summary ───
    logger.info("\n" + "=" * 60)
    logger.info("FACTUALITY COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  Base: {results['base']['passed']}/{results['base']['total']} ({results['base']['accuracy']:.1%})")
    logger.info(f"  SFT:  {results['sft']['passed']}/{results['sft']['total']} ({results['sft']['accuracy']:.1%})")
    logger.info(f"  DPO:  {results['dpo']['passed']}/{results['dpo']['total']} ({results['dpo']['accuracy']:.1%})")
    logger.info("=" * 60)

    # Save
    output_path = Path("outputs/factuality_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
