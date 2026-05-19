"""
Factuality Evaluation Script.

Tests model responses against known facts to detect hallucination.

USAGE:
    python scripts/eval_factuality.py --model-path ./outputs/sft/final_adapter --base-model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_factuality")


def evaluate_response(response: str, must_include: list[str], must_not_include: list[str]) -> dict:
    """Check if response contains required facts and avoids hallucinations."""
    response_lower = response.lower()

    included = []
    missing = []
    for term in must_include:
        if term.lower() in response_lower:
            included.append(term)
        else:
            missing.append(term)

    hallucinated = []
    for term in must_not_include:
        if term.lower() in response_lower:
            hallucinated.append(term)

    passed = len(missing) == 0 and len(hallucinated) == 0

    return {
        "passed": passed,
        "included": included,
        "missing": missing,
        "hallucinated": hallucinated,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model factuality")
    parser.add_argument("--eval-file", type=str, default="data/eval_factuality.jsonl")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    # Load eval data
    eval_data = []
    with open(args.eval_file) as f:
        for line in f:
            eval_data.append(json.loads(line))
    logger.info(f"Loaded {len(eval_data)} factuality test cases")

    # Load model
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model: {args.base_model} + {args.model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Evaluate
    results = []
    passed = 0

    for _i, item in enumerate(eval_data):
        messages = [{"role": "user", "content": item["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_tokens,
                temperature=args.temperature, top_p=0.8, do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result = evaluate_response(response, item["must_include"], item["must_not_include"])
        result["prompt"] = item["prompt"]
        result["response"] = response[:300]
        results.append(result)

        if result["passed"]:
            passed += 1

        status = "✓" if result["passed"] else "✗"
        logger.info(f"  [{status}] {item['prompt'][:60]}...")
        if result["missing"]:
            logger.info(f"      Missing: {result['missing']}")
        if result["hallucinated"]:
            logger.info(f"      Hallucinated: {result['hallucinated']}")

    # Summary
    accuracy = passed / len(eval_data)
    logger.info(f"\n{'='*60}")
    logger.info(f"FACTUALITY RESULTS: {passed}/{len(eval_data)} passed ({accuracy:.1%})")
    logger.info(f"{'='*60}")

    # Save results
    output_path = Path("outputs/factuality_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "passed": passed, "total": len(eval_data), "results": results}, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
