"""
Semantic Factuality Evaluation — LLM-Judge + Embedding Similarity.

Evaluates model responses using three methods:
1. Exact match (existing keyword-based approach)
2. Semantic similarity (embedding cosine similarity)
3. LLM judge (GPT-4o-mini or local model scores correctness)

USAGE:
    # Full evaluation with all three methods
    python scripts/eval_semantic.py \
        --responses outputs/factuality_responses.json \
        --eval-file data/eval_factuality_500.jsonl \
        --judge openai

    # Just semantic similarity (no API needed)
    python scripts/eval_semantic.py \
        --responses outputs/factuality_responses.json \
        --eval-file data/eval_factuality_500.jsonl \
        --methods exact,semantic
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_semantic")


def exact_match_eval(response: str, must_include: list, must_not_include: list) -> dict:
    """Original strict keyword matching."""
    response_lower = response.lower()
    included = [t for t in must_include if t.lower() in response_lower]
    missing = [t for t in must_include if t.lower() not in response_lower]
    hallucinated = [t for t in must_not_include if t.lower() in response_lower]
    passed = len(missing) == 0 and len(hallucinated) == 0
    return {"method": "exact", "passed": passed, "score": 1.0 if passed else 0.0,
            "missing": missing, "hallucinated": hallucinated}


def semantic_similarity_eval(response: str, reference_terms: list, model=None, gold_answer: str = "") -> dict:
    """Evaluate using embedding cosine similarity.

    Compares the first sentence of the response against the gold_answer
    (or must_include terms if no gold_answer). Score > 0.5 is considered a pass.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning("sentence-transformers not installed. pip install sentence-transformers")
        return {"method": "semantic", "passed": False, "score": 0.0, "error": "missing dependency"}

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Use gold_answer if available, otherwise join must_include terms
    reference_text = gold_answer if gold_answer else " ".join(reference_terms)

    # Use first 2 sentences of response (not full paragraph) to reduce dilution
    response_short = ". ".join(response.split(". ")[:2])

    embeddings = model.encode([response_short, reference_text])
    similarity = float(np.dot(embeddings[0], embeddings[1]) /
                       (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))

    passed = similarity > 0.50
    return {"method": "semantic", "passed": passed, "score": similarity}


def llm_judge_eval(prompt: str, response: str, must_include: list,
                   judge_model: str = "openai") -> dict:
    """Evaluate using an LLM judge.

    Asks GPT-4o-mini (or local model) to score whether the response
    correctly answers the question with the expected concepts.
    """
    reference = ", ".join(must_include)
    judge_prompt = f"""You are evaluating whether an AI model's response correctly answers a technical ML question.

Question: {prompt}
Expected concepts that should be mentioned: {reference}
Model's response: {response}

Score the response on factual correctness from 0 to 5:
- 5: Completely correct, mentions all key concepts
- 4: Mostly correct, minor omissions
- 3: Partially correct, some key concepts present
- 2: Mostly incorrect but shows some relevant knowledge
- 1: Incorrect, contains hallucinations
- 0: Completely wrong or irrelevant

Respond with ONLY a JSON object: {{"score": <0-5>, "reasoning": "<brief explanation>"}}"""

    if judge_model == "openai":
        try:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=200,
            )
            result_text = resp.choices[0].message.content.strip()
            result = json.loads(result_text)
            score = result.get("score", 0) / 5.0
            return {"method": "llm_judge", "passed": score >= 0.6,
                    "score": score, "reasoning": result.get("reasoning", "")}
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return {"method": "llm_judge", "passed": False, "score": 0.0, "error": str(e)}
    else:
        return {"method": "llm_judge", "passed": False, "score": 0.0, "error": "unsupported judge"}


def main():
    parser = argparse.ArgumentParser(description="Semantic factuality evaluation")
    parser.add_argument("--responses", type=str, required=True,
                        help="JSON file with model responses (from eval_factuality_all.py)")
    parser.add_argument("--eval-file", type=str, default="data/eval_factuality_500.jsonl")
    parser.add_argument("--methods", type=str, default="exact,semantic,llm_judge",
                        help="Comma-separated eval methods")
    parser.add_argument("--judge", type=str, default="openai", help="LLM judge backend")
    parser.add_argument("--output", type=str, default="outputs/semantic_eval_results.json")
    args = parser.parse_args()

    methods = args.methods.split(",")

    # Load eval data
    eval_data = []
    with open(args.eval_file) as f:
        for line in f:
            eval_data.append(json.loads(line))
    logger.info(f"Loaded {len(eval_data)} eval prompts")

    # Load responses
    with open(args.responses) as f:
        responses = json.load(f)
    logger.info(f"Loaded responses for models: {list(responses.keys())}")

    # Initialize semantic model if needed
    sem_model = None
    if "semantic" in methods:
        try:
            from sentence_transformers import SentenceTransformer
            sem_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers model")
        except ImportError:
            logger.warning("sentence-transformers not available, skipping semantic eval")
            methods = [m for m in methods if m != "semantic"]

    results = {}
    for model_name, model_responses in responses.items():
        logger.info(f"\nEvaluating: {model_name}")
        model_results = {"exact": [], "semantic": [], "llm_judge": []}

        for i, (item, response_data) in enumerate(zip(eval_data, model_responses)):
            # Handle both plain string responses and dict responses
            if isinstance(response_data, dict):
                response = response_data.get("response", "")
            else:
                response = response_data

            if "exact" in methods:
                r = exact_match_eval(response, item["must_include"], item["must_not_include"])
                model_results["exact"].append(r)

            if "semantic" in methods:
                gold = item.get("gold_answer", "")
                r = semantic_similarity_eval(response, item["must_include"], sem_model, gold_answer=gold)
                model_results["semantic"].append(r)

            if "llm_judge" in methods:
                r = llm_judge_eval(item["prompt"], response, item["must_include"], args.judge)
                model_results["llm_judge"].append(r)

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(eval_data)}")

        # Compute summary
        summary = {}
        for method in methods:
            if model_results[method]:
                scores = [r["score"] for r in model_results[method]]
                passed = sum(1 for r in model_results[method] if r["passed"])
                summary[method] = {
                    "passed": passed,
                    "total": len(scores),
                    "accuracy": passed / len(scores),
                    "avg_score": sum(scores) / len(scores),
                }
        results[model_name] = {"summary": summary, "details": model_results}
        logger.info(f"  {model_name} summary: {summary}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("EVALUATION COMPARISON")
    print("=" * 70)
    print(f"{'Model':<15} {'Exact Match':<15} {'Semantic':<15} {'LLM Judge':<15}")
    print("-" * 70)
    for model_name, data in results.items():
        row = f"{model_name:<15}"
        for method in methods:
            if method in data["summary"]:
                s = data["summary"][method]
                row += f"{s['passed']}/{s['total']} ({s['accuracy']:.1%})  "
            else:
                row += f"{'N/A':<15}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
