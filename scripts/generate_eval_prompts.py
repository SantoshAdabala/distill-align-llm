"""
generate_eval_prompts.py
========================
Generates 500 high-quality factuality eval prompts for distill-align-llm.

Strategy:
  - Seeds: 51 existing prompts from eval_factuality_v1.jsonl
  - Generator: GPT-4o-mini (cheap, fast, good at structured output)
  - 6 question types per concept to break the definitional ceiling
  - 5 domain categories, 100 prompts each
  - Dedup + quality filter to get exactly 500 best prompts

Cost estimate: ~$1.50 with GPT-4o-mini

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/generate_eval_prompts.py \
      --seed_file data/eval_factuality_v1.jsonl \
      --output_file data/eval_factuality_v2.jsonl \
      --n_target 500
"""

import json
import os
import re
import time
import argparse
import hashlib
from collections import defaultdict
from openai import OpenAI

# ── Categories and target counts ─────────────────────────────────────────────

CATEGORIES = {
    "architecture_facts":     100,   # model internals, layers, dimensions
    "training_mechanics":     100,   # optimizers, loss, hyperparams
    "alignment_concepts":     100,   # RLHF, SFT, DPO, reward models
    "quantization_efficiency": 100,  # QLoRA, NF4, bitsandbytes, PEFT
    "empirical_reasoning":    100,   # apply knowledge, compare, explain why
}

# Map existing seed prompts to categories based on keywords
CATEGORY_KEYWORDS = {
    "architecture_facts":     ["RoPE", "GQA", "SwiGLU", "RMSNorm", "KV-cache",
                                "attention_mask", "lm_head", "causal attention",
                                "Flash Attention", "tensor parallelism", "device_map",
                                "multi-head attention", "feed-forward", "layer norm",
                                "positional encoding", "embedding dimension", "hidden size",
                                "attention head", "context window", "decoder", "encoder",
                                "residual connection", "softmax", "query", "key", "value",
                                "logits", "probabilities", "temperature", "sampling",
                                "greedy", "beam search", "tokenizer", "vocabulary",
                                "special token", "chat template", "causal language"],
    "training_mechanics":     ["gradient checkpointing", "warmup", "weight decay",
                                "AdamW", "gradient accumulation", "mixed precision",
                                "fp16", "bf16", "loss masking", "sequence packing",
                                "NEFTune", "prepare_model"],
    "alignment_concepts":     ["DPO", "RLHF", "GRPO", "SFT", "reward accuracy",
                                "reference model", "Bradley-Terry", "reward hacking",
                                "implicit reward", "online", "offline", "KL penalty"],
    "quantization_efficiency": ["LoRA", "QLoRA", "NF4", "PEFT", "BitsAndBytes",
                                 "double quantization", "AWQ", "GGUF", "adapter merging",
                                 "TRL", "SFTTrainer", "DPOTrainer"],
    "empirical_reasoning":    ["vLLM", "PagedAttention", "speculative decoding",
                                "continuous batching", "model merging", "perplexity"],
}

# ── Question type templates ───────────────────────────────────────────────────

QUESTION_TYPES = {
    "definition": {
        "description": "Direct definition or expansion of an acronym",
        "example": "What does NF4 stand for in QLoRA?",
        "difficulty": 1,
    },
    "mechanism": {
        "description": "How or why something works internally",
        "example": "Why does DPO not require a separate reward model during training?",
        "difficulty": 2,
    },
    "comparison": {
        "description": "Difference between two related concepts",
        "example": "How does bf16 differ from fp16 in terms of training stability for large models?",
        "difficulty": 2,
    },
    "application": {
        "description": "Given a scenario, what would you do or expect?",
        "example": "You have 24GB VRAM and want to fine-tune Llama-3.1-8B. Which quantization approach fits?",
        "difficulty": 3,
    },
    "causal": {
        "description": "What causes a specific effect, or what is the consequence of an action",
        "example": "Why does stacking LoRA adapters (instead of merging) hurt DPO training?",
        "difficulty": 3,
    },
    "numerical": {
        "description": "A specific number, threshold, or quantitative fact",
        "example": "What LoRA rank (r) did the distill-align-llm project use for fine-tuning?",
        "difficulty": 2,
    },
}

# ── Generation prompt ─────────────────────────────────────────────────────────

GENERATION_SYSTEM = """You are an expert ML engineer creating evaluation prompts to test whether
a language model has retained factual knowledge about LLM training, alignment, and inference.

Your prompts will be used to measure the Alignment-Factuality Gap (AFG): whether a model that
achieves high reward accuracy on preference data also retains domain-specific factual knowledge.

Rules for generating prompts:
1. Each prompt must have a single unambiguous correct answer
2. The answer must be verifiable — not opinion-based
3. The prompt must be specific enough that keyword matching can verify correctness
4. Avoid prompts that are too easy (any LLM knows this) or too obscure (not in training data)
5. Prefer prompts where Base Llama-3.1-8B would likely fail but a domain-tuned model should pass
6. Return ONLY valid JSON, no markdown, no preamble
"""

GENERATION_USER = """Generate {n_prompts} evaluation prompts about the concept: "{concept}"

Question type: {q_type}
Type description: {q_type_desc}
Example of this type: {q_type_example}
Category: {category}
Difficulty: {difficulty}/3

Each prompt must follow this exact JSON schema:
{{
  "prompt": "The question to ask the model",
  "reference_answer": "The complete correct answer (2-4 sentences)",
  "must_include": ["keyword1", "keyword2", "keyword3"],
  "must_not_include": ["wrong_term1", "wrong_term2"],
  "category": "{category}",
  "question_type": "{q_type}",
  "difficulty": {difficulty},
  "concept": "{concept}"
}}

Return a JSON array of {n_prompts} prompts. No markdown. No extra text. Just the JSON array.
"""

# ── Core extraction from existing prompts ────────────────────────────────────

def extract_concepts_from_seeds(seed_file: str) -> list[str]:
    """Pull unique concepts from existing 51 prompts."""
    concepts = []
    with open(seed_file) as f:
        for line in f:
            data = json.loads(line.strip())
            # Extract the subject from "What is X?" or "What does X stand for?"
            prompt = data["prompt"]
            for pattern in [
                r"What is (?:the )?(.+?)\??$",
                r"What does (.+?) (?:stand for|do|mean)",
                r"What is the (?:purpose|difference|role) of (.+?)\??$",
            ]:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    concept = match.group(1).strip().rstrip("?").strip()
                    if concept not in concepts:
                        concepts.append(concept)
                    break
    return concepts

def assign_category(concept: str) -> str:
    """Assign a concept to a category based on keyword matching."""
    concept_lower = concept.lower()
    scores = defaultdict(int)
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in concept_lower:
                scores[cat] += 2
    # Fallback heuristics
    if "dpo" in concept_lower or "rlhf" in concept_lower or "reward" in concept_lower:
        scores["alignment_concepts"] += 3
    if "lora" in concept_lower or "quant" in concept_lower or "nf4" in concept_lower:
        scores["quantization_efficiency"] += 3
    if "gradient" in concept_lower or "loss" in concept_lower or "train" in concept_lower:
        scores["training_mechanics"] += 3
    if scores:
        return max(scores, key=scores.get)
    return "empirical_reasoning"  # default

# ── Additional concepts not in seeds ─────────────────────────────────────────

EXTRA_CONCEPTS = [
    # Architecture facts (target: 30+ concepts for ~120 candidates)
    "multi-head attention", "feed-forward network in transformers", "layer normalization",
    "positional encoding", "tokenizer vocabulary size", "embedding dimension",
    "context window length", "transformer decoder vs encoder",
    "residual connection in transformers", "softmax in attention",
    "query key value projections", "number of attention heads in Llama-3.1-8B",
    "hidden size in Llama-3.1-8B", "number of transformer layers",
    "tokenizer byte-pair encoding", "special tokens in chat templates",
    "system prompt in instruction tuning", "chat template format",
    "causal language model vs masked language model",
    "logits vs probabilities in language model output",
    "temperature in sampling", "top-p nucleus sampling",
    "greedy decoding", "beam search decoding",
    # Training mechanics (target: 30+ concepts)
    "cosine learning rate schedule", "early stopping", "overfitting in SFT",
    "catastrophic forgetting", "epoch vs steps", "per-device batch size",
    "effective batch size", "gradient clipping",
    "AdamW optimizer betas", "weight initialization in transformers",
    "training loss vs validation loss", "learning rate warmup ratio",
    "checkpoint saving during training", "evaluation steps during training",
    "packing dataset sequences for efficiency", "data collator in SFT",
    "chat template tokenization", "instruction following format",
    "cross-entropy loss for next token prediction",
    "token prediction loss masking on prompt",
    # Alignment concepts (target: 30+ concepts)
    "chosen vs rejected in DPO dataset", "preference data collection",
    "reward model training", "PPO in RLHF", "SimPO vs DPO",
    "Alignment-Factuality Gap", "domain factuality evaluation",
    "SFT before DPO pipeline", "merged-SFT strategy",
    "DPO beta hyperparameter tuning", "DPO learning rate",
    "KL divergence in preference optimization",
    "reward overoptimization", "goodhart's law in RLHF",
    "preference dataset quality", "UltraFeedback dataset",
    "OpenHermes dataset", "domain-specific SFT data",
    # Quantization and efficiency (target: 30+ concepts)
    "LoRA alpha parameter", "LoRA target modules", "4-bit NF4 vs 8-bit int8",
    "PEFT adapter saving", "merged vs stacked LoRA adapters",
    "bitsandbytes load_in_4bit", "LoRA rank r=16",
    "LoRA rank vs alpha ratio", "which modules to apply LoRA to",
    "inference with merged LoRA adapter", "PEFT get_peft_model",
    "gradient checkpointing with LoRA", "bf16 compute dtype in QLoRA",
    "double quantization memory savings", "NF4 information theory basis",
    # Empirical reasoning (target: 30+ concepts)
    "perplexity as eval metric", "epoch sensitivity in domain SFT",
    "data dilution in fine-tuning", "reward accuracy vs factuality",
    "temperature=0 for deterministic eval", "keyword matching eval limitations",
    "LLM-as-judge evaluation", "factuality benchmark design",
    "scaling laws for fine-tuning", "sample efficiency in SFT",
    "domain adaptation vs general fine-tuning",
    "technical domain knowledge retention after alignment",
]

# ── Generation logic ──────────────────────────────────────────────────────────

def generate_prompts_for_concept(
    client: OpenAI,
    concept: str,
    category: str,
    q_type: str,
    n_prompts: int = 2,
) -> list[dict]:
    """Call GPT-4o-mini to generate prompts for one concept + question type."""
    qt = QUESTION_TYPES[q_type]
    user_msg = GENERATION_USER.format(
        n_prompts=n_prompts,
        concept=concept,
        q_type=q_type,
        q_type_desc=qt["description"],
        q_type_example=qt["example"],
        category=category,
        difficulty=qt["difficulty"],
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        # GPT sometimes wraps in {"prompts": [...]} or returns array directly
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        # Find the first list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    except Exception as e:
        print(f"  [WARN] Failed for {concept}/{q_type}: {e}")
        return []

def deduplicate(prompts: list[dict]) -> list[dict]:
    """Remove near-duplicate prompts using prompt text hash."""
    seen = set()
    unique = []
    for p in prompts:
        key = hashlib.md5(p["prompt"].lower().strip().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique

def quality_filter(prompts: list[dict]) -> list[dict]:
    """Remove prompts missing required fields or with weak must_include lists."""
    good = []
    required_fields = {"prompt", "reference_answer", "must_include", "category", "question_type"}
    for p in prompts:
        if not required_fields.issubset(p.keys()):
            continue
        if len(p["must_include"]) < 2:
            continue
        if len(p["prompt"].strip()) < 15:
            continue
        if len(p["reference_answer"].strip()) < 20:
            continue
        good.append(p)
    return good

def balance_categories(prompts: list[dict], targets: dict[str, int]) -> list[dict]:
    """Select prompts to hit target counts per category."""
    by_cat = defaultdict(list)
    for p in prompts:
        by_cat[p.get("category", "empirical_reasoning")].append(p)

    selected = []
    for cat, target in targets.items():
        pool = by_cat[cat]
        # Prefer harder prompts (difficulty 2-3) in the selection
        pool.sort(key=lambda x: x.get("difficulty", 1), reverse=True)
        selected.extend(pool[:target])
        if len(pool) < target:
            print(f"  [WARN] Category '{cat}': only {len(pool)}/{target} prompts available")

    return selected

def convert_seeds_to_v2_format(seed_file: str) -> list[dict]:
    """Convert existing 51 prompts to the v2 schema."""
    converted = []
    with open(seed_file) as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            converted.append({
                "id": f"seed_{i:03d}",
                "prompt": data["prompt"],
                "reference_answer": "",   # will be empty for seeds — judge uses must_include
                "must_include": data.get("must_include", []),
                "must_not_include": data.get("must_not_include", []),
                "category": assign_category(data["prompt"]),
                "question_type": "definition",
                "difficulty": 1,
                "concept": data["prompt"].replace("What is ", "").replace("What does ", "").rstrip("?"),
                "source": "seed_v1",
            })
    return converted

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_file", default="data/eval_factuality_v1.jsonl")
    parser.add_argument("--output_file", default="data/eval_factuality_v2.jsonl")
    parser.add_argument("--n_target", type=int, default=500)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without calling API")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key) if not args.dry_run else None

    print("=" * 60)
    print("BSFT Eval Prompt Generator")
    print(f"  Target: {args.n_target} prompts across {len(CATEGORIES)} categories")
    print("=" * 60)

    # Step 1: Convert seeds to v2 format
    print("\n[1/4] Converting 51 seed prompts to v2 format...")
    all_prompts = convert_seeds_to_v2_format(args.seed_file)
    print(f"  Converted: {len(all_prompts)} seed prompts")

    # Step 2: Build full concept list
    seed_concepts = extract_concepts_from_seeds(args.seed_file)
    all_concepts = list(set(seed_concepts + EXTRA_CONCEPTS))
    print(f"\n[2/4] Concept pool: {len(all_concepts)} unique concepts")

    if args.dry_run:
        print("\n[DRY RUN] Would generate prompts for:")
        for c in all_concepts[:10]:
            cat = assign_category(c)
            print(f"  {cat:30s} | {c}")
        print(f"  ... and {len(all_concepts)-10} more")
        print("\nEstimated API calls:", len(all_concepts) * 3)
        print("Estimated cost: ~$1.20-1.80")
        return

    # Step 3: Generate new prompts
    print(f"\n[3/4] Generating prompts via GPT-4o-mini...")
    print("  This takes ~3-5 minutes. Cost: ~$1.50")

    # For each concept, generate 2-3 question types to get variety
    # We aim for ~800 raw candidates, then filter down to 500
    q_type_rotation = [
        "mechanism", "comparison", "application",  # harder types first
        "causal", "numerical", "definition",
    ]

    # Thin categories get 3 prompts per call instead of 2
    thin_categories = {"quantization_efficiency", "training_mechanics"}

    total_generated = 0
    for i, concept in enumerate(all_concepts):
        category = assign_category(concept)
        n_per_call = 3 if category in thin_categories else 2
        # Pick 2 question types per concept to avoid over-generating
        qt1 = q_type_rotation[i % len(q_type_rotation)]
        qt2 = q_type_rotation[(i + 2) % len(q_type_rotation)]

        for q_type in [qt1, qt2]:
            new_prompts = generate_prompts_for_concept(
                client, concept, category, q_type, n_prompts=n_per_call
            )
            for j, p in enumerate(new_prompts):
                p["id"] = f"gen_{i:03d}_{q_type[:3]}_{j}"
                p["source"] = "gpt4o_mini"
                p.setdefault("must_not_include", [])
                p.setdefault("difficulty", QUESTION_TYPES[q_type]["difficulty"])
            all_prompts.extend(new_prompts)
            total_generated += len(new_prompts)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(all_concepts)} concepts "
                  f"({total_generated} prompts generated so far)")
        time.sleep(0.3)  # gentle rate limiting

    print(f"  Total raw candidates: {len(all_prompts)}")

    # Step 4: Filter and balance
    print("\n[4/4] Filtering and balancing...")
    all_prompts = quality_filter(all_prompts)
    print(f"  After quality filter: {len(all_prompts)}")
    all_prompts = deduplicate(all_prompts)
    print(f"  After deduplication: {len(all_prompts)}")
    final_prompts = balance_categories(all_prompts, CATEGORIES)
    print(f"  After category balancing: {len(final_prompts)}")

    # Assign final IDs and write
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        for i, p in enumerate(final_prompts):
            p["id"] = f"eval_v2_{i:04d}"
            f.write(json.dumps(p) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! Written {len(final_prompts)} prompts to {args.output_file}")

    # Summary table
    cat_counts = defaultdict(int)
    type_counts = defaultdict(int)
    diff_counts = defaultdict(int)
    for p in final_prompts:
        cat_counts[p.get("category", "unknown")] += 1
        type_counts[p.get("question_type", "unknown")] += 1
        diff_counts[p.get("difficulty", 0)] += 1

    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items()):
        bar = "█" * (count // 5)
        print(f"  {cat:30s} {count:3d} {bar}")

    print("\nQuestion type breakdown:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt:20s} {count:3d}")

    print("\nDifficulty breakdown:")
    for diff in sorted(diff_counts.keys()):
        print(f"  Level {diff}: {diff_counts[diff]}")

if __name__ == "__main__":
    main()
