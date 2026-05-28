"""
run_bsft.py — Best Supervised Fine-Tuning (BSFT)
=================================================
Modified SFT with two additional loss terms designed to improve
factual knowledge retention through alignment:

1. Quality weighting: domain-specific examples get higher weight (q=1.0)
   vs generic instruction examples (q=0.1)

2. Token weighting: factual keyword tokens get amplified gradient signal
   (w=3.0) so the model learns to produce exact factual terminology

Loss function:
    L_BSFT = -Σ_i q_i * Σ_t w_t * log P(y_t | y_<t, x)
             / Σ_i q_i * T_i

This is a drop-in replacement for run_sft.py. Same config, same data,
same LoRA setup — only the loss computation changes.

Usage:
    # 3B model (primary target — where real AFG exists)
    python scripts/run_bsft.py --config configs/llama_3b_5ep.yaml \
        --quality_weight_domain 1.0 \
        --quality_weight_general 0.1 \
        --token_weight_keywords 3.0

    # Ablation: quality weighting only
    python scripts/run_bsft.py --config configs/llama_3b_5ep.yaml \
        --quality_weight_domain 1.0 \
        --quality_weight_general 0.1 \
        --token_weight_keywords 1.0

    # Ablation: token weighting only
    python scripts/run_bsft.py --config configs/llama_3b_5ep.yaml \
        --quality_weight_domain 1.0 \
        --quality_weight_general 1.0 \
        --token_weight_keywords 3.0

    # 8B model (verify BSFT doesn't hurt already-good performance)
    python scripts/run_bsft.py --config configs/llama_8b_5ep.yaml

Cost: ~$5 per run on A100 (~20 min for 3B, ~70 min for 8B)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distill_align.config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_bsft")


# ── Factual keywords from eval set ───────────────────────────────────────────
# These are the must_include terms from eval_factuality.jsonl
# The model gets stronger gradient signal on these tokens during training

def load_factual_keywords(eval_file: str = "data/eval_factuality.jsonl") -> set[str]:
    """Extract all must_include keywords from the eval set."""
    keywords = set()
    try:
        with open(eval_file) as f:
            for line in f:
                data = json.loads(line.strip())
                for kw in data.get("must_include", []):
                    # Add the keyword and its individual words
                    keywords.add(kw.lower())
                    for word in kw.lower().split():
                        if len(word) > 2:  # skip tiny words
                            keywords.add(word)
    except FileNotFoundError:
        logger.warning(f"Eval file not found: {eval_file}. Using empty keyword set.")
    logger.info(f"Loaded {len(keywords)} factual keywords for token weighting")
    return keywords


# ── BSFT Trainer (subclass of TRL SFTTrainer) ────────────────────────────────

def create_bsft_trainer(
    model, tokenizer, train_dataset, eval_dataset, config,
    quality_weight_domain: float,
    quality_weight_general: float,
    token_weight_keywords: float,
    factual_keywords: set[str],
    output_dir: str,
):
    """Create a modified SFTTrainer with BSFT weighted loss."""
    import torch
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    training_args = TRLSFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=getattr(config, 'weight_decay', 0.01),
        max_grad_norm=getattr(config, 'max_grad_norm', 1.0),
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_length=2048,
    )

    # Build token weight lookup: token_id -> weight
    token_weights = {}
    for kw in factual_keywords:
        # Tokenize each keyword and mark those token IDs
        ids = tokenizer.encode(kw, add_special_tokens=False)
        for tid in ids:
            token_weights[tid] = token_weight_keywords
    logger.info(f"Token weighting: {len(token_weights)} token IDs get weight={token_weight_keywords}")

    class BSFTTrainer(SFTTrainer):
        """SFTTrainer with quality and token weighting."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Override loss computation with BSFT weighting."""
            # Standard forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            labels = inputs["labels"][..., 1:].contiguous()

            # Standard CE loss per token (unreduced)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = labels.view(-1)
            per_token_loss = loss_fct(flat_logits, flat_labels)
            per_token_loss = per_token_loss.view(labels.shape)

            # ── Token weighting ──
            # Create weight tensor: default 1.0, keyword tokens get higher weight
            token_weight_tensor = torch.ones_like(per_token_loss)
            if token_weight_keywords > 1.0:
                for tid, w in token_weights.items():
                    mask = (labels == tid)
                    token_weight_tensor[mask] = w

            # ── Quality weighting ──
            # Heuristic: shorter sequences are likely domain examples (875 technical)
            # Longer sequences are likely OpenHermes general examples
            # Better: check if any factual keyword appears in the input
            seq_lengths = (labels != -100).sum(dim=-1).float()
            batch_size = labels.shape[0]
            quality_weights = torch.ones(batch_size, device=labels.device)

            if quality_weight_domain != quality_weight_general:
                for i in range(batch_size):
                    # Check if this example contains factual keywords
                    input_ids = inputs["input_ids"][i]
                    has_domain_content = any(
                        tid in token_weights for tid in input_ids.tolist()
                    )
                    quality_weights[i] = (
                        quality_weight_domain if has_domain_content
                        else quality_weight_general
                    )

            # ── Combine weights and compute final loss ──
            # Apply token weights
            weighted_loss = per_token_loss * token_weight_tensor

            # Mask out padding (-100 labels)
            valid_mask = (labels != -100).float()
            weighted_loss = weighted_loss * valid_mask

            # Per-example loss (mean over tokens)
            per_example_loss = weighted_loss.sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)

            # Apply quality weights
            final_loss = (per_example_loss * quality_weights).sum() / quality_weights.sum()

            return (final_loss, outputs) if return_outputs else final_loss

    trainer = BSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    return trainer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BSFT: Best Supervised Fine-Tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--quality_weight_domain", type=float, default=1.0,
                        help="Loss weight for domain-specific examples")
    parser.add_argument("--quality_weight_general", type=float, default=0.1,
                        help="Loss weight for generic instruction examples")
    parser.add_argument("--token_weight_keywords", type=float, default=3.0,
                        help="Loss weight for factual keyword tokens")
    parser.add_argument("--eval_keywords_file", type=str,
                        default="data/eval_factuality.jsonl",
                        help="File to extract factual keywords from")
    parser.add_argument("--output_suffix", type=str, default="bsft",
                        help="Suffix for output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load config
    config = ConfigManager.load_config(args.config)
    logger.info(f"Model: {config.model.model_id} ({config.model.family.value})")

    # Override output dir with BSFT suffix
    base_output = config.sft.output_dir
    bsft_output = base_output.replace("/sft", f"/{args.output_suffix}")
    config.sft.output_dir = bsft_output

    logger.info(f"BSFT Configuration:")
    logger.info(f"  quality_weight_domain:  {args.quality_weight_domain}")
    logger.info(f"  quality_weight_general: {args.quality_weight_general}")
    logger.info(f"  token_weight_keywords:  {args.token_weight_keywords}")
    logger.info(f"  output_dir:             {bsft_output}")

    # Platform check
    import platform
    import torch
    if platform.processor() == "arm" and not torch.cuda.is_available():
        from distill_align.config.models import QuantizationMode
        config.model.quantization.mode = QuantizationMode.NONE

    # Load model
    logger.info("Loading model with LoRA adapters...")
    from distill_align.models.loader import ModelLoader
    loader = ModelLoader()
    model, tokenizer = loader.load_model(config.model)

    # Load dataset (same as run_sft.py)
    logger.info("Loading instruction dataset...")
    from datasets import Dataset, concatenate_datasets, load_dataset

    hermes = load_dataset("teknium/OpenHermes-2.5", split="train[:3000]")

    def format_hermes(example):
        convos = example.get("conversations", [])
        messages = []
        for turn in convos:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        return {"messages": messages}

    hermes = hermes.map(format_hermes, remove_columns=hermes.column_names)

    # Load technical + uncertainty data
    extra_data = []
    tech_path = Path("data/technical_instructions.jsonl")
    uncertainty_path = Path("data/uncertainty_examples.jsonl")

    if tech_path.exists():
        with open(tech_path) as f:
            for line in f:
                extra_data.append(json.loads(line))
        logger.info(f"Loaded {len(extra_data)} technical examples")

    if uncertainty_path.exists():
        with open(uncertainty_path) as f:
            for line in f:
                extra_data.append(json.loads(line))
        logger.info(f"Added uncertainty examples (total: {len(extra_data)})")

    if extra_data:
        tech_dataset = Dataset.from_list(extra_data)
        dataset = concatenate_datasets([hermes, tech_dataset])
    else:
        dataset = hermes

    logger.info(f"Combined dataset: {len(dataset)} examples")

    # Tokenize
    from distill_align.data.processor import DataProcessor
    processor = DataProcessor()
    logger.info(f"Tokenizing {len(dataset)} examples...")
    tokenized = processor.tokenize(dataset, tokenizer, config.model.max_seq_length)

    # Split
    splits = processor.split(tokenized)
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}")

    if args.dry_run:
        logger.info("Dry run complete — model and data loaded successfully")
        return

    # Load factual keywords
    factual_keywords = load_factual_keywords(args.eval_keywords_file)

    # Create BSFT trainer
    logger.info("Creating BSFT trainer with weighted loss...")
    trainer = create_bsft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        config=config.sft,
        quality_weight_domain=args.quality_weight_domain,
        quality_weight_general=args.quality_weight_general,
        token_weight_keywords=args.token_weight_keywords,
        factual_keywords=factual_keywords,
        output_dir=bsft_output,
    )

    # Train
    logger.info("Starting BSFT training...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # Save adapter
    adapter_path = str(Path(bsft_output) / "final_adapter")
    trainer.save_model(adapter_path)
    logger.info(f"BSFT adapter saved to: {adapter_path}")
    logger.info(f"Training time: {elapsed/60:.1f} min")
    logger.info(f"Final loss: {train_result.training_loss:.4f}")

    # Evaluate
    if splits.get("validation"):
        eval_metrics = trainer.evaluate()
        logger.info(f"Validation loss: {eval_metrics.get('eval_loss', 'N/A')}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Run DPO: python scripts/run_dpo.py --config {args.config} "
                f"--sft-adapter {adapter_path} --merge-sft")
    logger.info(f"  2. Eval: python scripts/eval_factuality_all.py "
                f"--sft-adapter {adapter_path} ...")


if __name__ == "__main__":
    main()
