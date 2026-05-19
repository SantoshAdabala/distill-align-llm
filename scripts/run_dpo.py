"""
Run DPO Alignment — Entry Point Script.

USAGE:
    # Local mode (RunPod RTX 3090)
    python scripts/run_dpo.py --config configs/local_small.yaml --sft-adapter ./outputs/sft/final_adapter

    # Dry run (load model + data, don't train)
    python scripts/run_dpo.py --config configs/local_small.yaml --sft-adapter ./outputs/sft/final_adapter --dry-run

    # Custom sample size
    python scripts/run_dpo.py --config configs/local_small.yaml --sft-adapter ./outputs/sft/final_adapter --num-samples 10000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distill_align.config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_dpo")


def main():
    parser = argparse.ArgumentParser(description="Run DPO alignment training (v2 — improved)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--sft-adapter", type=str, required=True, help="Path to SFT adapter weights")
    parser.add_argument("--merge-sft", action="store_true", help="Merge SFT adapter into base before DPO (Experiment 3)")
    parser.add_argument("--cloud", action="store_true", help="Run on cloud GPU (RunPod)")
    parser.add_argument("--dry-run", action="store_true", help="Load but don't train")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of preference pairs")
    parser.add_argument("--eval-samples", type=int, default=500, help="Number of eval samples")
    args = parser.parse_args()

    # Load config
    config = ConfigManager.load_config(args.config)
    logger.info(f"Model: {config.model.model_id}, DPO beta: {config.dpo.beta}, LR: {config.dpo.learning_rate}")

    # Platform check
    import platform

    import torch

    if platform.processor() == "arm" and not torch.cuda.is_available():
        logger.info("Apple Silicon detected — using MPS, disabling quantization")
        from distill_align.config.models import QuantizationMode
        config.model.quantization.mode = QuantizationMode.NONE

    # Load base model with LoRA
    from distill_align.models.loader import ModelLoader

    loader = ModelLoader()
    model, tokenizer = loader.load_model(config.model)

    # Load the trained SFT adapter weights
    logger.info(f"Loading SFT adapter from: {args.sft_adapter}")
    if args.merge_sft:
        # Experiment 3: Merge SFT into base weights before DPO
        # Must load in bf16 (not quantized) to merge, then re-apply quantization
        logger.info("EXPERIMENT 3: Merging SFT adapter into base model before DPO...")
        logger.info("Loading base model in bf16 for merge (cannot merge into quantized model)...")

        import torch as _torch
        from peft import PeftModel as _PeftModel
        from transformers import AutoModelForCausalLM as _AutoModel

        # Load base in bf16 (no quantization) for clean merge
        base_for_merge = _AutoModel.from_pretrained(
            config.model.model_id,
            torch_dtype=_torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Load SFT adapter
        merged = _PeftModel.from_pretrained(base_for_merge, args.sft_adapter)
        # Merge adapter into base weights
        merged = merged.merge_and_unload()
        logger.info("SFT adapter merged successfully.")

        # Save merged model temporarily
        merge_path = "./outputs/sft_merged"
        merged.save_pretrained(merge_path)
        tokenizer.save_pretrained(merge_path)
        del merged, base_for_merge, model
        import gc
        _torch.cuda.empty_cache()
        gc.collect()

        # Reload merged model with quantization + fresh LoRA for DPO
        logger.info("Reloading merged model with QLoRA for DPO training...")
        from distill_align.models.loader import ModelLoader as _Loader
        # Create a config pointing to the merged model
        merged_config = config.model.model_copy()
        merged_config.model_id = merge_path
        loader2 = _Loader()
        model, tokenizer = loader2.load_model(merged_config)
        logger.info("Merged model loaded with fresh LoRA. DPO will train on merged base.")
    else:
        # Standard: Load SFT adapter, train DPO adapter on top
        model.load_adapter(args.sft_adapter, adapter_name="sft")
        model.set_adapter("sft")
        logger.info("SFT adapter loaded (stacked adapter approach)")

    # ─── Load preference dataset ───
    logger.info("Loading preference dataset: argilla/ultrafeedback-binarized-preferences-cleaned")
    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        split="train",
    )
    logger.info(f"Full dataset: {len(dataset)} preference pairs")
    logger.info(f"Columns: {dataset.column_names}")

    # Select train and eval subsets
    total_needed = args.num_samples + args.eval_samples
    if len(dataset) < total_needed:
        logger.warning(f"Dataset has {len(dataset)} samples, need {total_needed}. Using all available.")
        total_needed = len(dataset)
        args.num_samples = int(total_needed * 0.9)
        args.eval_samples = total_needed - args.num_samples

    train_dataset = dataset.select(range(args.num_samples))
    eval_dataset = dataset.select(range(args.num_samples, args.num_samples + args.eval_samples))
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ─── Format dataset for DPO ───
    def format_for_dpo(example):
        """Convert UltraFeedback cleaned format to DPO format."""
        chosen_msgs = example.get("chosen", [])
        rejected_msgs = example.get("rejected", [])

        if isinstance(chosen_msgs, list) and len(chosen_msgs) >= 2:
            prompt = chosen_msgs[0]["content"] if isinstance(chosen_msgs[0], dict) else str(chosen_msgs[0])
            chosen = chosen_msgs[1]["content"] if isinstance(chosen_msgs[1], dict) else str(chosen_msgs[1])
        else:
            prompt = str(example.get("prompt", example.get("instruction", "")))
            chosen = str(chosen_msgs)

        if isinstance(rejected_msgs, list) and len(rejected_msgs) >= 2:
            rejected = rejected_msgs[1]["content"] if isinstance(rejected_msgs[1], dict) else str(rejected_msgs[1])
        else:
            rejected = str(rejected_msgs)

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    logger.info("Formatting dataset for DPO...")
    train_dataset = train_dataset.map(format_for_dpo, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_for_dpo, remove_columns=eval_dataset.column_names)

    # Add factual DPO pairs — upsample to ~20% of training data (reduces hallucination)
    factual_path = Path("data/factual_dpo_pairs.jsonl")
    if factual_path.exists():
        factual_pairs = []
        with open(factual_path) as f:
            for line in f:
                factual_pairs.append(json.loads(line))
        if factual_pairs:
            # Upsample factual pairs to reach ~20% of total training data
            # With 5000 generic pairs, we need ~1250 factual pairs (20% of 6250)
            import random
            target_factual_count = max(len(factual_pairs), int(len(train_dataset) * 0.25))
            upsampled = []
            while len(upsampled) < target_factual_count:
                upsampled.extend(factual_pairs)
            upsampled = upsampled[:target_factual_count]
            random.shuffle(upsampled)

            from datasets import Dataset as HFDataset
            from datasets import concatenate_datasets
            factual_ds = HFDataset.from_list(upsampled)
            train_dataset = concatenate_datasets([train_dataset, factual_ds])
            logger.info(f"Added {len(upsampled)} factual DPO pairs (upsampled from {len(factual_pairs)}, total: {len(train_dataset)})")

    logger.info(f"Formatted columns: {train_dataset.column_names}")

    # Filter out any empty entries
    def is_valid(example):
        return (
            len(example["prompt"].strip()) > 0
            and len(example["chosen"].strip()) > 0
            and len(example["rejected"].strip()) > 0
        )

    train_dataset = train_dataset.filter(is_valid)
    eval_dataset = eval_dataset.filter(is_valid)
    logger.info(f"After filtering — Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    if args.dry_run:
        logger.info("✅ Dry run complete — model and preference data loaded")
        logger.info(f"   Model: {config.model.model_id}")
        logger.info(f"   DPO LR: {config.dpo.learning_rate}")
        logger.info(f"   DPO beta: {config.dpo.beta}")
        logger.info(f"   Train samples: {len(train_dataset)}")
        logger.info(f"   Eval samples: {len(eval_dataset)}")
        return

    # ─── Train DPO ───
    logger.info("Starting DPO training (v2 — improved)...")
    logger.info(f"  learning_rate: {config.dpo.learning_rate}")
    logger.info(f"  beta: {config.dpo.beta}")
    logger.info(f"  batch_size: {config.dpo.batch_size}")
    logger.info(f"  gradient_accumulation: {config.dpo.gradient_accumulation_steps}")
    logger.info(f"  eval_steps: {config.dpo.eval_steps}")

    from distill_align.training.dpo import DPOTrainerWrapper

    trainer = DPOTrainerWrapper()
    result = trainer.train(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset={"train": train_dataset, "validation": eval_dataset},
        config=config.dpo,
    )

    logger.info(f"DPO complete: {result}")
    logger.info(f"Aligned adapter saved to: {result.adapter_path}")
    logger.info(f"Reward accuracy: {result.reward_accuracy:.2%}")
    logger.info(f"Reward margin: {result.reward_margin:.4f}")

    # ─── Print final evaluation metrics ───
    logger.info("=" * 60)
    logger.info("FINAL DPO METRICS:")
    logger.info(f"  Loss: {result.final_loss:.4f}")
    logger.info(f"  Reward Accuracy: {result.reward_accuracy:.2%}")
    logger.info(f"  Reward Margin: {result.reward_margin:.4f}")
    logger.info(f"  Total Steps: {result.total_steps}")
    logger.info(f"  Training Time: {result.training_time_seconds / 60:.1f} min")
    logger.info(f"  Low Accuracy Warnings: {result.low_accuracy_warnings}")
    logger.info("=" * 60)

    if result.reward_accuracy >= 0.65:
        logger.info("🎯 Strong alignment — reward accuracy ≥ 65%")
    elif result.reward_accuracy >= 0.60:
        logger.info("✅ Good alignment — reward accuracy ≥ 60%")
    elif result.reward_accuracy >= 0.55:
        logger.info("⚠️  Okay alignment — reward accuracy ≥ 55%, consider more data or lower LR")
    else:
        logger.warning("❌ Weak alignment — reward accuracy < 55%. Try: more data, lower LR, cleaner preferences")


if __name__ == "__main__":
    main()
