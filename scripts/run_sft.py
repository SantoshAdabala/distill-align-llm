"""
Run Supervised Fine-Tuning (SFT) — Entry Point Script.

USAGE:
    # Local mode (Mac M1/M2 — tiny model for testing)
    python scripts/run_sft.py --config configs/local_small.yaml

    # Cloud mode (used runpod.io for better GPU cost and efficiency)
    python scripts/run_sft.py --config configs/cloud_large.yaml --cloud
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distill_align.config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sft")


def main():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--cloud", action="store_true", help="Submit to RunPod cloud GPU"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Load model and data but don't train"
    )
    args = parser.parse_args()

    # ─── Load config ───
    logger.info(f"Loading config from: {args.config}")
    config = ConfigManager.load_config(args.config)
    logger.info(f"Model: {config.model.model_id} ({config.model.family.value})")
    logger.info(f"Quantization: {config.model.quantization.mode.value}")
    logger.info(f"LoRA rank: {config.model.lora.rank}")

    # ─── Check platform compatibility ───
    import platform

    import torch

    if platform.processor() == "arm" and not torch.cuda.is_available():
        logger.info("Detected Apple Silicon (M1/M2) — using MPS backend")
        if config.model.quantization.mode.value == "int4_nf4":
            logger.warning(
                "⚠️  4-bit quantization (bitsandbytes) is NOT supported on Mac.\n"
                "   Switching to fp32 for local testing.\n"
                "   Use RunPod with RTX A5000 for real training with quantization."
            )
            from distill_align.config.models import QuantizationMode
            config.model.quantization.mode = QuantizationMode.NONE

    # ─── Load model ───
    logger.info("Loading model with LoRA adapters...")
    from distill_align.models.loader import ModelLoader

    loader = ModelLoader()

    try:
        model, tokenizer = loader.load_model(config.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # ─── Load dataset ───
    logger.info("Loading instruction dataset...")
    from datasets import Dataset

    from distill_align.config.models import DatasetType
    from distill_align.data.processor import DataProcessor

    processor = DataProcessor()

    instruction_datasets = [
        ds for ds in config.datasets if ds.dataset_type == DatasetType.INSTRUCTION
    ]

    if not instruction_datasets:
        logger.info("No instruction dataset in config — using OpenHermes-2.5 + custom technical data")
        from datasets import concatenate_datasets, load_dataset

        # Load high-quality instruction data (OpenHermes-2.5 is much better than Alpaca)
        hermes = load_dataset("teknium/OpenHermes-2.5", split="train[:3000]")
        # Format to messages
        def format_hermes(example):
            convos = example.get("conversations", [])
            messages = []
            for turn in convos:
                role = "user" if turn["from"] == "human" else "assistant"
                messages.append({"role": role, "content": turn["value"]})
            return {"messages": messages}

        hermes = hermes.map(format_hermes, remove_columns=hermes.column_names)

        # Load custom technical dataset
        import json
        from pathlib import Path
        tech_path = Path("data/technical_instructions.jsonl")
        uncertainty_path = Path("data/uncertainty_examples.jsonl")

        extra_data = []
        if tech_path.exists():
            with open(tech_path) as f:
                for line in f:
                    extra_data.append(json.loads(line))
            logger.info(f"Loaded {len(extra_data)} technical examples")

        if uncertainty_path.exists():
            with open(uncertainty_path) as f:
                for line in f:
                    extra_data.append(json.loads(line))
            logger.info(f"Added uncertainty examples (total extra: {len(extra_data)})")

        if extra_data:
            tech_dataset = Dataset.from_list(extra_data)
            dataset = concatenate_datasets([hermes, tech_dataset])
        else:
            dataset = hermes

        logger.info(f"Combined SFT dataset: {len(dataset)} examples")
    else:
        ds_config = instruction_datasets[0]
        dataset = processor.load_dataset(ds_config)
        if hasattr(dataset, "keys"):
            dataset = dataset["train"]

    # Tokenize
    logger.info(f"Tokenizing {len(dataset)} examples...")
    tokenized = processor.tokenize(dataset, tokenizer, config.model.max_seq_length)

    # Split
    splits = processor.split(tokenized)
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}")

    if args.dry_run:
        logger.info("✅ Dry run complete — model and data loaded successfully")
        return

    # ─── Train ───
    if args.cloud:
        logger.info("Cloud training via RunPod — run this script directly on your RunPod pod.")
        logger.error("This script is designed to run directly on the GPU pod, not submit remotely.")
        sys.exit(1)
    else:
        logger.info("Starting local SFT training...")
        from distill_align.training.sft import SFTTrainerWrapper

        trainer = SFTTrainerWrapper()
        result = trainer.train(
            model=model,
            tokenizer=tokenizer,
            dataset=splits,
            config=config.sft,
        )

        logger.info(f"Training complete: {result}")
        logger.info(f"Adapter saved to: {result.adapter_path}")


if __name__ == "__main__":
    main()
