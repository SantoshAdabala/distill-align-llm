"""
SFT Scaling Experiment — Ablation over data size and epochs.

Runs a matrix of SFT experiments:
  - Data sizes: 875, 2500, 5000, 10000
  - Epochs: 1, 3, 5

Then evaluates each checkpoint on the factuality benchmark.

USAGE:
    # Run a single configuration
    python scripts/run_sft_scaling.py --config configs/local_small.yaml \
        --num-examples 2500 --epochs 3

    # Dry run (show what would be trained)
    python scripts/run_sft_scaling.py --config configs/local_small.yaml \
        --num-examples 5000 --epochs 5 --dry-run

    # Run full matrix (use with nohup on RunPod)
    python scripts/run_sft_scaling.py --config configs/local_small.yaml --run-all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sft_scaling")

# Experiment matrix
SCALING_MATRIX = [
    {"num_examples": 875, "epochs": 1},
    {"num_examples": 875, "epochs": 3},
    {"num_examples": 875, "epochs": 5},
    {"num_examples": 2500, "epochs": 1},
    {"num_examples": 2500, "epochs": 3},
    {"num_examples": 5000, "epochs": 1},
    {"num_examples": 5000, "epochs": 3},
    {"num_examples": 10000, "epochs": 1},
]


def load_sft_data(num_examples: int):
    """Load SFT training data, scaling to requested size.

    Strategy:
    - Always include all 875 technical instructions
    - Fill remaining with OpenHermes-2.5 examples
    - Add 15 uncertainty examples
    """
    from datasets import load_dataset as hf_load_dataset

    # Load technical instructions (always include all)
    tech_data = []
    tech_path = Path("data/technical_instructions.jsonl")
    if tech_path.exists():
        with open(tech_path) as f:
            for line in f:
                tech_data.append(json.loads(line))
    logger.info(f"Technical instructions: {len(tech_data)}")

    # Load uncertainty examples
    unc_data = []
    unc_path = Path("data/uncertainty_examples.jsonl")
    if unc_path.exists():
        with open(unc_path) as f:
            for line in f:
                unc_data.append(json.loads(line))
    logger.info(f"Uncertainty examples: {len(unc_data)}")

    # Calculate how many OpenHermes examples we need
    fixed_count = len(tech_data) + len(unc_data)
    openhermes_needed = max(0, num_examples - fixed_count)

    # Load OpenHermes
    if openhermes_needed > 0:
        logger.info(f"Loading {openhermes_needed} examples from OpenHermes-2.5...")
        oh_dataset = hf_load_dataset(
            "teknium/OpenHermes-2.5",
            split="train",
        )
        # Sample
        oh_dataset = oh_dataset.shuffle(seed=42).select(range(min(openhermes_needed, len(oh_dataset))))

        # Convert to messages format
        oh_data = []
        for item in oh_dataset:
            conversations = item.get("conversations", [])
            if len(conversations) >= 2:
                messages = []
                for conv in conversations:
                    role = "user" if conv["from"] == "human" else "assistant"
                    messages.append({"role": role, "content": conv["value"]})
                oh_data.append({"messages": messages})
        logger.info(f"OpenHermes examples loaded: {len(oh_data)}")
    else:
        oh_data = []

    # Combine all data
    all_data = tech_data + unc_data + oh_data
    logger.info(f"Total SFT data: {len(all_data)} (target: {num_examples})")

    return all_data


def run_single_experiment(config_path: str, num_examples: int, epochs: int, dry_run: bool = False):
    """Run a single SFT scaling experiment."""
    from distill_align.config import ConfigManager

    config = ConfigManager.load_config(config_path)
    experiment_name = f"sft_{num_examples}ex_{epochs}ep"
    output_dir = f"./outputs/scaling/{experiment_name}"

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"  Examples: {num_examples}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # Override config
    config.sft.num_epochs = epochs
    config.sft.output_dir = output_dir

    # Load data
    all_data = load_sft_data(num_examples)

    if dry_run:
        logger.info(f"✅ Dry run — would train on {len(all_data)} examples for {epochs} epochs")
        logger.info(f"   Estimated steps: {len(all_data) * epochs // (config.sft.batch_size * config.sft.gradient_accumulation_steps)}")
        return None

    # Load model
    import platform
    import torch
    from distill_align.models.loader import ModelLoader

    if platform.processor() == "arm" and not torch.cuda.is_available():
        from distill_align.config.models import QuantizationMode
        config.model.quantization.mode = QuantizationMode.NONE

    loader = ModelLoader()
    model, tokenizer = loader.load_model(config.model)

    # Convert data to HF Dataset
    from datasets import Dataset as HFDataset

    # Format for SFT
    def format_messages(item):
        messages = item.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    dataset = HFDataset.from_list(all_data)
    dataset = dataset.map(format_messages)

    # Split train/eval (95/5)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Train
    from distill_align.training.sft import SFTTrainerWrapper

    trainer = SFTTrainerWrapper()
    result = trainer.train(
        model=model,
        tokenizer=tokenizer,
        dataset={"train": train_dataset, "validation": eval_dataset},
        config=config.sft,
    )

    logger.info(f"Training complete: {result}")
    logger.info(f"  Final loss: {result.final_loss:.4f}")
    logger.info(f"  Adapter saved: {result.adapter_path}")

    # Save experiment metadata
    metadata = {
        "experiment": experiment_name,
        "num_examples": num_examples,
        "epochs": epochs,
        "final_loss": result.final_loss,
        "total_steps": result.total_steps,
        "training_time_seconds": result.training_time_seconds,
        "adapter_path": str(result.adapter_path),
    }
    meta_path = Path(output_dir) / "experiment_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="SFT Scaling Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num-examples", type=int, default=875, help="Number of SFT examples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without training")
    parser.add_argument("--run-all", action="store_true", help="Run full scaling matrix")
    args = parser.parse_args()

    if args.run_all:
        logger.info("Running full scaling matrix:")
        for exp in SCALING_MATRIX:
            logger.info(f"  {exp['num_examples']} examples × {exp['epochs']} epochs")

        results = []
        for exp in SCALING_MATRIX:
            result = run_single_experiment(
                args.config, exp["num_examples"], exp["epochs"], args.dry_run
            )
            if result:
                results.append(result)

        # Save all results
        if results:
            with open("outputs/scaling/all_results.json", "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nAll results saved to outputs/scaling/all_results.json")
    else:
        run_single_experiment(args.config, args.num_examples, args.epochs, args.dry_run)


if __name__ == "__main__":
    main()
