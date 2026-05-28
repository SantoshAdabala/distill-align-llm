"""
simpo.py — SimPO (Simple Preference Optimization) Training
==========================================================
Trains SimPO on Llama-3.1-8B as an alternative alignment method to DPO.

Key differences from DPO:
- No reference model needed (saves ~8GB VRAM)
- Length-normalized reward: avg log prob per token
- Target reward margin (simpo_gamma): chosen must beat rejected by γ

Uses TRL's CPOTrainer with loss_type="simpo", cpo_alpha=0.0.

Usage:
    python scripts/simpo.py \
        --sft_merged  ./outputs/sft_merged \
        --output_dir  ./outputs/simpo \
        --dataset     trl-lib/ultrafeedback_binarized

Cost: ~$10 on A100 SXM
Time: ~2-3 hours
"""

import os
import argparse
import torch
import gc
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


@dataclass
class SimPOArgs:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    sft_merged: str = "./outputs/sft_merged"
    dataset: str = "trl-lib/ultrafeedback_binarized"
    max_samples: Optional[int] = None
    beta: float = 0.1
    simpo_gamma: float = 0.5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512
    output_dir: str = "./outputs/simpo"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


def load_model_for_simpo(args: SimPOArgs):
    """Load merged SFT model with QLoRA for SimPO training."""
    print(f"\nLoading base for SimPO: {args.sft_merged}")
    print("  Note: SimPO requires NO reference model — saving ~8GB vs DPO")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_merged)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_merged,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_preference_dataset(args: SimPOArgs, tokenizer):
    """Load preference dataset — same as DPO for fair comparison."""
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    train_data = dataset["train"]
    eval_data = dataset.get("test", dataset["train"].select(range(200)))

    if args.max_samples:
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))

    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval:  {len(eval_data)} examples")

    def normalize(example):
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        if isinstance(chosen, list):
            chosen = next(
                (m["content"] for m in reversed(chosen) if m["role"] == "assistant"),
                str(chosen),
            )
        if isinstance(rejected, list):
            rejected = next(
                (m["content"] for m in reversed(rejected) if m["role"] == "assistant"),
                str(rejected),
            )
        prompt = example.get("prompt", "")
        if isinstance(prompt, list):
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=False
            )
        return {"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected)}

    train_data = train_data.map(normalize, remove_columns=train_data.column_names)
    eval_data = eval_data.map(normalize, remove_columns=eval_data.column_names)
    return train_data, eval_data


def build_simpo_config(args: SimPOArgs) -> DPOConfig:
    """DPOConfig with SimPO loss (loss_type='simpo')."""
    return DPOConfig(
        loss_type="simpo",
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )


def main():
    parser = argparse.ArgumentParser(description="SimPO training")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sft_merged", default="./outputs/sft_merged")
    parser.add_argument("--output_dir", default="./outputs/simpo")
    parser.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--simpo_gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    cli = parser.parse_args()

    args = SimPOArgs(
        base_model=cli.base_model,
        sft_merged=cli.sft_merged,
        output_dir=cli.output_dir,
        dataset=cli.dataset,
        max_samples=cli.max_samples,
        simpo_gamma=cli.simpo_gamma,
        beta=cli.beta,
        num_train_epochs=cli.epochs,
    )

    print("=" * 60)
    print("SimPO Training")
    print(f"  Base:         {args.sft_merged}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  beta:         {args.beta}")
    print(f"  simpo_gamma:  {args.simpo_gamma}")
    print(f"  No reference model needed")
    print("=" * 60)

    model, tokenizer = load_model_for_simpo(args)
    train_data, eval_data = load_preference_dataset(args, tokenizer)
    config = build_simpo_config(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nStarting SimPO training...")
    print("Watch 'rewards/accuracies' — compare to DPO (82%)\n")

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    trainer.train()

    adapter_path = os.path.join(args.output_dir, "simpo_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nSimPO adapter saved: {adapter_path}")

    if trainer.state.log_history:
        final_logs = [l for l in trainer.state.log_history
                      if "eval_rewards/accuracies" in l]
        if final_logs:
            final_acc = final_logs[-1]["eval_rewards/accuracies"]
            print(f"\nFinal SimPO reward accuracy: {final_acc:.3f}")
            print(f"DPO reward accuracy was:      0.820")
            print(f"Difference:                   {final_acc - 0.820:+.3f}")

    print("\nNext: merge adapter and run judge eval")
    print(f"  python scripts/cross_method_eval.py merge "
          f"--base {args.sft_merged} --adapter {adapter_path} "
          f"--output ./outputs/simpo_merged")


if __name__ == "__main__":
    main()
