"""
ipo.py — IPO (Identity Preference Optimization) Training
=========================================================
Trains IPO on Llama-3.1-8B as an alternative alignment method to DPO.

IPO vs DPO:
- Same format: {prompt, chosen, rejected}
- Same TRL DPOTrainer — just change loss_type to "ipo"
- IPO uses a squared loss that avoids DPO's overfitting on
  deterministic preferences (Azar et al. 2023)
- Available in TRL 1.5.1

If IPO also shows near-zero AFG under judge evaluation, the paper
claim becomes: "measurement artifact holds across DPO and its
theoretically-motivated variants."

Usage:
    python scripts/ipo.py \
        --sft_merged  ./outputs/sft_merged \
        --output_dir  ./outputs/ipo \
        --max_samples 5000 \
        --beta 0.1

Cost: ~$3-5 on A100 (5000 samples, ~35 min)
"""

import os
import ast
import argparse
import torch
import gc
from typing import Optional

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(sft_merged: str):
    """Load policy model with LoRA from merged SFT checkpoint."""
    print(f"\nLoading model from: {sft_merged}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_merged)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        sft_merged,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Dataset parsing ───────────────────────────────────────────────────────────

def parse_conversation(raw) -> tuple[str, str]:
    """Extract (prompt, response) from ultrafeedback_binarized format.

    The dataset has NO separate prompt column. Each of chosen/rejected
    is a full conversation: list of {role, content} dicts where:
      turn 0: role=user      → this is the prompt
      turn 1: role=assistant → this is the response
    """
    if isinstance(raw, str):
        try:
            turns = ast.literal_eval(raw)
        except Exception:
            return "", raw
    elif isinstance(raw, list):
        turns = raw
    else:
        return "", str(raw)

    prompt = ""
    response = ""
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user" and not prompt:
            prompt = content
        elif role == "assistant":
            response = content

    return prompt, response


def load_dataset_for_ipo(dataset_name: str, max_samples: Optional[int]):
    """Load and parse ultrafeedback_binarized for IPO training."""
    print(f"\nLoading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    train_data = dataset["train"]

    # Use last 500 for eval
    eval_data = train_data.select(range(max(0, len(train_data) - 500), len(train_data)))
    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data) - 500)))

    def normalize(example):
        prompt, chosen = parse_conversation(example["chosen"])
        prompt_r, rejected = parse_conversation(example["rejected"])
        if not prompt and prompt_r:
            prompt = prompt_r
        if not prompt.strip():
            prompt = "Answer the following:"
        return {
            "prompt": prompt.strip(),
            "chosen": chosen.strip(),
            "rejected": rejected.strip(),
        }

    # Verify parsing on first example
    sample = normalize(train_data[0])
    print(f"  Dataset format check:")
    print(f"    prompt[:80]:   {sample['prompt'][:80]}")
    print(f"    chosen[:60]:   {sample['chosen'][:60]}")
    print(f"    rejected[:60]: {sample['rejected'][:60]}")
    assert sample["prompt"].strip(), "ERROR: prompt is empty after parsing!"
    assert sample["chosen"].strip(), "ERROR: chosen is empty after parsing!"
    print(f"  Format OK. Mapping dataset...")

    train_data = train_data.map(normalize, remove_columns=train_data.column_names)
    eval_data = eval_data.map(normalize, remove_columns=eval_data.column_names)
    print(f"  Train: {len(train_data)}  Eval: {len(eval_data)}")
    return train_data, eval_data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IPO training")
    parser.add_argument("--sft_merged", default="./outputs/sft_merged")
    parser.add_argument("--output_dir", default="./outputs/ipo")
    parser.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    print("=" * 60)
    print("IPO Training (alternative alignment method)")
    print(f"  Base:       {args.sft_merged}")
    print(f"  loss_type:  ipo")
    print(f"  beta:       {args.beta}")
    print(f"  Dataset:    {args.dataset}")
    print("=" * 60)

    model, tokenizer = load_model(args.sft_merged)
    train_data, eval_data = load_dataset_for_ipo(args.dataset, args.max_samples)

    config = DPOConfig(
        loss_type="ipo",
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        output_dir=args.output_dir,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print("\nStarting IPO training...")
    print("Watch 'rewards/accuracies' — compare to DPO (82%)\n")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer handles ref internally for IPO
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    trainer.train()

    adapter_path = os.path.join(args.output_dir, "ipo_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nIPO adapter saved: {adapter_path}")

    if trainer.state.log_history:
        accs = [l["eval_rewards/accuracies"]
                for l in trainer.state.log_history
                if "eval_rewards/accuracies" in l]
        if accs:
            print(f"\nFinal IPO reward accuracy:  {accs[-1]:.3f}")
            print(f"DPO reward accuracy was:     0.820")
            print(f"Difference:                  {accs[-1] - 0.820:+.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nNext steps:")
    print(f"  python scripts/cross_method_eval.py merge \\")
    print(f"      --base {args.sft_merged} \\")
    print(f"      --adapter {adapter_path} \\")
    print(f"      --output ./outputs/ipo_merged")


if __name__ == "__main__":
    main()
