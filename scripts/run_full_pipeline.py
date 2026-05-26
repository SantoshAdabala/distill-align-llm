"""
run_full_pipeline.py
====================
End-to-end pipeline: SFT → Merge → DPO → Factuality Eval.

Runs the complete training + eval for any model config.
Designed to be kicked off on Lightning.ai or RunPod and left to run.

USAGE:
    # Mistral-7B (cross-architecture validation)
    python scripts/run_full_pipeline.py --config configs/mistral_7b.yaml

    # Llama-3.2-3B (scale sensitivity)
    python scripts/run_full_pipeline.py --config configs/llama_3b.yaml

    # Original Llama-3.1-8B (reproduce existing results)
    python scripts/run_full_pipeline.py --config configs/local_small.yaml

    # Skip SFT if already done
    python scripts/run_full_pipeline.py --config configs/mistral_7b.yaml \
        --skip-sft --sft-adapter ./outputs/mistral_7b/sft/final_adapter

ESTIMATED TIMES (A100 SXM 80GB):
    Mistral-7B:   SFT ~12min, DPO ~70min, Eval ~15min  → ~1.5 hours
    Llama-3.2-3B: SFT ~6min,  DPO ~30min, Eval ~10min  → ~50 min
    Llama-3.1-8B: SFT ~12min, DPO ~70min, Eval ~15min  → ~1.5 hours
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_pipeline")


def run_command(cmd: list[str], description: str) -> bool:
    """Run a subprocess command and log output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE: {description}")
    logger.info(f"CMD: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"FAILED: {description} (exit code {result.returncode})")
        return False

    logger.info(f"DONE: {description} ({elapsed/60:.1f} min)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Full SFT → DPO → Eval pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT (use existing adapter)")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip DPO (use existing adapter)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip factuality eval")
    parser.add_argument("--sft-adapter", type=str, default=None, help="Path to existing SFT adapter")
    parser.add_argument("--dpo-adapter", type=str, default=None, help="Path to existing DPO adapter")
    parser.add_argument("--num-sft-epochs", type=int, default=None, help="Override SFT epochs")
    parser.add_argument("--eval-file", type=str, default="data/eval_factuality.jsonl",
                        help="Eval prompts file")
    parser.add_argument("--num-samples", type=int, default=5000, help="DPO training samples")
    args = parser.parse_args()

    # Load config to get model info and output dirs
    from distill_align.config import ConfigManager
    config = ConfigManager.load_config(args.config)

    model_id = config.model.model_id
    model_name = model_id.split("/")[-1]  # e.g., "Mistral-7B-Instruct-v0.3"
    sft_output = config.sft.output_dir
    dpo_output = config.dpo.output_dir

    logger.info(f"\n{'#'*60}")
    logger.info(f"FULL PIPELINE: {model_name}")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  SFT output: {sft_output}")
    logger.info(f"  DPO output: {dpo_output}")
    logger.info(f"{'#'*60}")

    pipeline_start = time.time()
    results = {"model": model_id, "config": args.config, "stages": {}}

    # ═══════════════════════════════════════════════
    # STAGE 1: SFT
    # ═══════════════════════════════════════════════
    sft_adapter_path = args.sft_adapter or f"{sft_output}/final_adapter"

    if not args.skip_sft:
        sft_cmd = [
            sys.executable, "scripts/run_sft.py",
            "--config", args.config,
        ]
        success = run_command(sft_cmd, f"SFT Training ({model_name})")
        if not success:
            logger.error("SFT failed. Aborting pipeline.")
            sys.exit(1)

        # Check if adapter was saved
        if not Path(sft_adapter_path).exists():
            # Try checkpoint directory
            checkpoints = sorted(Path(sft_output).glob("checkpoint-*"))
            if checkpoints:
                sft_adapter_path = str(checkpoints[-1])
                logger.info(f"Using last checkpoint as SFT adapter: {sft_adapter_path}")
            else:
                logger.error(f"No SFT adapter found at {sft_adapter_path}")
                sys.exit(1)

        results["stages"]["sft"] = {"adapter_path": sft_adapter_path, "status": "complete"}
    else:
        logger.info(f"Skipping SFT — using adapter: {sft_adapter_path}")
        results["stages"]["sft"] = {"adapter_path": sft_adapter_path, "status": "skipped"}

    # ═══════════════════════════════════════════════
    # STAGE 2: DPO (with merged-SFT strategy)
    # ═══════════════════════════════════════════════
    dpo_adapter_path = args.dpo_adapter or f"{dpo_output}/dpo_adapter"
    merged_path = sft_output.replace("/sft", "/sft_merged")

    if not args.skip_dpo:
        dpo_cmd = [
            sys.executable, "scripts/run_dpo.py",
            "--config", args.config,
            "--sft-adapter", sft_adapter_path,
            "--merge-sft",  # Always use merged-SFT strategy (best results)
            "--num-samples", str(args.num_samples),
        ]
        success = run_command(dpo_cmd, f"DPO Training ({model_name}, merged-SFT)")
        if not success:
            logger.error("DPO failed. Continuing to eval with SFT-only model.")
            results["stages"]["dpo"] = {"status": "failed"}
        else:
            # Find DPO adapter
            if not Path(dpo_adapter_path).exists():
                checkpoints = sorted(Path(dpo_output).glob("checkpoint-*"))
                if checkpoints:
                    dpo_adapter_path = str(checkpoints[-1])
            results["stages"]["dpo"] = {"adapter_path": dpo_adapter_path, "status": "complete"}
    else:
        logger.info(f"Skipping DPO — using adapter: {dpo_adapter_path}")
        results["stages"]["dpo"] = {"adapter_path": dpo_adapter_path, "status": "skipped"}

    # ═══════════════════════════════════════════════
    # STAGE 3: Factuality Evaluation
    # ═══════════════════════════════════════════════
    if not args.skip_eval:
        # Determine DPO base (merged model path)
        dpo_base_arg = merged_path if Path(merged_path).exists() else model_id

        eval_cmd = [
            sys.executable, "scripts/eval_factuality_all.py",
            "--base-model", model_id,
            "--sft-adapter", sft_adapter_path,
            "--dpo-adapter", dpo_adapter_path,
            "--dpo-base", dpo_base_arg,
            "--eval-file", args.eval_file,
            "--save-responses",
        ]
        success = run_command(eval_cmd, f"Factuality Eval ({model_name})")
        if success:
            results["stages"]["eval"] = {"status": "complete"}
        else:
            results["stages"]["eval"] = {"status": "failed"}
    else:
        results["stages"]["eval"] = {"status": "skipped"}

    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    total_time = time.time() - pipeline_start
    results["total_time_minutes"] = round(total_time / 60, 1)

    logger.info(f"\n{'#'*60}")
    logger.info(f"PIPELINE COMPLETE: {model_name}")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    for stage, info in results["stages"].items():
        logger.info(f"  {stage}: {info['status']}")
    logger.info(f"{'#'*60}")

    # Save pipeline results
    output_dir = Path(f"outputs/{model_name.lower().replace('-', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Pipeline results saved to: {results_path}")


if __name__ == "__main__":
    main()
