"""DPO Trainer: Direct Preference Optimization using TRL's DPOTrainer."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distill_align.config.models import DPOConfig

logger = logging.getLogger(__name__)


@dataclass
class DPOResult:
    """Training result from a DPO run."""

    final_loss: float = 0.0
    reward_accuracy: float = 0.0
    reward_margin: float = 0.0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    adapter_path: str = ""
    metrics_history: list[dict[str, float]] = field(default_factory=list)
    low_accuracy_warnings: int = 0

    def __repr__(self) -> str:
        acc_status = "LOW" if self.reward_accuracy < 0.5 else "ok"
        return (
            f"DPOResult(loss={self.final_loss:.4f}, "
            f"reward_acc={self.reward_accuracy:.2%} [{acc_status}], "
            f"margin={self.reward_margin:.4f}, steps={self.total_steps})"
        )


class RewardAccuracyMonitor:
    """Warns when reward accuracy stays below threshold for consecutive evals."""

    def __init__(self, threshold: float = 0.5, max_consecutive: int = 3):
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self._consecutive_low: int = 0
        self._total_warnings: int = 0

    def check(self, reward_accuracy: float, step: int) -> bool:
        """Returns True if the warning threshold has been exceeded."""
        if reward_accuracy < self.threshold:
            self._consecutive_low += 1
            logger.warning(
                f"Low reward accuracy at step {step}: {reward_accuracy:.2%} "
                f"(threshold={self.threshold:.2%}, "
                f"consecutive_low={self._consecutive_low}/{self.max_consecutive})"
            )

            if self._consecutive_low >= self.max_consecutive:
                self._total_warnings += 1
                logger.error(
                    f"REWARD ACCURACY WARNING at step {step}! "
                    f"Accuracy has been below {self.threshold:.0%} for "
                    f"{self._consecutive_low} consecutive evaluations.\n"
                    f"Possible causes:\n"
                    f"  1. Chosen/rejected labels may be swapped in the dataset\n"
                    f"  2. Beta may be too high — try lowering it\n"
                    f"  3. Learning rate may be too low\n"
                    f"  4. Dataset may have ambiguous/noisy preferences"
                )
                return True
        else:
            self._consecutive_low = 0

        return False

    @property
    def total_warnings(self) -> int:
        return self._total_warnings


class DPOTrainerWrapper:
    """Wraps TRL's DPOTrainer with reward accuracy monitoring and pipeline integration."""

    def __init__(self, experiment_tracker: Any | None = None):
        self.experiment_tracker = experiment_tracker
        self._accuracy_monitor: RewardAccuracyMonitor | None = None
        self._metrics_history: list[dict[str, float]] = []

    def train(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        dataset: Any,
        config: DPOConfig,
    ) -> DPOResult:
        """Run DPO training. ref_model is a frozen copy used as the KL anchor."""
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer

        logger.info(f"Starting DPO training: beta={config.beta}, lr={config.learning_rate}")

        start_time = time.time()

        self._accuracy_monitor = RewardAccuracyMonitor(
            threshold=config.reward_accuracy_warning_threshold,
            max_consecutive=config.reward_accuracy_warning_consecutive,
        )

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TRLDPOConfig(
            output_dir=str(output_dir),
            beta=config.beta,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            warmup_ratio=config.warmup_ratio,
            fp16=config.fp16,
            bf16=config.bf16,
            gradient_checkpointing=config.gradient_checkpointing,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_strategy="steps",
            eval_strategy="no",
            report_to="none",
        )

        train_dataset = dataset["train"] if hasattr(dataset, "__getitem__") else dataset
        eval_dataset = dataset.get("validation") if hasattr(dataset, "get") else None

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        try:
            train_result = trainer.train()

            final_loss = train_result.training_loss
            total_steps = train_result.global_step

        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise

        adapter_path = str(output_dir / "dpo_adapter")
        trainer.save_model(adapter_path)
        logger.info(f"Saved DPO-aligned adapter to: {adapter_path}")

        # NOTE: Final evaluation with eval_strategy="no" can be unreliable
        # (OOM issues after training fills VRAM). We report training-log metrics
        # as primary and attempt final eval as secondary.
        reward_accuracy = 0.0
        reward_margin = 0.0

        # Extract from training logs (more reliable than post-training eval)
        train_log_accuracies = []
        if hasattr(trainer, "state") and trainer.state.log_history:
            for log_entry in trainer.state.log_history:
                if "rewards/accuracies" in log_entry:
                    train_log_accuracies.append(log_entry["rewards/accuracies"])

        if train_log_accuracies:
            # Use second-half average as the reported metric (more stable)
            second_half = train_log_accuracies[len(train_log_accuracies) // 2:]
            reward_accuracy = sum(second_half) / len(second_half) if second_half else 0.0
            logger.info(f"Training-log reward accuracy (2nd half avg): {reward_accuracy:.2%}")
            logger.info(f"Training-log reward accuracy (peak): {max(train_log_accuracies):.2%}")

        if eval_dataset is not None:
            try:
                eval_metrics = trainer.evaluate()
                eval_reward_acc = eval_metrics.get("eval_rewards/accuracies", 0.0)
                reward_margin = eval_metrics.get("eval_rewards/margins", 0.0)

                if eval_reward_acc > 0.0:
                    # Only override training-log metric if eval actually worked
                    reward_accuracy = eval_reward_acc
                    logger.info(f"Final eval reward accuracy: {reward_accuracy:.2%}")
                else:
                    logger.warning("Final eval returned 0% — likely OOM. Using training-log metrics.")
            except Exception as e:
                logger.warning(f"Final evaluation failed ({e}). Using training-log metrics.")

        self._accuracy_monitor.check(reward_accuracy, total_steps)

        training_time = time.time() - start_time

        result = DPOResult(
            final_loss=final_loss,
            reward_accuracy=reward_accuracy,
            reward_margin=reward_margin,
            total_steps=total_steps,
            training_time_seconds=training_time,
            adapter_path=adapter_path,
            metrics_history=self._metrics_history,
            low_accuracy_warnings=self._accuracy_monitor.total_warnings,
        )

        logger.info(f"DPO training complete: {result}")

        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(
                {
                    "dpo/final_loss": final_loss,
                    "dpo/reward_accuracy": reward_accuracy,
                    "dpo/reward_margin": reward_margin,
                    "dpo/total_steps": total_steps,
                    "dpo/training_time_seconds": training_time,
                    "dpo/low_accuracy_warnings": self._accuracy_monitor.total_warnings,
                },
                step=total_steps,
            )

        return result
