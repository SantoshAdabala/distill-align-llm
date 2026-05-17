"""DPO Trainer: Direct Preference Optimization using TRL's DPOTrainer."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distill_align.config.models import DPOConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════


@dataclass
class DPOResult:
    """Result of DPO training.

    Attributes:
        final_loss: Final DPO loss value.
        reward_accuracy: Fraction of examples where model prefers chosen over rejected.
        reward_margin: Average log-probability gap between chosen and rejected.
        total_steps: Total training steps completed.
        training_time_seconds: Wall-clock training time.
        adapter_path: Path to saved aligned LoRA adapter weights.
        metrics_history: List of metric dicts logged at each logging step.
        low_accuracy_warnings: Number of times reward accuracy dropped below threshold.
    """

    final_loss: float = 0.0
    reward_accuracy: float = 0.0
    reward_margin: float = 0.0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    adapter_path: str = ""
    metrics_history: list[dict[str, float]] = field(default_factory=list)
    low_accuracy_warnings: int = 0

    def __repr__(self) -> str:
        acc_status = "⚠️ LOW" if self.reward_accuracy < 0.5 else "✓"
        return (
            f"DPOResult(loss={self.final_loss:.4f}, "
            f"reward_acc={self.reward_accuracy:.2%} {acc_status}, "
            f"margin={self.reward_margin:.4f}, steps={self.total_steps})"
        )


# ═══════════════════════════════════════════════
# REWARD ACCURACY MONITOR
# ═══════════════════════════════════════════════


class RewardAccuracyMonitor:
    """Monitors reward accuracy and warns on sustained low performance.

    Tracks whether the model correctly prefers the chosen response over
    the rejected one. Logs warnings with actionable suggestions if accuracy
    drops below threshold for multiple consecutive evaluations.
    """

    def __init__(self, threshold: float = 0.5, max_consecutive: int = 3):
        """Initialize the reward accuracy monitor.

        Args:
            threshold: Minimum acceptable reward accuracy (default 0.5 = random).
            max_consecutive: Number of consecutive low evals before warning.
        """
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self._consecutive_low: int = 0
        self._total_warnings: int = 0

    def check(self, reward_accuracy: float, step: int) -> bool:
        """Check reward accuracy and warn if consistently low.

        Args:
            reward_accuracy: Current reward accuracy (0.0 to 1.0).
            step: Current training step.

        Returns:
            True if warning threshold exceeded, False otherwise.
        """
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
                    f"🚨 REWARD ACCURACY WARNING at step {step}! "
                    f"Accuracy has been below {self.threshold:.0%} for "
                    f"{self._consecutive_low} consecutive evaluations.\n"
                    f"Possible causes:\n"
                    f"  1. Chosen/rejected labels may be swapped in the dataset\n"
                    f"  2. Beta ({self.threshold}) may be too high — try lowering it\n"
                    f"  3. Learning rate may be too low\n"
                    f"  4. Dataset may have ambiguous/noisy preferences"
                )
                return True
        else:
            self._consecutive_low = 0

        return False

    @property
    def total_warnings(self) -> int:
        """Total number of warning events triggered."""
        return self._total_warnings


# ═══════════════════════════════════════════════
# MAIN DPO TRAINER CLASS
# ═══════════════════════════════════════════════


class DPOTrainerWrapper:
    """Wraps TRL's DPOTrainer with pipeline integration and monitoring.

    Provides:
    1. Simplified interface for DPO training
    2. Reward accuracy monitoring (warns if model can't distinguish good/bad)
    3. DPO-specific metric logging (loss, reward accuracy, reward margins)
    4. Saves aligned LoRA adapter weights on completion
    5. Integration with experiment tracking
    """

    def __init__(self, experiment_tracker: Any | None = None):
        """Initialize the DPO trainer wrapper.

        Args:
            experiment_tracker: Optional experiment tracker for logging metrics.
        """
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
        """Execute DPO training loop.

        Trains the model to prefer chosen responses over rejected ones,
        using the reference model as a KL-divergence anchor.

        Args:
            model: SFT model with LoRA adapters (the model being aligned).
            ref_model: Frozen copy of the SFT model (reference/anchor).
            tokenizer: Model tokenizer.
            dataset: DatasetDict with 'train' and optionally 'validation' splits.
                     Each split must have 'prompt', 'chosen', 'rejected' columns.
            config: DPO hyperparameters (beta, learning_rate, etc.).

        Returns:
            DPOResult with final metrics and adapter path.
        """
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer

        logger.info(
            f"Starting DPO training:\n"
            f"  beta={config.beta} (KL regularization strength)\n"
            f"  learning_rate={config.learning_rate}\n"
            f"  epochs={config.num_epochs}\n"
            f"  batch_size={config.batch_size}\n"
            f"  gradient_accumulation_steps={config.gradient_accumulation_steps}\n"
            f"  effective_batch_size={config.batch_size * config.gradient_accumulation_steps}"
        )

        start_time = time.time()

        self._accuracy_monitor = RewardAccuracyMonitor(
            threshold=config.reward_accuracy_warning_threshold,
            max_consecutive=config.reward_accuracy_warning_consecutive,
        )

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ─── Configure TRL's DPOConfig ───
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
            eval_steps=config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )

        # ─── Prepare datasets ───
        train_dataset = dataset["train"] if hasattr(dataset, "__getitem__") else dataset
        eval_dataset = dataset.get("validation") if hasattr(dataset, "get") else None

        # ─── Create TRL DPOTrainer ───
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        # ─── Run training ───
        try:
            train_result = trainer.train()

            final_loss = train_result.training_loss
            total_steps = train_result.global_step

        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise

        # ─── Save aligned LoRA adapter weights ───
        adapter_path = str(output_dir / "dpo_adapter")
        trainer.save_model(adapter_path)
        logger.info(f"Saved DPO-aligned adapter to: {adapter_path}")

        # ─── Evaluate and extract DPO-specific metrics ───
        reward_accuracy = 0.0
        reward_margin = 0.0

        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            reward_accuracy = eval_metrics.get("eval_rewards/accuracies", 0.0)
            reward_margin = eval_metrics.get("eval_rewards/margins", 0.0)

            self._accuracy_monitor.check(reward_accuracy, total_steps)

            logger.info(
                f"DPO eval metrics: "
                f"reward_accuracy={reward_accuracy:.2%}, "
                f"reward_margin={reward_margin:.4f}"
            )

        training_time = time.time() - start_time

        # ─── Build result ───
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

        # ─── Log to experiment tracker ───
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
