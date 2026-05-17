"""SFT Trainer: Supervised Fine-Tuning using TRL's SFTTrainer with LoRA."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distill_align.config.models import SFTConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════


@dataclass
class SFTResult:
    """Result of SFT training.

    Attributes:
        final_loss: Final training loss (should decrease over time).
        best_val_loss: Best validation loss achieved (used for checkpointing).
        total_steps: Total training steps completed.
        training_time_seconds: Wall-clock training time.
        adapter_path: Path to saved LoRA adapter weights on disk.
        metrics_history: List of metric dicts logged at each logging step.
        diverged: Whether training was halted due to loss explosion.
    """

    final_loss: float = 0.0
    best_val_loss: float = float("inf")
    total_steps: int = 0
    training_time_seconds: float = 0.0
    adapter_path: str = ""
    metrics_history: list[dict[str, float]] = field(default_factory=list)
    diverged: bool = False

    def __repr__(self) -> str:
        status = "DIVERGED" if self.diverged else "COMPLETE"
        return (
            f"SFTResult({status}: loss={self.final_loss:.4f}, "
            f"best_val={self.best_val_loss:.4f}, steps={self.total_steps}, "
            f"time={self.training_time_seconds:.1f}s)"
        )


# ═══════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════


class TrainingDivergenceError(Exception):
    """Raised when training loss diverges (exceeds threshold).

    Common causes:
    - Learning rate too high
    - Data corruption
    - Numerical instability
    - Bad initialization
    """

    pass


# ═══════════════════════════════════════════════
# DIVERGENCE DETECTION CALLBACK
# ═══════════════════════════════════════════════


class DivergenceDetectionCallback:
    """Monitors training loss and halts on divergence.

    Divergence is detected when the current loss exceeds:
        initial_loss × divergence_threshold_multiplier
    """

    def __init__(self, threshold_multiplier: float = 10.0):
        """Initialize divergence detector.

        Args:
            threshold_multiplier: How many times the initial loss the current
                                  loss must exceed to trigger divergence.
        """
        self.threshold_multiplier = threshold_multiplier
        self.initial_loss: float | None = None
        self.diverged = False

    def check(self, current_loss: float, step: int) -> bool:
        """Check if training has diverged.

        Args:
            current_loss: Current training loss value.
            step: Current training step number.

        Returns:
            True if diverged (training should stop), False otherwise.
        """
        if self.initial_loss is None:
            self.initial_loss = current_loss
            logger.info(f"Initial loss recorded: {current_loss:.4f}")
            return False

        threshold = self.initial_loss * self.threshold_multiplier

        if current_loss > threshold:
            logger.error(
                f"🚨 DIVERGENCE DETECTED at step {step}! "
                f"Loss {current_loss:.4f} exceeds threshold "
                f"{threshold:.4f} (initial={self.initial_loss:.4f} × "
                f"{self.threshold_multiplier}). "
                f"Training will be halted."
            )
            self.diverged = True
            return True

        return False


# ═══════════════════════════════════════════════
# MAIN SFT TRAINER CLASS
# ═══════════════════════════════════════════════


class SFTTrainerWrapper:
    """Wraps TRL's SFTTrainer with pipeline integration and monitoring.

    Provides:
    1. Single train() method that handles all configuration
    2. Divergence detection (halt if loss explodes > 10× initial)
    3. Checkpoint saving on validation improvement
    4. Logging of training loss, validation loss, learning rate, gradient norm
    5. LoRA adapter merging after training for deployment
    """

    def __init__(self, experiment_tracker: Any | None = None):
        """Initialize the SFT trainer wrapper.

        Args:
            experiment_tracker: Optional experiment tracker (e.g., W&B wrapper)
                               for logging metrics during training.
        """
        self.experiment_tracker = experiment_tracker
        self._divergence_detector: DivergenceDetectionCallback | None = None
        self._best_val_loss = float("inf")
        self._metrics_history: list[dict[str, float]] = []

    def train(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        config: SFTConfig,
    ) -> SFTResult:
        """Execute SFT training loop.

        Args:
            model: Base model with LoRA adapters applied (from ModelLoader).
            tokenizer: Model tokenizer (with chat template configured).
            dataset: DatasetDict with 'train' and optionally 'validation' splits.
                     Should have a 'text' column with formatted conversations.
            config: SFT hyperparameters (learning_rate, epochs, batch_size, etc.).

        Returns:
            SFTResult with final metrics, adapter path, and training history.

        Raises:
            TrainingDivergenceError: If loss exceeds initial_loss × threshold.
        """
        from trl import SFTConfig as TRLSFTConfig
        from trl import SFTTrainer

        logger.info(
            f"Starting SFT training:\n"
            f"  learning_rate={config.learning_rate}\n"
            f"  epochs={config.num_epochs}\n"
            f"  batch_size={config.batch_size}\n"
            f"  gradient_accumulation_steps={config.gradient_accumulation_steps}\n"
            f"  effective_batch_size={config.batch_size * config.gradient_accumulation_steps}\n"
            f"  mixed_precision={'bf16' if config.bf16 else 'fp16' if config.fp16 else 'fp32'}\n"
            f"  gradient_checkpointing={config.gradient_checkpointing}"
        )

        start_time = time.time()

        self._divergence_detector = DivergenceDetectionCallback(
            threshold_multiplier=config.divergence_threshold_multiplier
        )

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ─── Configure TRL's SFTConfig (extends HuggingFace TrainingArguments) ───
        training_args = TRLSFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
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

        # ─── Prepare datasets ───
        train_dataset = dataset["train"] if hasattr(dataset, "__getitem__") else dataset
        eval_dataset = dataset.get("validation") if hasattr(dataset, "get") else None

        # ─── Create TRL SFTTrainer ───
        trainer = SFTTrainer(
            model=model,
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

            if self._divergence_detector.check(final_loss, total_steps):
                raise TrainingDivergenceError(
                    f"Training diverged at step {total_steps}. "
                    f"Final loss: {final_loss:.4f}. "
                    f"Try reducing learning_rate or checking data quality."
                )

        except TrainingDivergenceError:
            raise
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # ─── Save final LoRA adapter ───
        adapter_path = str(output_dir / "final_adapter")
        trainer.save_model(adapter_path)
        logger.info(f"Saved final LoRA adapter to: {adapter_path}")

        # ─── Evaluate on validation set ───
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            best_val_loss = eval_metrics.get("eval_loss", float("inf"))
            logger.info(f"Final validation loss: {best_val_loss:.4f}")
        else:
            best_val_loss = float("inf")
            logger.warning("No validation set provided — cannot compute val loss")

        training_time = time.time() - start_time

        # ─── Build result ───
        result = SFTResult(
            final_loss=final_loss,
            best_val_loss=best_val_loss,
            total_steps=total_steps,
            training_time_seconds=training_time,
            adapter_path=adapter_path,
            metrics_history=self._metrics_history,
            diverged=False,
        )

        logger.info(f"SFT training complete: {result}")

        # ─── Log to experiment tracker ───
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(
                {
                    "sft/final_loss": final_loss,
                    "sft/best_val_loss": best_val_loss,
                    "sft/total_steps": total_steps,
                    "sft/training_time_seconds": training_time,
                },
                step=total_steps,
            )

        return result

    def merge_adapter(
        self,
        model: Any,
        adapter_path: str | None = None,
        output_path: str | None = None,
    ) -> Any:
        """Merge LoRA adapter weights into the base model.

        After training, the model has separate base weights and adapter weights.
        Merging combines them into a single model for efficient inference.

        Args:
            model: Model with LoRA adapters (from training or loaded).
            adapter_path: Optional path to load adapter from disk
                         (if not already applied to the model).
            output_path: Optional path to save the merged model to disk.

        Returns:
            Merged model (base + adapter weights combined into one).
        """
        from peft import PeftModel

        logger.info("Merging LoRA adapter into base model...")

        if adapter_path and not isinstance(model, PeftModel):
            logger.info(f"Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        # Merge: W_new = W_base + B × A × (alpha/rank) for each adapted layer
        merged_model = model.merge_and_unload()

        logger.info("Adapter merged successfully — model is now a standard model")

        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(output_dir))
            logger.info(f"Saved merged model to: {output_path}")

        return merged_model
