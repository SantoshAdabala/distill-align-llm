"""RLHF Trainer: Reinforcement Learning from Human Feedback using reward models and GRPO."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distill_align.config.models import PPOConfig, RewardModelConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardModelResult:
    """Training result from reward model training."""

    final_loss: float = 0.0
    accuracy: float = 0.0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    model_path: str = ""
    model: Any = None

    def __repr__(self) -> str:
        return (
            f"RewardModelResult(loss={self.final_loss:.4f}, "
            f"accuracy={self.accuracy:.2%}, steps={self.total_steps})"
        )


@dataclass
class GRPOResult:
    """Training result from GRPO (RLHF) training."""

    mean_reward: float = 0.0
    mean_kl: float = 0.0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    adapter_path: str = ""
    metrics_history: list[dict[str, float]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"GRPOResult(reward={self.mean_reward:.4f}, "
            f"kl={self.mean_kl:.4f}, steps={self.total_steps})"
        )


# Keep backward compatibility
PPOResult = GRPOResult


class AdaptiveKLController:
    """Adjusts KL penalty coefficient to maintain a target divergence level.

    If KL > threshold: increase coefficient (pull model back toward reference).
    If KL < threshold / 2: decrease coefficient (give model more freedom).
    """

    def __init__(self, init_coeff: float, kl_threshold: float):
        self.coeff = init_coeff
        self.kl_threshold = kl_threshold
        self._violations = 0

    def update(self, current_kl: float, step: int) -> float:
        """Update and return the KL coefficient based on current divergence."""
        if current_kl > self.kl_threshold:
            old_coeff = self.coeff
            self.coeff *= 1.5
            self._violations += 1
            logger.warning(
                f"KL violation at step {step}: KL={current_kl:.4f} > "
                f"threshold={self.kl_threshold:.4f}. "
                f"Increasing KL coeff: {old_coeff:.4f} -> {self.coeff:.4f}"
            )
        elif current_kl < self.kl_threshold / 2:
            old_coeff = self.coeff
            self.coeff *= 0.8
            logger.debug(
                f"KL well below threshold at step {step}: KL={current_kl:.4f}. "
                f"Decreasing KL coeff: {old_coeff:.4f} -> {self.coeff:.4f}"
            )

        return self.coeff

    @property
    def violations(self) -> int:
        """Number of times KL exceeded the threshold."""
        return self._violations


class RLHFTrainerWrapper:
    """Wraps TRL's RewardTrainer and GRPOTrainer for full RLHF.

    Two main methods: train_reward_model() and train_grpo().
    """

    def __init__(self, experiment_tracker: Any | None = None):
        self.experiment_tracker = experiment_tracker
        self._kl_controller: AdaptiveKLController | None = None
        self._metrics_history: list[dict[str, float]] = []

    def train_reward_model(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        config: RewardModelConfig,
    ) -> RewardModelResult:
        """Train a reward model on preference data (chosen/rejected pairs)."""
        from trl import RewardConfig, RewardTrainer

        logger.info(f"Starting reward model training: lr={config.learning_rate}, epochs={config.num_epochs}")

        start_time = time.time()

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = RewardConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
        )

        train_dataset = dataset["train"] if hasattr(dataset, "__getitem__") else dataset
        eval_dataset = dataset.get("validation") if hasattr(dataset, "get") else None

        trainer = RewardTrainer(
            model=model,
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
            logger.error(f"Reward model training failed: {e}")
            raise

        accuracy = 0.0
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            accuracy = eval_metrics.get("eval_accuracy", 0.0)
            logger.info(f"Reward model accuracy: {accuracy:.2%}")

        model_path = str(output_dir / "reward_model")
        trainer.save_model(model_path)
        logger.info(f"Saved reward model to: {model_path}")

        training_time = time.time() - start_time

        result = RewardModelResult(
            final_loss=final_loss,
            accuracy=accuracy,
            total_steps=total_steps,
            training_time_seconds=training_time,
            model_path=model_path,
            model=trainer.model,
        )

        logger.info(f"Reward model training complete: {result}")

        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(
                {
                    "reward_model/final_loss": final_loss,
                    "reward_model/accuracy": accuracy,
                    "reward_model/total_steps": total_steps,
                    "reward_model/training_time_seconds": training_time,
                },
                step=total_steps,
            )

        return result

    def train_grpo(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        reward_funcs: Any,
        config: PPOConfig,
    ) -> GRPOResult:
        """Train the policy model using GRPO with a reward function.

        GRPO generates multiple completions per prompt and uses relative ranking
        within the group as the advantage signal — no separate value model needed.
        reward_funcs signature: def fn(completions, **kwargs) -> list[float]
        """
        from trl import GRPOConfig, GRPOTrainer

        logger.info(f"Starting GRPO training: lr={config.learning_rate}, kl_coeff={config.kl_penalty_coeff}")

        start_time = time.time()

        self._kl_controller = AdaptiveKLController(
            init_coeff=config.kl_penalty_coeff,
            kl_threshold=config.kl_threshold,
        )

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        grpo_config = GRPOConfig(
            output_dir=str(output_dir),
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=1,
            num_generations=config.ppo_epochs,
            beta=config.kl_penalty_coeff,
            bf16=config.bf16,
            gradient_checkpointing=config.gradient_checkpointing,
            logging_steps=config.logging_steps,
            report_to="none",
            max_completion_length=256,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            processing_class=tokenizer,
        )

        try:
            train_result = trainer.train()
            total_steps = train_result.global_step

        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            raise

        adapter_path = str(output_dir / "rlhf_adapter")
        trainer.save_model(adapter_path)
        logger.info(f"Saved RLHF-aligned model to: {adapter_path}")

        training_time = time.time() - start_time

        mean_reward = 0.0
        mean_kl = 0.0
        if hasattr(trainer, "state") and trainer.state.log_history:
            rewards = [
                entry.get("reward", 0.0)
                for entry in trainer.state.log_history
                if "reward" in entry
            ]
            kls = [
                entry.get("kl", 0.0)
                for entry in trainer.state.log_history
                if "kl" in entry
            ]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            mean_kl = sum(kls) / len(kls) if kls else 0.0

        result = GRPOResult(
            mean_reward=mean_reward,
            mean_kl=mean_kl,
            total_steps=total_steps,
            training_time_seconds=training_time,
            adapter_path=adapter_path,
            metrics_history=self._metrics_history,
        )

        logger.info(f"GRPO training complete: {result}")

        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(
                {
                    "grpo/mean_reward": mean_reward,
                    "grpo/mean_kl": mean_kl,
                    "grpo/total_steps": total_steps,
                    "grpo/training_time_seconds": training_time,
                },
                step=total_steps,
            )

        return result

    # Backward compatibility alias
    train_ppo = train_grpo

    def create_reward_function(self, reward_model: Any, tokenizer: Any) -> Any:
        """Wrap a trained reward model as a GRPO-compatible reward function."""
        import torch

        def reward_fn(completions: list[str], **kwargs) -> list[float]:
            """Score completions using the trained reward model."""
            rewards = []
            device = next(reward_model.parameters()).device

            for text in completions:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = reward_model(**inputs)

                if hasattr(outputs, "logits"):
                    reward = outputs.logits.squeeze(-1).item()
                else:
                    reward = outputs[0].squeeze(-1).item()

                rewards.append(reward)

            return rewards

        return reward_fn
