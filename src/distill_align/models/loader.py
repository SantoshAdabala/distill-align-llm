"""ModelLoader: Loads pre-trained models with quantization and LoRA configuration."""

import logging
from typing import Any

from distill_align.config.models import (
    ModelConfig,
    ModelFamily,
    QuantizationConfig,
    QuantizationMode,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# SUPPORTED MODEL FAMILIES AND THEIR CONFIGURATIONS
# ═══════════════════════════════════════════════

MODEL_FAMILY_CONFIG = {
    ModelFamily.QWEN2_5: {
        "trust_remote_code": True,
        "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "Qwen 2.5 series (0.5B to 72B) — Alibaba's multilingual model",
    },
    ModelFamily.LLAMA_3_1: {
        "trust_remote_code": False,
        "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "Meta Llama 3.1 series (8B, 70B, 405B) — strong English baseline",
    },
    ModelFamily.LLAMA_3_2: {
        "trust_remote_code": False,
        "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "Meta Llama 3.2 series (1B, 3B) — lightweight for local dev",
    },
    ModelFamily.MISTRAL: {
        "trust_remote_code": False,
        "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "Mistral AI series (7B, Mixtral) — efficient sliding window attention",
    },
    ModelFamily.PHI_3: {
        "trust_remote_code": True,
        # Phi-3 uses fused QKV projection — one matrix instead of three separate ones
        "default_target_modules": ["qkv_proj", "o_proj"],
        "description": "Microsoft Phi-3 series (mini, small, medium) — strong for size",
    },
    ModelFamily.SMOLLM2: {
        "trust_remote_code": True,
        "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "HuggingFace SmolLM2 series (135M, 360M, 1.7B) — tiny but capable",
    },
}


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded.

    Common causes:
    - Out of memory (OOM) — model too large for available GPU
    - Invalid model ID — typo or private model without auth
    - Network issues — can't reach HuggingFace Hub
    - Unsupported architecture — model family not in our config
    - Missing dependencies — bitsandbytes not installed for quantization
    """

    pass


class ModelLoader:
    """Loads pre-trained models with quantization and LoRA configuration.

    Orchestrates the full model loading pipeline:
    1. Validate the model family is supported
    2. Build quantization config (BitsAndBytesConfig for 4-bit NF4)
    3. Load the base model from HuggingFace Hub or local path
    4. Apply LoRA adapters via PEFT (get_peft_model)
    5. Load the tokenizer with correct padding/chat template settings
    6. Handle OOM errors with actionable suggestions
    """

    def load_model(self, config: ModelConfig) -> tuple[Any, Any]:
        """Load a model with quantization and LoRA adapters.

        Args:
            config: Model configuration specifying model_id, family,
                    quantization settings, and LoRA hyperparameters.

        Returns:
            Tuple of (model_with_lora, tokenizer) ready for training.

        Raises:
            ModelLoadError: If model cannot be loaded (OOM, invalid ID,
                           unsupported family, missing dependencies).
        """
        import torch

        self._validate_model_family(config.family)

        logger.info(
            f"Loading model: {config.model_id} "
            f"(family={config.family.value}, "
            f"quantization={config.quantization.mode.value})"
        )

        try:
            quant_config = self._build_quantization_config(config.quantization)
            model = self._load_base_model(config, quant_config)
            model = self._apply_lora(model, config)
            tokenizer = self._load_tokenizer(config)
            self._log_model_metadata(model, config)

            return model, tokenizer

        except torch.cuda.OutOfMemoryError as e:
            self._handle_oom_error(config, e)
            raise  # unreachable, but satisfies mypy
        except Exception as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                self._handle_oom_error(config, e)
            raise ModelLoadError(
                f"Failed to load model '{config.model_id}': {e}\n"
                f"Suggestions:\n"
                f"  1. Check the model ID is correct on HuggingFace Hub\n"
                f"  2. Ensure you have internet access (or use a local path)\n"
                f"  3. Try a smaller model or enable quantization\n"
                f"  4. Check that required libraries are installed (transformers, peft)"
            ) from e

    def _build_quantization_config(self, config: QuantizationConfig) -> Any | None:
        """Build a BitsAndBytesConfig for model quantization.

        Args:
            config: Our quantization configuration.

        Returns:
            BitsAndBytesConfig for INT4_NF4 mode, or None for other modes.
            FP16/BF16 are handled via torch_dtype in _load_base_model instead.
        """
        import torch

        if config.mode in (QuantizationMode.NONE, QuantizationMode.FP16, QuantizationMode.BF16):
            return None

        if config.mode == QuantizationMode.INT4_NF4:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ModelLoadError(
                    "bitsandbytes is required for 4-bit quantization.\n"
                    "Install with: pip install bitsandbytes\n"
                    "Note: bitsandbytes requires a CUDA GPU on Linux. "
                    "On macOS, use quantization.mode = 'none' or 'bf16'."
                ) from e

            compute_dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            compute_dtype = compute_dtype_map.get(config.compute_dtype, torch.bfloat16)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.quant_type,
                bnb_4bit_use_double_quant=config.use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )

            logger.info(
                f"Quantization config: 4-bit NF4, "
                f"double_quant={config.use_double_quant}, "
                f"compute_dtype={config.compute_dtype}"
            )
            return bnb_config

        return None

    def _load_base_model(self, config: ModelConfig, quant_config: Any | None) -> Any:
        """Load the base model from HuggingFace Hub or local path.

        Args:
            config: Model configuration with model_id, family, etc.
            quant_config: BitsAndBytesConfig or None.

        Returns:
            Loaded model (not yet wrapped with LoRA).
        """
        import torch
        from transformers import AutoModelForCausalLM

        if config.quantization.mode == QuantizationMode.BF16:
            torch_dtype = torch.bfloat16
        elif config.quantization.mode == QuantizationMode.FP16:
            torch_dtype = torch.float16
        elif config.quantization.mode == QuantizationMode.INT4_NF4:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        family_config = MODEL_FAMILY_CONFIG.get(config.family, {})
        trust_remote_code = config.trust_remote_code or family_config.get(
            "trust_remote_code", False
        )

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": config.device_map,
            "trust_remote_code": trust_remote_code,
        }

        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        logger.info(f"Loading base model from: {config.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, **model_kwargs
        )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        return model

    def _apply_lora(self, model: Any, config: ModelConfig) -> Any:
        """Apply LoRA adapters to the model using PEFT.

        Wraps the model with LoRA adapters, making only the adapter
        parameters trainable while freezing the base model weights.

        Args:
            model: Base model loaded from HuggingFace.
            config: Model configuration with LoRA settings.

        Returns:
            Model wrapped with LoRA adapters (only adapters are trainable).
        """
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        # For quantized models, prepare internal state for k-bit training
        if config.quantization.mode == QuantizationMode.INT4_NF4:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)

        family_config = MODEL_FAMILY_CONFIG.get(config.family, {})
        target_modules: list[str] = list(config.lora.target_modules)

        # Use family-specific defaults if user hasn't customized
        if target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]:
            default_targets = family_config.get(
                "default_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            target_modules = list(default_targets)

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }
        task_type = task_type_map.get(config.lora.task_type, TaskType.CAUSAL_LM)

        lora_config = PeftLoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=target_modules,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,                 # type: ignore[arg-type]
            task_type=task_type,
        )

        logger.info(
            f"LoRA config: rank={config.lora.rank}, alpha={config.lora.alpha}, "
            f"targets={target_modules}, dropout={config.lora.dropout}"
        )

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = trainable_params / total_params * 100

        logger.info(
            f"LoRA applied: trainable={trainable_params:,} "
            f"({trainable_pct:.2f}%) / total={total_params:,}"
        )

        return model

    def _load_tokenizer(self, config: ModelConfig) -> Any:
        """Load the tokenizer for the model.

        Args:
            config: Model configuration with model_id and family.

        Returns:
            Configured tokenizer ready for training.
        """
        from transformers import AutoTokenizer

        family_config = MODEL_FAMILY_CONFIG.get(config.family, {})
        trust_remote_code = config.trust_remote_code or family_config.get(
            "trust_remote_code", False
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=trust_remote_code,
        )

        # Many decoder-only models lack a dedicated pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # padding_side must be "right" for causal LM training
        tokenizer.padding_side = "right"

        logger.info(
            f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
            f"pad_token='{tokenizer.pad_token}', "
            f"has_chat_template={tokenizer.chat_template is not None}"
        )

        return tokenizer

    def _validate_model_family(self, family: ModelFamily) -> None:
        """Validate that the model family is in our supported list.

        Args:
            family: The ModelFamily enum value to validate.

        Raises:
            ModelLoadError: If the family is not supported.
        """
        if family not in MODEL_FAMILY_CONFIG:
            supported = [f.value for f in MODEL_FAMILY_CONFIG]
            raise ModelLoadError(
                f"Unsupported model family: '{family.value}'. "
                f"Supported families: {supported}\n"
                f"To add support for a new family, add it to MODEL_FAMILY_CONFIG "
                f"in loader.py with trust_remote_code and default_target_modules."
            )

    def _log_model_metadata(self, model: Any, config: ModelConfig) -> None:
        """Log model size, parameter count, trainable params, and memory footprint.

        Args:
            model: Model with LoRA adapters applied.
            config: Model configuration for context.
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        if config.quantization.mode == QuantizationMode.INT4_NF4:
            base_memory_gb = total_params * 0.5 / (1024**3)
            adapter_memory_gb = trainable_params * 4 / (1024**3)
            model_memory_gb = base_memory_gb + adapter_memory_gb
        elif config.quantization.mode in (QuantizationMode.FP16, QuantizationMode.BF16):
            model_memory_gb = total_params * 2 / (1024**3)
        else:
            model_memory_gb = total_params * 4 / (1024**3)

        metadata = {
            "model_id": config.model_id,
            "family": config.family.value,
            "total_parameters": f"{total_params:,}",
            "trainable_parameters": f"{trainable_params:,}",
            "trainable_percentage": f"{trainable_params / total_params * 100:.2f}%",
            "quantization": config.quantization.mode.value,
            "lora_rank": config.lora.rank,
            "lora_alpha": config.lora.alpha,
            "estimated_memory_gb": f"{model_memory_gb:.2f}",
        }

        logger.info(f"Model metadata: {metadata}")

    def _handle_oom_error(self, config: ModelConfig, error: Exception) -> None:
        """Handle Out-of-Memory errors with descriptive, actionable messages.

        Args:
            config: Current model configuration (to tailor suggestions).
            error: The original OOM exception.

        Raises:
            ModelLoadError: Always raised with helpful suggestions.
        """
        suggestions = []

        if config.quantization.mode == QuantizationMode.NONE:
            suggestions.append(
                "Enable 4-bit quantization: set quantization.mode = 'int4_nf4' "
                "(saves ~8× memory)"
            )
        elif config.quantization.mode in (QuantizationMode.FP16, QuantizationMode.BF16):
            suggestions.append(
                "Switch to 4-bit: set quantization.mode = 'int4_nf4' "
                "(saves ~4× more memory vs fp16)"
            )

        suggestions.extend([
            "Reduce max_seq_length (shorter sequences = less activation memory)",
            f"Use a smaller model (current: {config.model_id})",
            "Enable gradient checkpointing in training config (saves ~30% memory)",
            "Reduce batch size to 1 (minimum memory usage)",
            "Close other GPU-using processes (check with: nvidia-smi)",
        ])

        suggestion_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(suggestions))

        raise ModelLoadError(
            f"Out of Memory loading '{config.model_id}' "
            f"(quantization={config.quantization.mode.value}).\n\n"
            f"Suggestions to reduce memory usage:\n{suggestion_text}\n\n"
            f"Original error: {error}"
        ) from error
