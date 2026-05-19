"""Configuration models for the Distill + Align pipeline using Pydantic."""

from enum import Enum

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════
# SECTION 1: Enums
# ═══════════════════════════════════════════════


class ModelFamily(str, Enum):
    """Supported model architecture families."""

    QWEN2_5 = "qwen2.5"
    LLAMA_3_1 = "llama-3.1"
    LLAMA_3_2 = "llama-3.2"
    MISTRAL = "mistral"
    PHI_3 = "phi-3"
    SMOLLM2 = "smollm2"


class QuantizationMode(str, Enum):
    """Model weight precision formats."""

    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT4_NF4 = "int4_nf4"


class DatasetType(str, Enum):
    """Types of training datasets."""

    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


class DatasetSource(str, Enum):
    """Known dataset sources with pre-built transformations."""

    HH_RLHF = "anthropic_hh_rlhf"
    ULTRAFEEDBACK = "ultrafeedback"
    OASST = "openassistant"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class TrainingMode(str, Enum):
    """Where training runs: locally or on cloud GPU (RunPod)."""

    LOCAL = "local"
    CLOUD = "cloud"


# ═══════════════════════════════════════════════
# SECTION 2: Small reusable config blocks
# ═══════════════════════════════════════════════


class QuantizationConfig(BaseModel):
    """Settings for model weight quantization."""

    mode: QuantizationMode = QuantizationMode.NONE

    quant_type: str = Field(
        default="nf4",
        description="Quantization type. 'nf4' is NormalFloat4, optimized for normally-distributed weights.",
    )
    use_double_quant: bool = Field(
        default=True,
        description="Quantize the quantization constants too. Saves ~0.4 bits/param extra.",
    )
    compute_dtype: str = Field(
        default="bfloat16",
        description="Dtype for computation during forward pass. bf16 is best for training stability.",
    )


class LoRAConfig(BaseModel):
    """Settings for Low-Rank Adaptation (LoRA).

    Args:
        rank: Size of the low-rank matrices. Higher = more capacity but more memory.
        alpha: Scaling factor. Usually set to 2x the rank.
        target_modules: Which layers get LoRA adapters.
        dropout: Regularization to prevent overfitting the adapter.
    """

    rank: int = Field(default=16, ge=1, le=256, description="LoRA rank (low-rank dimension)")
    alpha: int = Field(default=32, ge=1, description="LoRA scaling factor (usually 2x rank)")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Which model layers get LoRA adapters",
    )
    dropout: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Dropout rate for LoRA layers"
    )
    bias: str = Field(default="none", description="Bias training strategy: 'none', 'all', or 'lora_only'")
    task_type: str = Field(default="CAUSAL_LM", description="Task type for PEFT (always CAUSAL_LM for us)")


class ModelConfig(BaseModel):
    """Configuration for loading a base model."""

    model_id: str = Field(
        ...,
        description="HuggingFace Hub model ID or S3 URI",
    )
    family: ModelFamily = Field(
        ...,
        description="Model architecture family (determines chat template and LoRA targets)",
    )
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=32768,
        description="Maximum sequence length for tokenization",
    )
    device_map: str = Field(
        default="auto",
        description="Device placement strategy. 'auto' distributes across available GPUs.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow executing code from the model repo. Set True for some models (Qwen, Phi).",
    )


# ═══════════════════════════════════════════════
# SECTION 3: Training stage configs
# ═══════════════════════════════════════════════


class SFTConfig(BaseModel):
    """Supervised Fine-Tuning hyperparameters."""

    learning_rate: float = Field(default=2e-4, gt=0, description="Peak learning rate")
    batch_size: int = Field(default=4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, description="Accumulate gradients over N steps (effective batch = batch_size × N)"
    )
    num_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    warmup_steps: int = Field(default=100, ge=0, description="Linear warmup steps")
    weight_decay: float = Field(default=0.01, ge=0, description="L2 regularization strength")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Gradient clipping threshold")
    fp16: bool = Field(default=False, description="Use fp16 mixed precision")
    bf16: bool = Field(default=True, description="Use bf16 mixed precision (preferred for modern GPUs)")
    gradient_checkpointing: bool = Field(
        default=True,
        description="Trade compute for memory: recompute activations during backward pass",
    )
    logging_steps: int = Field(default=10, ge=1, description="Log metrics every N steps")
    save_steps: int = Field(default=500, ge=1, description="Save checkpoint every N steps")
    eval_steps: int = Field(default=500, ge=1, description="Run validation every N steps")
    output_dir: str = Field(default="./outputs/sft", description="Directory for checkpoints")
    divergence_threshold_multiplier: float = Field(
        default=10.0,
        gt=1.0,
        description="Halt training if loss exceeds initial_loss × this value",
    )


class DPOConfig(BaseModel):
    """Direct Preference Optimization hyperparameters.

    Args:
        beta: Controls how much the aligned model can deviate from the SFT model.
              High beta = conservative, low beta = aggressive alignment.
    """

    beta: float = Field(default=0.1, gt=0, description="DPO regularization strength")
    learning_rate: float = Field(default=1e-5, gt=0, description="Peak learning rate")
    batch_size: int = Field(default=1, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    num_epochs: int = Field(default=1, ge=1, description="Usually 1 epoch is enough for DPO")
    warmup_ratio: float = Field(default=0.1, ge=0, le=1.0, description="Warmup as fraction of total steps")
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=20, ge=1)
    save_steps: int = Field(default=250, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    output_dir: str = "./outputs/dpo"
    reward_accuracy_warning_threshold: float = Field(
        default=0.5,
        description="Warn if reward accuracy drops below this (means model can't distinguish good from bad)",
    )
    reward_accuracy_warning_consecutive: int = Field(
        default=3,
        description="Number of consecutive low-accuracy evals before warning",
    )


class RewardModelConfig(BaseModel):
    """Reward model training hyperparameters (used in RLHF)."""

    learning_rate: float = Field(default=1e-5, gt=0)
    batch_size: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=1, ge=1)
    output_dir: str = "./outputs/reward_model"


class PPOConfig(BaseModel):
    """Proximal Policy Optimization hyperparameters (used in RLHF/GRPO)."""

    kl_penalty_coeff: float = Field(default=0.2, gt=0, description="KL divergence penalty weight")
    kl_threshold: float = Field(
        default=10.0, gt=0, description="Auto-increase KL coeff when KL exceeds this"
    )
    learning_rate: float = Field(default=1e-5, gt=0)
    batch_size: int = Field(default=16, ge=1, description="PPO batch size (number of prompts per step)")
    mini_batch_size: int = Field(default=4, ge=1, description="Mini-batch for PPO gradient updates")
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    ppo_epochs: int = Field(default=4, ge=1, description="PPO optimization epochs per batch")
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=10, ge=1)
    output_dir: str = "./outputs/rlhf"


# ═══════════════════════════════════════════════
# SECTION 4: Infrastructure configs
# ═══════════════════════════════════════════════


class SageMakerConfig(BaseModel):
    """Cloud GPU training configuration (RunPod / cloud providers)."""

    training_mode: TrainingMode = TrainingMode.LOCAL
    instance_type: str = Field(
        default="RTX-A5000",
        description="GPU type. RTX A5000 (24GB VRAM), ~$0.27/hr on RunPod",
    )
    instance_count: int = Field(default=1, ge=1)
    use_spot_instances: bool = Field(
        default=True, description="Use spot instances for up to 90% cost savings"
    )
    max_wait_time_seconds: int = Field(
        default=7200, description="Max time to wait for spot capacity (2 hours)"
    )
    max_run_time_seconds: int = Field(
        default=86400, description="Max training time before timeout (24 hours)"
    )
    checkpoint_s3_uri: str = Field(default="", description="S3 path for checkpoint storage")
    ecr_image_uri: str = Field(default="", description="ECR Docker image URI for training container")
    role_arn: str = Field(default="", description="IAM role ARN (if using AWS)")
    s3_output_path: str = Field(default="", description="S3 path for training output artifacts")


class ServingConfig(BaseModel):
    """Model serving configuration (local vLLM)."""

    engine: str = Field(default="vllm", description="Serving engine: 'vllm' or 'huggingface'")
    model_path: str = Field(default="", description="Path to model weights (local or S3)")
    quantization: QuantizationMode = QuantizationMode.NONE
    max_model_len: int = Field(default=2048, ge=128, description="Max sequence length for serving")
    gpu_memory_utilization: float = Field(
        default=0.9, gt=0, le=1.0, description="Fraction of GPU memory vLLM can use"
    )
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class ProductionVariantConfig(BaseModel):
    """A single model variant for A/B testing."""

    variant_name: str
    model_artifact_s3_uri: str
    instance_type: str = "ml.g5.xlarge"
    initial_instance_count: int = Field(default=1, ge=0)
    initial_weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Traffic weight (0.0 to 1.0)"
    )


class EndpointConfig(BaseModel):
    """Endpoint configuration with auto-scaling."""

    endpoint_name: str = ""
    instance_type: str = "ml.g5.xlarge"
    min_instance_count: int = Field(default=0, ge=0, description="0 = scale to zero when idle")
    max_instance_count: int = Field(default=2, ge=1)
    target_invocations_per_instance: int = Field(
        default=10, ge=1, description="Auto-scaling target: requests per instance"
    )
    variants: list[ProductionVariantConfig] = Field(default_factory=list)


class MonitoringConfig(BaseModel):
    """Local monitoring settings (Prometheus)."""

    enabled: bool = True
    metrics_port: int = Field(default=9090, ge=1, le=65535)
    error_rate_threshold: float = Field(default=0.05, description="Alert if error rate > 5%")
    p99_latency_threshold_ms: float = Field(default=5000.0, description="Alert if p99 > 5 seconds")
    rolling_window_seconds: int = Field(default=300, description="5-minute rolling window")


class AlarmConfig(BaseModel):
    """Monitoring alarm settings."""

    p99_latency_threshold_ms: float = Field(default=5000.0)
    error_rate_threshold: float = Field(default=0.05)
    evaluation_periods: int = Field(default=3, ge=1, description="Consecutive periods before alarm")
    period_seconds: int = Field(default=60, ge=10)
    sns_topic_arn: str = Field(default="", description="SNS topic for alarm notifications")


class EvalConfig(BaseModel):
    """Evaluation and benchmarking settings."""

    benchmarks: list[str] = Field(
        default=["mmlu", "hellaswag", "truthfulqa"],
        description="lm-evaluation-harness benchmark tasks to run",
    )
    mt_bench_enabled: bool = Field(default=True, description="Run MT-Bench multi-turn eval")
    alpaca_eval_enabled: bool = Field(default=False, description="Run AlpacaEval")
    num_samples: int = Field(default=100, ge=1, description="Number of samples for custom evals")
    latency_input_lengths: list[int] = Field(
        default=[64, 128, 256, 512],
        description="Input lengths to test for latency benchmarking",
    )


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    source: str = Field(..., description="HuggingFace ID, local path, or S3 URI")
    dataset_type: DatasetType
    dataset_source: DatasetSource = DatasetSource.CUSTOM
    max_length: int = Field(default=2048, ge=1)
    padding_strategy: str = Field(default="max_length", description="'max_length' or 'longest'")
    train_ratio: float = Field(default=0.9, gt=0, lt=1.0)
    val_ratio: float = Field(default=0.05, gt=0, lt=1.0)
    test_ratio: float = Field(default=0.05, gt=0, lt=1.0)
    seed: int = Field(default=42)


class ETLConfig(BaseModel):
    """AWS Glue ETL pipeline settings."""

    raw_s3_path: str = Field(default="", description="S3 path to raw datasets")
    processed_s3_path: str = Field(default="", description="S3 path for processed Parquet output")
    dedup_strategy: str = Field(default="exact", description="'exact' or 'fuzzy' deduplication")
    min_token_length: int = Field(default=10, ge=1)
    max_token_length: int = Field(default=4096, ge=1)
    quality_failure_threshold: float = Field(
        default=0.1, gt=0, le=1.0, description="Halt if more than 10% of records fail quality checks"
    )


class WandbConfig(BaseModel):
    """Weights & Biases experiment tracking settings."""

    project: str = Field(default="distill-align-llm", description="W&B project name")
    entity: str | None = Field(default=None, description="W&B team/user name")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering runs")
    log_interval: int = Field(default=10, ge=1, description="Log metrics every N steps")


# ═══════════════════════════════════════════════
# SECTION 5: Top-level pipeline config
# ═══════════════════════════════════════════════


class PipelineConfig(BaseModel):
    """Top-level configuration for the entire Distill + Align pipeline.

    Example usage:
        config = ConfigManager.load_config("configs/local_small.yaml")
        print(config.model.model_id)
        print(config.sft.learning_rate)
    """

    model: ModelConfig
    sft: SFTConfig = Field(default_factory=SFTConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    reward_model: RewardModelConfig = Field(default_factory=RewardModelConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    sagemaker: SageMakerConfig = Field(default_factory=SageMakerConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    alarms: AlarmConfig = Field(default_factory=AlarmConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    etl: ETLConfig = Field(default_factory=ETLConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    seed: int = Field(default=42, description="Global random seed for reproducibility")
