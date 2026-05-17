"""Unit tests for the configuration system."""

import logging
from pathlib import Path

import pytest
import yaml

from distill_align.config import (
    ConfigManager,
    ConfigValidationError,
    DatasetConfig,
    ModelFamily,
    QuantizationMode,
)

# ═══════════════════════════════════════════════
# HELPER: Create a minimal valid config dict
# ═══════════════════════════════════════════════


def make_minimal_config_dict() -> dict:
    """Create the smallest valid config dictionary.

    Only includes required fields. All optional fields will use
    their Pydantic defaults.
    """
    return {
        "model": {
            "model_id": "Qwen/Qwen2.5-1.5B",
            "family": "qwen2.5",
        }
    }


def make_full_config_dict() -> dict:
    """Create a config dict with many fields explicitly set."""
    return {
        "model": {
            "model_id": "meta-llama/Llama-3.1-8B",
            "family": "llama-3.1",
            "max_seq_length": 2048,
            "quantization": {
                "mode": "int4_nf4",
                "quant_type": "nf4",
                "use_double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "lora": {
                "rank": 32,
                "alpha": 64,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "dropout": 0.05,
            },
        },
        "sft": {
            "learning_rate": 0.0002,
            "batch_size": 4,
            "num_epochs": 3,
        },
        "dpo": {
            "beta": 0.1,
            "learning_rate": 0.00005,
        },
        "seed": 42,
    }


# ═══════════════════════════════════════════════
# TEST 1: Round-trip serialization
# ═══════════════════════════════════════════════


class TestRoundTrip:
    """Test that config survives save → load without data loss."""

    def test_minimal_config_round_trip(self, tmp_path: Path):
        """Minimal config (only required fields) survives round-trip."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        yaml_path = tmp_path / "test_config.yaml"
        ConfigManager.save_config(config, yaml_path)

        loaded_config = ConfigManager.load_config(yaml_path)

        assert loaded_config.model.model_id == config.model.model_id
        assert loaded_config.model.family == config.model.family
        assert loaded_config.seed == config.seed

    def test_full_config_round_trip(self, tmp_path: Path):
        """Config with many explicit fields survives round-trip."""
        config = ConfigManager.from_dict(make_full_config_dict())

        yaml_path = tmp_path / "test_config.yaml"
        ConfigManager.save_config(config, yaml_path)
        loaded_config = ConfigManager.load_config(yaml_path)

        assert loaded_config.model.model_id == "meta-llama/Llama-3.1-8B"
        assert loaded_config.model.family == ModelFamily.LLAMA_3_1
        assert loaded_config.model.quantization.mode == QuantizationMode.INT4_NF4
        assert loaded_config.model.quantization.use_double_quant is True
        assert loaded_config.model.lora.rank == 32
        assert loaded_config.model.lora.alpha == 64
        assert loaded_config.sft.learning_rate == 0.0002
        assert loaded_config.sft.batch_size == 4
        assert loaded_config.dpo.beta == 0.1
        assert loaded_config.seed == 42

    def test_round_trip_preserves_all_defaults(self, tmp_path: Path):
        """Default values are preserved through round-trip."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        yaml_path = tmp_path / "test_config.yaml"
        ConfigManager.save_config(config, yaml_path)
        loaded_config = ConfigManager.load_config(yaml_path)

        assert loaded_config.sft.learning_rate == 2e-4
        assert loaded_config.sft.gradient_checkpointing is True
        assert loaded_config.dpo.beta == 0.1
        assert loaded_config.ppo.kl_penalty_coeff == 0.2
        assert loaded_config.model.lora.rank == 16


# ═══════════════════════════════════════════════
# TEST 2: Validation errors for missing required fields
# ═══════════════════════════════════════════════


class TestValidationErrors:
    """Test that invalid configs produce clear error messages."""

    def test_missing_model_section_raises_error(self):
        """Config without 'model' section should fail."""
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager.from_dict({"sft": {"learning_rate": 0.001}})

        assert "model" in str(exc_info.value).lower()

    def test_missing_model_id_raises_error(self):
        """Config with model section but no model_id should fail."""
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager.from_dict({
                "model": {"family": "qwen2.5"}
            })

        assert "model_id" in str(exc_info.value).lower()

    def test_missing_model_family_raises_error(self):
        """Config with model_id but no family should fail."""
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager.from_dict({
                "model": {"model_id": "Qwen/Qwen2.5-1.5B"}
            })

        assert "family" in str(exc_info.value).lower()

    def test_invalid_model_family_raises_error(self):
        """Invalid model family string should fail."""
        with pytest.raises(ConfigValidationError):
            ConfigManager.from_dict({
                "model": {
                    "model_id": "some-model",
                    "family": "gpt-4",
                }
            })

    def test_invalid_quantization_mode_raises_error(self):
        """Invalid quantization mode should fail."""
        with pytest.raises(ConfigValidationError):
            ConfigManager.from_dict({
                "model": {
                    "model_id": "some-model",
                    "family": "qwen2.5",
                    "quantization": {"mode": "int3_super"},
                }
            })

    def test_learning_rate_must_be_positive(self):
        """Negative learning rate should fail validation."""
        with pytest.raises(ConfigValidationError):
            ConfigManager.from_dict({
                "model": {"model_id": "x", "family": "qwen2.5"},
                "sft": {"learning_rate": -0.001},
            })

    def test_file_not_found_raises_error(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config("/nonexistent/path/config.yaml")

    def test_empty_yaml_raises_error(self, tmp_path: Path):
        """Empty YAML file should raise ConfigValidationError."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ConfigValidationError):
            ConfigManager.load_config(empty_file)


# ═══════════════════════════════════════════════
# TEST 3: Unknown fields (warn but don't reject)
# ═══════════════════════════════════════════════


class TestUnknownFields:
    """Test that unknown YAML fields produce warnings but don't crash."""

    def test_unknown_top_level_field_is_warned(self, tmp_path: Path, caplog):
        """Unknown fields at top level should log a warning."""
        config_dict = make_minimal_config_dict()
        config_dict["future_feature"] = {"enabled": True}

        yaml_path = tmp_path / "config_with_unknown.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        with caplog.at_level(logging.WARNING):
            config = ConfigManager.load_config(yaml_path)

        assert config.model.model_id == "Qwen/Qwen2.5-1.5B"
        assert "future_feature" in caplog.text


# ═══════════════════════════════════════════════
# TEST 4: Default values
# ═══════════════════════════════════════════════


class TestDefaults:
    """Test that default values are correct."""

    def test_sft_defaults(self):
        """SFT config should have sensible defaults."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        assert config.sft.learning_rate == 2e-4
        assert config.sft.batch_size == 4
        assert config.sft.gradient_accumulation_steps == 4
        assert config.sft.num_epochs == 3
        assert config.sft.bf16 is True
        assert config.sft.gradient_checkpointing is True

    def test_dpo_defaults(self):
        """DPO config should have sensible defaults."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        assert config.dpo.beta == 0.1
        assert config.dpo.learning_rate == 1e-5
        assert config.dpo.num_epochs == 1

    def test_lora_defaults(self):
        """LoRA config should have sensible defaults."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        assert config.model.lora.rank == 16
        assert config.model.lora.alpha == 32
        assert "q_proj" in config.model.lora.target_modules
        assert config.model.lora.dropout == 0.05

    def test_quantization_defaults_to_none(self):
        """Quantization should default to NONE (full precision)."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        assert config.model.quantization.mode == QuantizationMode.NONE

    def test_sagemaker_defaults_to_local(self):
        """Training mode should default to LOCAL."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        assert config.sagemaker.training_mode.value == "local"

    def test_seed_default(self):
        """Global seed should default to 42."""
        config = ConfigManager.from_dict(make_minimal_config_dict())

        assert config.seed == 42


# ═══════════════════════════════════════════════
# TEST 5: Enum validation
# ═══════════════════════════════════════════════


class TestEnums:
    """Test that enum values are validated correctly."""

    def test_valid_model_families(self):
        """All supported model families should be accepted."""
        for family in ["qwen2.5", "llama-3.1", "llama-3.2", "mistral", "phi-3", "smollm2"]:
            config = ConfigManager.from_dict({
                "model": {"model_id": "test", "family": family}
            })
            assert config.model.family.value == family

    def test_valid_quantization_modes(self):
        """All quantization modes should be accepted."""
        for mode in ["none", "fp16", "bf16", "int4_nf4"]:
            config = ConfigManager.from_dict({
                "model": {
                    "model_id": "test",
                    "family": "qwen2.5",
                    "quantization": {"mode": mode},
                }
            })
            assert config.model.quantization.mode.value == mode

    def test_valid_dataset_types(self):
        """Both dataset types should be accepted."""
        for dtype in ["instruction", "preference"]:
            ds = DatasetConfig(source="test", dataset_type=dtype)
            assert ds.dataset_type.value == dtype
