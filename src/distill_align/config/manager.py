"""Configuration Manager: loads, validates, and saves YAML config files."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from distill_align.config.models import PipelineConfig

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when a config file has missing or invalid fields."""

    pass


class ConfigManager:
    """Loads, validates, serializes, and deserializes pipeline configurations."""

    @staticmethod
    def load_config(config_path: str | Path) -> PipelineConfig:
        """Load a YAML config file and return a validated PipelineConfig."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            raw_data = yaml.safe_load(f)

        if raw_data is None:
            raise ConfigValidationError(f"Config file is empty: {config_path}")

        # Warn on unknown fields for forward compatibility
        known_fields = set(PipelineConfig.model_fields.keys())
        unknown_fields = set(raw_data.keys()) - known_fields
        if unknown_fields:
            logger.warning(
                f"Unknown fields in config (will be ignored): {sorted(unknown_fields)}"
            )
            for field_name in unknown_fields:
                del raw_data[field_name]

        try:
            config = PipelineConfig(**raw_data)
        except ValidationError as e:
            raise ConfigValidationError(
                f"Config validation failed for {config_path}:\n{e}"
            ) from e

        return config

    @staticmethod
    def save_config(config: PipelineConfig, output_path: str | Path) -> None:
        """Serialize a PipelineConfig to a YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(mode="json")

        with open(output_path, "w") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @staticmethod
    def from_dict(data: dict) -> PipelineConfig:
        """Create a PipelineConfig from a dictionary."""
        try:
            return PipelineConfig(**data)
        except ValidationError as e:
            raise ConfigValidationError(f"Config validation failed:\n{e}") from e
