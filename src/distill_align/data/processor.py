"""DataProcessor: loads, validates, and preprocesses datasets for training."""

import logging

from datasets import Dataset, DatasetDict
from datasets import load_dataset as hf_load_dataset

from distill_align.config.models import DatasetConfig, DatasetType

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when a dataset fails schema validation."""

    pass


class ValidationReport:
    """Report from dataset schema validation.

    Attributes:
        passed: Whether validation passed overall.
        total_records: Total number of records in the dataset.
        valid_records: Number of records that passed validation.
        skipped_records: Number of records that failed validation.
        errors: List of error descriptions.
    """

    def __init__(
        self,
        passed: bool,
        total_records: int,
        valid_records: int,
        skipped_records: int,
        errors: list[str],
    ):
        self.passed = passed
        self.total_records = total_records
        self.valid_records = valid_records
        self.skipped_records = skipped_records
        self.errors = errors

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ValidationReport({status}: {self.valid_records}/{self.total_records} valid, "
            f"{self.skipped_records} skipped, {len(self.errors)} errors)"
        )


# ═══════════════════════════════════════════════
# EXPECTED DATASET FORMATS
# ═══════════════════════════════════════════════

INSTRUCTION_FORMATS = {
    "chat": {"required": ["messages"]},
    "prompt_completion": {"required": ["prompt", "completion"]},
    "input_output": {"required": ["input", "output"]},
    "instruction_output": {"required": ["instruction", "output"]},
}

PREFERENCE_REQUIRED_FIELDS = ["prompt", "chosen", "rejected"]


class DataProcessor:
    """Loads, validates, and preprocesses datasets for the training pipeline."""

    def load_dataset(self, config: DatasetConfig) -> Dataset | DatasetDict:
        """Load a dataset from HuggingFace Hub or a local path.

        Args:
            config: Dataset configuration specifying source, type, etc.

        Returns:
            A HuggingFace Dataset or DatasetDict object.

        Raises:
            FileNotFoundError: If local path doesn't exist.
            Exception: If HuggingFace Hub download fails.
        """
        source = config.source
        logger.info(f"Loading dataset from: {source}")

        if source.startswith("s3://"):
            logger.info("S3 source detected — downloading via boto3")
            raise NotImplementedError(
                "S3 loading will be implemented in SageMaker integration. "
                "Use HuggingFace Hub ID or local path for now."
            )

        try:
            dataset = hf_load_dataset(source)
            logger.info(
                f"Loaded dataset: {source} "
                f"(splits: {list(dataset.keys()) if isinstance(dataset, DatasetDict) else 'single'})"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset '{source}': {e}")
            raise

        return dataset

    def validate_schema(
        self, dataset: Dataset | DatasetDict, dataset_type: DatasetType
    ) -> ValidationReport:
        """Validate that a dataset has the expected schema.

        Args:
            dataset: The loaded dataset to validate.
            dataset_type: Expected type (INSTRUCTION or PREFERENCE).

        Returns:
            ValidationReport with pass/fail status and details.
        """
        if isinstance(dataset, DatasetDict):
            split_name = list(dataset.keys())[0]
            ds = dataset[split_name]
            logger.info(f"Validating schema on split: '{split_name}'")
        else:
            ds = dataset

        columns = set(ds.column_names)
        total_records = len(ds)
        errors = []

        if dataset_type == DatasetType.INSTRUCTION:
            return self._validate_instruction_schema(ds, columns, total_records)
        elif dataset_type == DatasetType.PREFERENCE:
            return self._validate_preference_schema(ds, columns, total_records)
        else:
            errors.append(f"Unknown dataset type: {dataset_type}")
            return ValidationReport(
                passed=False,
                total_records=total_records,
                valid_records=0,
                skipped_records=total_records,
                errors=errors,
            )

    def _validate_instruction_schema(
        self, ds: Dataset, columns: set[str], total_records: int
    ) -> ValidationReport:
        """Validate an instruction dataset's schema."""
        errors = []

        detected_format = None
        for fmt_name, fmt_spec in INSTRUCTION_FORMATS.items():
            if all(field in columns for field in fmt_spec["required"]):
                detected_format = fmt_name
                break

        if detected_format is None:
            errors.append(
                f"Dataset columns {sorted(columns)} don't match any known instruction format. "
                f"Expected one of: {list(INSTRUCTION_FORMATS.keys())}"
            )
            return ValidationReport(
                passed=False,
                total_records=total_records,
                valid_records=0,
                skipped_records=total_records,
                errors=errors,
            )

        logger.info(f"Detected instruction format: '{detected_format}'")

        skipped = 0
        required_fields = INSTRUCTION_FORMATS[detected_format]["required"]

        for i in range(min(total_records, 1000)):
            record = ds[i]
            for field in required_fields:
                value = record.get(field)
                if value is None or (isinstance(value, (str, list)) and len(value) == 0):
                    skipped += 1
                    break

        sample_size = min(total_records, 1000)
        skip_rate = skipped / sample_size if sample_size > 0 else 0
        estimated_skipped = int(skip_rate * total_records)
        valid_records = total_records - estimated_skipped

        if estimated_skipped > 0:
            logger.warning(
                f"Estimated {estimated_skipped}/{total_records} records have empty/null fields "
                f"(sampled {sample_size} records, {skipped} had issues)"
            )

        passed = valid_records > 0
        if not passed:
            errors.append("No valid records found in dataset")

        return ValidationReport(
            passed=passed,
            total_records=total_records,
            valid_records=valid_records,
            skipped_records=estimated_skipped,
            errors=errors,
        )

    def _validate_preference_schema(
        self, ds: Dataset, columns: set[str], total_records: int
    ) -> ValidationReport:
        """Validate a preference dataset's schema."""
        errors = []

        missing = set(PREFERENCE_REQUIRED_FIELDS) - columns
        if missing:
            errors.append(
                f"Missing required columns for preference dataset: {sorted(missing)}. "
                f"Available columns: {sorted(columns)}"
            )
            return ValidationReport(
                passed=False,
                total_records=total_records,
                valid_records=0,
                skipped_records=total_records,
                errors=errors,
            )

        logger.info("Preference dataset schema validated: prompt/chosen/rejected found")

        skipped = 0
        for i in range(min(total_records, 1000)):
            record = ds[i]
            for field in PREFERENCE_REQUIRED_FIELDS:
                value = record.get(field)
                if value is None or (isinstance(value, str) and len(value.strip()) == 0):
                    skipped += 1
                    break

        sample_size = min(total_records, 1000)
        skip_rate = skipped / sample_size if sample_size > 0 else 0
        estimated_skipped = int(skip_rate * total_records)
        valid_records = total_records - estimated_skipped

        if estimated_skipped > 0:
            logger.warning(
                f"Estimated {estimated_skipped}/{total_records} records have empty/null fields"
            )

        passed = valid_records > 0
        if not passed:
            errors.append("No valid records found in dataset")

        return ValidationReport(
            passed=passed,
            total_records=total_records,
            valid_records=valid_records,
            skipped_records=estimated_skipped,
            errors=errors,
        )

    def filter_invalid_records(
        self, dataset: Dataset, dataset_type: DatasetType
    ) -> Dataset:
        """Remove records with missing or empty required fields.

        Args:
            dataset: Dataset to filter.
            dataset_type: Type determines which fields are required.

        Returns:
            Filtered dataset with only valid records.
        """
        original_count = len(dataset)

        if dataset_type == DatasetType.INSTRUCTION:
            columns = set(dataset.column_names)
            for _fmt_name, fmt_spec in INSTRUCTION_FORMATS.items():
                if all(f in columns for f in fmt_spec["required"]):
                    required_fields = fmt_spec["required"]
                    break
            else:
                logger.error("Cannot filter: unknown instruction format")
                return dataset

        elif dataset_type == DatasetType.PREFERENCE:
            required_fields = PREFERENCE_REQUIRED_FIELDS
        else:
            return dataset

        def is_valid(record):
            for field in required_fields:
                value = record.get(field)
                if value is None:
                    return False
                if isinstance(value, str) and len(value.strip()) == 0:
                    return False
                if isinstance(value, list) and len(value) == 0:
                    return False
            return True

        filtered = dataset.filter(is_valid)
        removed_count = original_count - len(filtered)

        if removed_count > 0:
            logger.warning(
                f"Filtered out {removed_count}/{original_count} invalid records "
                f"({removed_count/original_count*100:.1f}% removed)"
            )
        else:
            logger.info(f"All {original_count} records passed validation")

        return filtered


    # ═══════════════════════════════════════════════
    # TOKENIZATION
    # ═══════════════════════════════════════════════

    def tokenize(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 2048,
        dataset_type: DatasetType = DatasetType.INSTRUCTION,
    ) -> Dataset:
        """Tokenize a dataset using the model's tokenizer and chat template.

        Args:
            dataset: Validated dataset to tokenize.
            tokenizer: HuggingFace tokenizer (with chat template).
            max_length: Maximum sequence length (truncate longer sequences).
            dataset_type: Determines how to format the text before tokenizing.

        Returns:
            Dataset with tokenized columns added (input_ids, attention_mask).
        """
        logger.info(f"Tokenizing dataset ({len(dataset)} records, max_length={max_length})")

        if dataset_type == DatasetType.INSTRUCTION:
            tokenized = self._tokenize_instruction(dataset, tokenizer, max_length)
        elif dataset_type == DatasetType.PREFERENCE:
            tokenized = self._tokenize_preference(dataset, tokenizer, max_length)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        logger.info(f"Tokenization complete: {len(tokenized)} records")
        return tokenized

    def _tokenize_instruction(
        self, dataset: Dataset, tokenizer, max_length: int
    ) -> Dataset:
        """Tokenize an instruction dataset using chat templates."""
        columns = set(dataset.column_names)

        def format_and_tokenize(examples):
            texts = []

            if "messages" in columns:
                for messages in examples["messages"]:
                    if hasattr(tokenizer, "apply_chat_template"):
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                    else:
                        text = "\n".join(
                            f"{m['role']}: {m['content']}" for m in messages
                        )
                    texts.append(text)

            elif "prompt" in columns and "completion" in columns:
                for prompt, completion in zip(examples["prompt"], examples["completion"]):
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                    if hasattr(tokenizer, "apply_chat_template"):
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                    else:
                        text = f"User: {prompt}\nAssistant: {completion}"
                    texts.append(text)

            elif "instruction" in columns and "output" in columns:
                for instruction, output in zip(examples["instruction"], examples["output"]):
                    messages = [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": output},
                    ]
                    if hasattr(tokenizer, "apply_chat_template"):
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                    else:
                        text = f"User: {instruction}\nAssistant: {output}"
                    texts.append(text)

            elif "input" in columns and "output" in columns:
                for inp, out in zip(examples["input"], examples["output"]):
                    messages = [
                        {"role": "user", "content": inp},
                        {"role": "assistant", "content": out},
                    ]
                    if hasattr(tokenizer, "apply_chat_template"):
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                    else:
                        text = f"User: {inp}\nAssistant: {out}"
                    texts.append(text)
            else:
                raise ValueError(f"Unknown instruction format. Columns: {columns}")

            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

            tokenized["text"] = texts
            return tokenized

        tokenized_dataset = dataset.map(
            format_and_tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing instruction data",
        )

        return tokenized_dataset

    def _tokenize_preference(
        self, dataset: Dataset, tokenizer, max_length: int
    ) -> Dataset:
        """Tokenize a preference dataset for DPO/RLHF training."""

        def format_preference(examples):
            formatted_prompts = []
            formatted_chosen = []
            formatted_rejected = []

            for prompt, chosen, rejected in zip(
                examples["prompt"], examples["chosen"], examples["rejected"]
            ):
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt_messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_chosen.append(chosen if isinstance(chosen, str) else str(chosen))
                    formatted_rejected.append(rejected if isinstance(rejected, str) else str(rejected))
                    formatted_prompts.append(formatted_prompt)
                else:
                    formatted_prompts.append(f"User: {prompt}\nAssistant: ")
                    formatted_chosen.append(chosen if isinstance(chosen, str) else str(chosen))
                    formatted_rejected.append(rejected if isinstance(rejected, str) else str(rejected))

            return {
                "prompt": formatted_prompts,
                "chosen": formatted_chosen,
                "rejected": formatted_rejected,
            }

        formatted_dataset = dataset.map(
            format_preference,
            batched=True,
            desc="Formatting preference data",
        )

        return formatted_dataset


    # ═══════════════════════════════════════════════
    # DATASET SPLITTING
    # ═══════════════════════════════════════════════

    def split(
        self,
        dataset: Dataset,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
    ) -> DatasetDict:
        """Split a dataset into train/validation/test partitions.

        Args:
            dataset: Dataset to split.
            train_ratio: Fraction for training (default 90%).
            val_ratio: Fraction for validation (default 5%).
            test_ratio: Fraction for testing (default 5%).
            seed: Random seed for reproducibility.

        Returns:
            DatasetDict with 'train', 'validation', 'test' splits.

        Raises:
            ValueError: If ratios don't sum to 1.0 (within tolerance).
        """
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.3f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        total_records = len(dataset)
        logger.info(
            f"Splitting {total_records} records: "
            f"train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}"
        )

        train_val_test = dataset.train_test_split(
            test_size=test_ratio,
            seed=seed,
        )

        remaining_val_ratio = val_ratio / (train_ratio + val_ratio)

        train_val = train_val_test["train"].train_test_split(
            test_size=remaining_val_ratio,
            seed=seed,
        )

        splits = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_val_test["test"],
        })

        logger.info(
            f"Split complete: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, "
            f"test={len(splits['test'])}"
        )

        return splits

    def get_dataset_statistics(self, dataset: Dataset | DatasetDict) -> dict:
        """Compute dataset statistics for logging to experiment tracker.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Dictionary of statistics.
        """
        if isinstance(dataset, DatasetDict):
            stats = {
                "splits": {},
                "total_records": 0,
            }
            for split_name, split_ds in dataset.items():
                split_stats = {
                    "num_records": len(split_ds),
                    "columns": split_ds.column_names,
                }
                if "input_ids" in split_ds.column_names:
                    lengths = [len(ids) for ids in split_ds["input_ids"]]
                    split_stats["token_length_mean"] = sum(lengths) / len(lengths) if lengths else 0
                    split_stats["token_length_min"] = min(lengths) if lengths else 0
                    split_stats["token_length_max"] = max(lengths) if lengths else 0

                stats["splits"][split_name] = split_stats
                stats["total_records"] += len(split_ds)

            return stats

        else:
            stats = {
                "num_records": len(dataset),
                "columns": dataset.column_names,
            }
            if "input_ids" in dataset.column_names:
                lengths = [len(ids) for ids in dataset["input_ids"]]
                stats["token_length_mean"] = sum(lengths) / len(lengths) if lengths else 0
                stats["token_length_min"] = min(lengths) if lengths else 0
                stats["token_length_max"] = max(lengths) if lengths else 0

            return stats
