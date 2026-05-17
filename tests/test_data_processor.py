"""Unit tests for the DataProcessor class."""

import pytest
from datasets import Dataset, DatasetDict

from distill_align.config.models import DatasetType
from distill_align.data.processor import (
    DataProcessor,
    ValidationReport,
)

# ═══════════════════════════════════════════════
# FIXTURES: Create mock datasets for testing
# ═══════════════════════════════════════════════


@pytest.fixture
def processor():
    """Create a DataProcessor instance for testing."""
    return DataProcessor()


@pytest.fixture
def valid_chat_dataset():
    """Create a valid instruction dataset in chat format (messages column)."""
    return Dataset.from_dict({
        "messages": [
            [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            [
                {"role": "user", "content": "Explain recursion."},
                {"role": "assistant", "content": "Recursion is when a function calls itself."},
            ],
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ]
    })


@pytest.fixture
def valid_prompt_completion_dataset():
    """Create a valid instruction dataset in prompt/completion format."""
    return Dataset.from_dict({
        "prompt": [
            "What is machine learning?",
            "Explain gradient descent.",
            "What is a neural network?",
        ],
        "completion": [
            "Machine learning is a subset of AI...",
            "Gradient descent is an optimization algorithm...",
            "A neural network is a computational model...",
        ],
    })


@pytest.fixture
def valid_instruction_output_dataset():
    """Create a valid instruction dataset in instruction/output format."""
    return Dataset.from_dict({
        "instruction": [
            "Summarize the following text.",
            "Translate to French.",
            "Write a haiku about coding.",
        ],
        "output": [
            "The text discusses...",
            "Bonjour le monde.",
            "Code flows like water / bugs hide in the deepest pools / debug with patience",
        ],
    })


@pytest.fixture
def valid_preference_dataset():
    """Create a valid preference dataset with prompt/chosen/rejected."""
    return Dataset.from_dict({
        "prompt": [
            "What is the capital of France?",
            "How do I sort a list in Python?",
            "Explain quantum computing.",
        ],
        "chosen": [
            "The capital of France is Paris.",
            "You can use sorted() or list.sort() in Python.",
            "Quantum computing uses qubits that can exist in superposition...",
        ],
        "rejected": [
            "France is a country in Europe.",
            "Just Google it.",
            "It's complicated.",
        ],
    })


@pytest.fixture
def dataset_with_empty_fields():
    """Create a dataset with some empty/null fields for testing filtering."""
    return Dataset.from_dict({
        "prompt": [
            "Valid prompt 1",
            "",
            "Valid prompt 3",
            "Valid prompt 4",
            "   ",
        ],
        "completion": [
            "Valid completion 1",
            "Valid completion 2",
            "",
            "Valid completion 4",
            "Valid completion 5",
        ],
    })


@pytest.fixture
def preference_dataset_with_nulls():
    """Create a preference dataset with some None/empty values."""
    return Dataset.from_dict({
        "prompt": [
            "Good prompt",
            "Another prompt",
            "",
            "Valid prompt",
        ],
        "chosen": [
            "Good response",
            "",
            "Some response",
            "Valid chosen",
        ],
        "rejected": [
            "Bad response",
            "Bad response 2",
            "Bad response 3",
            "",
        ],
    })


# ═══════════════════════════════════════════════
# TEST 1: Schema validation for instruction datasets
# ═══════════════════════════════════════════════


class TestInstructionSchemaValidation:
    """Test schema validation for instruction-type datasets."""

    def test_chat_format_passes_validation(self, processor, valid_chat_dataset):
        """Dataset with 'messages' column should pass instruction validation."""
        report = processor.validate_schema(valid_chat_dataset, DatasetType.INSTRUCTION)

        assert report.passed is True
        assert report.total_records == 3
        assert report.valid_records == 3
        assert report.skipped_records == 0
        assert len(report.errors) == 0

    def test_prompt_completion_format_passes(self, processor, valid_prompt_completion_dataset):
        """Dataset with 'prompt'/'completion' columns should pass."""
        report = processor.validate_schema(
            valid_prompt_completion_dataset, DatasetType.INSTRUCTION
        )

        assert report.passed is True
        assert report.total_records == 3
        assert report.valid_records == 3

    def test_instruction_output_format_passes(self, processor, valid_instruction_output_dataset):
        """Dataset with 'instruction'/'output' columns should pass."""
        report = processor.validate_schema(
            valid_instruction_output_dataset, DatasetType.INSTRUCTION
        )

        assert report.passed is True
        assert report.total_records == 3

    def test_unknown_columns_fail_validation(self, processor):
        """Dataset with unrecognized columns should fail instruction validation."""
        bad_dataset = Dataset.from_dict({
            "text": ["Hello world", "Foo bar"],
            "label": [0, 1],
        })

        report = processor.validate_schema(bad_dataset, DatasetType.INSTRUCTION)

        assert report.passed is False
        assert report.valid_records == 0
        assert len(report.errors) > 0
        assert "don't match" in report.errors[0].lower() or "columns" in report.errors[0].lower()

    def test_empty_messages_detected(self, processor):
        """Records with empty messages list should be counted as skipped."""
        dataset = Dataset.from_dict({
            "messages": [
                [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
                [],
                [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye"}],
            ]
        })

        report = processor.validate_schema(dataset, DatasetType.INSTRUCTION)

        assert report.passed is True
        assert report.skipped_records >= 1

    def test_dataset_dict_validates_first_split(self, processor, valid_chat_dataset):
        """DatasetDict should validate the first available split."""
        dataset_dict = DatasetDict({"train": valid_chat_dataset})

        report = processor.validate_schema(dataset_dict, DatasetType.INSTRUCTION)

        assert report.passed is True
        assert report.total_records == 3


# ═══════════════════════════════════════════════
# TEST 2: Schema validation for preference datasets
# ═══════════════════════════════════════════════


class TestPreferenceSchemaValidation:
    """Test schema validation for preference-type datasets."""

    def test_valid_preference_dataset_passes(self, processor, valid_preference_dataset):
        """Dataset with prompt/chosen/rejected should pass preference validation."""
        report = processor.validate_schema(valid_preference_dataset, DatasetType.PREFERENCE)

        assert report.passed is True
        assert report.total_records == 3
        assert report.valid_records == 3
        assert report.skipped_records == 0

    def test_missing_chosen_column_fails(self, processor):
        """Preference dataset without 'chosen' column should fail."""
        bad_dataset = Dataset.from_dict({
            "prompt": ["What is AI?"],
            "rejected": ["I don't know."],
        })

        report = processor.validate_schema(bad_dataset, DatasetType.PREFERENCE)

        assert report.passed is False
        assert "chosen" in report.errors[0].lower()

    def test_missing_rejected_column_fails(self, processor):
        """Preference dataset without 'rejected' column should fail."""
        bad_dataset = Dataset.from_dict({
            "prompt": ["What is AI?"],
            "chosen": ["AI is artificial intelligence."],
        })

        report = processor.validate_schema(bad_dataset, DatasetType.PREFERENCE)

        assert report.passed is False
        assert "rejected" in report.errors[0].lower()

    def test_missing_prompt_column_fails(self, processor):
        """Preference dataset without 'prompt' column should fail."""
        bad_dataset = Dataset.from_dict({
            "chosen": ["Good answer"],
            "rejected": ["Bad answer"],
        })

        report = processor.validate_schema(bad_dataset, DatasetType.PREFERENCE)

        assert report.passed is False
        assert "prompt" in report.errors[0].lower()

    def test_empty_strings_detected_in_preference(self, processor, preference_dataset_with_nulls):
        """Empty strings in preference data should be counted as skipped."""
        report = processor.validate_schema(
            preference_dataset_with_nulls, DatasetType.PREFERENCE
        )

        assert report.passed is True
        assert report.skipped_records >= 1


# ═══════════════════════════════════════════════
# TEST 3: Dataset splitting
# ═══════════════════════════════════════════════


class TestDatasetSplitting:
    """Test dataset splitting functionality."""

    def test_default_split_ratios(self, processor):
        """Default 90/5/5 split should produce correct proportions."""
        dataset = Dataset.from_dict({
            "prompt": [f"Prompt {i}" for i in range(100)],
            "completion": [f"Completion {i}" for i in range(100)],
        })

        splits = processor.split(dataset)

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

        assert len(splits["train"]) >= 88
        assert len(splits["train"]) <= 92
        assert len(splits["validation"]) >= 3
        assert len(splits["validation"]) <= 7
        assert len(splits["test"]) >= 3
        assert len(splits["test"]) <= 7

        total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total == 100

    def test_custom_split_ratios(self, processor):
        """Custom split ratios should be respected."""
        dataset = Dataset.from_dict({
            "prompt": [f"Prompt {i}" for i in range(200)],
            "completion": [f"Completion {i}" for i in range(200)],
        })

        splits = processor.split(
            dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        assert len(splits["train"]) >= 155
        assert len(splits["train"]) <= 165
        assert len(splits["validation"]) >= 15
        assert len(splits["validation"]) <= 25
        assert len(splits["test"]) >= 15
        assert len(splits["test"]) <= 25

    def test_invalid_ratios_raise_error(self, processor):
        """Ratios that don't sum to 1.0 should raise ValueError."""
        dataset = Dataset.from_dict({
            "prompt": ["test"] * 10,
            "completion": ["test"] * 10,
        })

        with pytest.raises(ValueError, match="sum to 1.0"):
            processor.split(dataset, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_split_reproducibility_with_seed(self, processor):
        """Same seed should produce identical splits."""
        dataset = Dataset.from_dict({
            "prompt": [f"Prompt {i}" for i in range(50)],
            "completion": [f"Completion {i}" for i in range(50)],
        })

        splits_1 = processor.split(dataset, seed=42)
        splits_2 = processor.split(dataset, seed=42)

        assert splits_1["train"]["prompt"] == splits_2["train"]["prompt"]
        assert splits_1["test"]["prompt"] == splits_2["test"]["prompt"]

    def test_different_seeds_produce_different_splits(self, processor):
        """Different seeds should produce different splits."""
        dataset = Dataset.from_dict({
            "prompt": [f"Prompt {i}" for i in range(50)],
            "completion": [f"Completion {i}" for i in range(50)],
        })

        splits_1 = processor.split(dataset, seed=42)
        splits_2 = processor.split(dataset, seed=123)

        assert splits_1["train"]["prompt"] != splits_2["train"]["prompt"]


# ═══════════════════════════════════════════════
# TEST 4: Malformed record handling
# ═══════════════════════════════════════════════


class TestMalformedRecordHandling:
    """Test that malformed records are handled gracefully."""

    def test_filter_removes_empty_prompts(self, processor, dataset_with_empty_fields):
        """filter_invalid_records should remove records with empty prompts."""
        filtered = processor.filter_invalid_records(
            dataset_with_empty_fields, DatasetType.INSTRUCTION
        )

        assert len(filtered) < len(dataset_with_empty_fields)
        assert len(filtered) <= 3

    def test_filter_removes_empty_preference_fields(self, processor, preference_dataset_with_nulls):
        """filter_invalid_records should remove preference records with empty fields."""
        filtered = processor.filter_invalid_records(
            preference_dataset_with_nulls, DatasetType.PREFERENCE
        )

        assert len(filtered) < len(preference_dataset_with_nulls)
        assert len(filtered) == 1

    def test_filter_preserves_valid_records(self, processor, valid_preference_dataset):
        """filter_invalid_records should not remove any valid records."""
        filtered = processor.filter_invalid_records(
            valid_preference_dataset, DatasetType.PREFERENCE
        )

        assert len(filtered) == len(valid_preference_dataset)

    def test_validation_report_repr(self):
        """ValidationReport __repr__ should show status clearly."""
        report = ValidationReport(
            passed=True,
            total_records=100,
            valid_records=95,
            skipped_records=5,
            errors=[],
        )

        repr_str = repr(report)
        assert "PASSED" in repr_str
        assert "95" in repr_str
        assert "100" in repr_str

    def test_failed_validation_report_repr(self):
        """Failed ValidationReport should show FAILED status."""
        report = ValidationReport(
            passed=False,
            total_records=100,
            valid_records=0,
            skipped_records=100,
            errors=["No valid records found"],
        )

        repr_str = repr(report)
        assert "FAILED" in repr_str


# ═══════════════════════════════════════════════
# TEST 5: Dataset statistics
# ═══════════════════════════════════════════════


class TestDatasetStatistics:
    """Test dataset statistics computation."""

    def test_statistics_for_single_dataset(self, processor, valid_preference_dataset):
        """get_dataset_statistics should return correct stats for a Dataset."""
        stats = processor.get_dataset_statistics(valid_preference_dataset)

        assert stats["num_records"] == 3
        assert "prompt" in stats["columns"]
        assert "chosen" in stats["columns"]
        assert "rejected" in stats["columns"]

    def test_statistics_for_dataset_dict(self, processor, valid_chat_dataset):
        """get_dataset_statistics should handle DatasetDict with multiple splits."""
        dataset_dict = DatasetDict({
            "train": valid_chat_dataset,
            "test": valid_chat_dataset,
        })

        stats = processor.get_dataset_statistics(dataset_dict)

        assert "splits" in stats
        assert "train" in stats["splits"]
        assert "test" in stats["splits"]
        assert stats["total_records"] == 6
        assert stats["splits"]["train"]["num_records"] == 3
