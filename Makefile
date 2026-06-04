# ──────────────────────────────────────────────
# Distill + Align: Common development commands
# ──────────────────────────────────────────────
# Usage:
#   make install     ← Install project + dev dependencies
#   make lint        ← Run linter (ruff)
#   make format      ← Auto-format code (black + ruff)
#   make typecheck   ← Run type checker (mypy)
#   make test        ← Run all tests
#   make test-cov    ← Run tests with coverage report
#   make check       ← Run all checks (lint + typecheck + test)
#   make clean       ← Remove build artifacts
#
#   make rejudge     ← Re-judge stored responses with a stronger judge (MODEL=gpt-4o)
#   make audit       ← Audit the gold reference answers (MODEL=gpt-4o)
#   make annotate    ← Launch the blind human-annotation session
#   make trap        ← Trap-set eval, base/SFT/DPO (ARGS="--tag llama8b ...")
#   make ece         ← Expected Calibration Error (ARGS="...")
#   make check-env   ← GPU / deps / HF-token diagnostic
# ──────────────────────────────────────────────

# Evaluation recipes use hidden key entry (read -s), which needs bash.
SHELL := /bin/bash

.PHONY: install lint format typecheck test test-cov check clean \
        rejudge audit annotate trap ece check-env

# Install the project in editable mode with dev dependencies.
# -e means "editable" — changes to source code take effect immediately
# without reinstalling. [dev] installs the optional dev dependencies.
install:
	pip install -e ".[dev]"

# Run ruff linter to catch code quality issues.
# --fix automatically fixes simple issues (like import sorting).
lint:
	ruff check src/ tests/ --fix

# Auto-format all code using black (consistent style)
# and ruff (import sorting).
format:
	black src/ tests/
	ruff check src/ tests/ --fix --select I

# Run mypy for static type checking.
# Catches type errors before runtime.
typecheck:
	mypy src/distill_align/

# Run all tests with pytest.
test:
	pytest tests/ -v

# Run tests with coverage report.
# Shows which lines of code are tested and which aren't.
test-cov:
	pytest tests/ -v --cov=src/distill_align --cov-report=term-missing

# Run ALL checks in sequence: lint, typecheck, then tests.
# This is what CI/CD will run on every push.
check: lint typecheck test

# Remove build artifacts and caches.
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ──────────────────────────────────────────────
# Evaluation / analysis (re-use stored responses; most need an API key, not a GPU)
# ──────────────────────────────────────────────

# Re-judge the 500 stored responses with a stronger judge.
# Override the judge:  make rejudge MODEL=o4-mini
rejudge:
	@MODEL=$${MODEL:-gpt-4o}; \
	if [ -z "$$OPENAI_API_KEY" ]; then \
		read -rsp "Paste your OpenAI API key (hidden), then press Enter: " OPENAI_API_KEY; echo; export OPENAI_API_KEY; \
	fi; \
	echo "Judging 500 responses with: $$MODEL"; \
	python scripts/rejudge.py --model "$$MODEL"

# Audit the 500 gold reference answers for factual errors.
# Override the auditor:  make audit MODEL=o4-mini
audit:
	@MODEL=$${MODEL:-gpt-4o}; \
	if [ -z "$$OPENAI_API_KEY" ]; then \
		read -rsp "Paste your OpenAI API key (hidden), then press Enter: " OPENAI_API_KEY; echo; export OPENAI_API_KEY; \
	fi; \
	echo "Auditing 500 reference answers with: $$MODEL"; \
	python scripts/audit_references.py --model "$$MODEL"

# Launch the blind human-annotation session (kappa validation of the LLM judge).
annotate:
	python scripts/human_annotation.py \
		--results_csv outputs/human_annotation/annotation_input.csv \
		--n_samples 100 \
		--output_dir outputs/human_annotation

# Trap-set eval: run base/SFT/DPO and report refusal vs fabrication per stage (GPU).
# Defaults to Llama-3.1-8B paths; override via ARGS, e.g.:
#   make trap ARGS="--tag mistral7b --base_model mistralai/Mistral-7B-Instruct-v0.3 \
#       --sft_adapter outputs/mistral_7b/sft_adapter --dpo_base outputs/mistral_merged \
#       --dpo_adapter outputs/mistral_7b/dpo_adapter"
trap:
	python scripts/trap_eval.py $(ARGS)

# Expected Calibration Error for one stage (defaults to the DPO Llama-3.1-8B).
# For base/sft, generate + judge those stages first, then pass their CSVs via ARGS.
ece:
	python scripts/ece.py $(ARGS)

# GPU / dependency / HF-token diagnostic for a fresh box.
check-env:
	@echo "--- GPU ---"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
	@python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__, 'peft', peft.__version__)" || echo "DEPS MISSING"
	@python -c "from huggingface_hub import HfFolder; print('HF token:', 'ok' if HfFolder.get_token() else 'MISSING')" 2>/dev/null || echo "hf check error"
