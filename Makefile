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
# ──────────────────────────────────────────────

.PHONY: install lint format typecheck test test-cov check clean

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
