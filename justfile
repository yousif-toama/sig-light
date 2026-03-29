# sig-light development commands

# Run all code quality checks
check: lint format-check typecheck

# Run linter
lint:
    uv run ruff check

# Check formatting
format-check:
    uv run ruff format --check

# Auto-format code
format:
    uv run ruff format

# Run type checker
typecheck:
    uv run ty check

# Run all tests
test:
    uv run pytest

# Run tests with coverage report
test-cov:
    uv run pytest --cov=src --cov-report=term-missing

# Run benchmark (install iisignature first for comparison)
bench:
    uv run python scripts/benchmark.py

# Install dependencies
sync:
    uv sync
