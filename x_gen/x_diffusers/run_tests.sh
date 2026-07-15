#!/bin/bash
# Run unit tests for x_diffusers

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run pytest with coverage
cd "$SCRIPT_DIR"

echo "Running x_diffusers unit tests..."
pytest tests/ \
    --cov=x_diffusers \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    -v \
    "$@"
