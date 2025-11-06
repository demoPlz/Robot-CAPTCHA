#!/bin/bash
# Format Python code in the repository with black, isort, and docformatter

set -e

echo "Formatting Python code..."

# Backend Python files
BACKEND_DIR="backend"

# Format with isort (import sorting) first
echo "Sorting imports with isort..."
isort "$BACKEND_DIR" --profile black

# Format with black (code formatting)
echo "Formatting code with black..."
black "$BACKEND_DIR" --line-length 120

# Format docstrings with docformatter
echo "Formatting docstrings with docformatter..."
docformatter --in-place --recursive "$BACKEND_DIR" \
    --wrap-summaries 120 \
    --wrap-descriptions 120 \
    --blank

echo "✅ Formatting complete!"
