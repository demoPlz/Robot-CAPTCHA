#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Convert a DCP (data collection policy) dataset into multiple variants
# with different numbers of approved actions per critical state.
#
# Naming convention:
#   Input:  <path>/<name>_dcp   (the "_dcp" suffix is stripped)
#   Output: <path>/<name>_k1, <name>_k2, <name>_k5, <name>_k10
#
# Usage:
#   ./scripts/convert_dcp_variants.sh <dataset_path> [k values...]
#
# Examples:
#   # Default: generates k1, k2, k5, k10
#   ./scripts/convert_dcp_variants.sh ~/.cache/huggingface/lerobot/pour/pour_15_mar6_dcp
#
#   # Custom k values
#   ./scripts/convert_dcp_variants.sh ~/.cache/huggingface/lerobot/pour/pour_15_mar6_dcp 1 3 5
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_path> [k values...]"
    echo ""
    echo "  <dataset_path>  Full path to the DCP dataset (must end in _dcp)"
    echo "  [k values]      Optional list of k values (default: 1 2 5 10)"
    echo ""
    echo "Examples:"
    echo "  $0 ~/.cache/huggingface/lerobot/pour/pour_15_mar6_dcp"
    echo "  $0 ~/.cache/huggingface/lerobot/pour/pour_15_mar6_dcp 1 3 5"
    exit 1
fi

SOURCE_PATH="$(realpath "$1")"
shift

# Default k values
if [[ $# -gt 0 ]]; then
    K_VALUES=("$@")
else
    K_VALUES=(1 2 5 10)
fi

# Strip trailing _dcp to get the base path
if [[ "$SOURCE_PATH" == *_dcp ]]; then
    BASE_PATH="${SOURCE_PATH%_dcp}"
else
    echo "WARNING: Source path '$SOURCE_PATH' doesn't end with '_dcp'."
    echo "         Using it as-is for the base name."
    BASE_PATH="$SOURCE_PATH"
fi

echo "============================================"
echo "DCP Dataset Variant Converter"
echo "============================================"
echo "  Source:      $SOURCE_PATH"
echo "  Base path:   $BASE_PATH"
echo "  K values:    ${K_VALUES[*]}"
echo "============================================"
echo ""

# ---- Activate the right Python environment ----
# Use the lerobot_trossen package from the project's external folder
export PYTHONPATH="$PROJECT_ROOT/external/lerobot_trossen:${PYTHONPATH:-}"

FAILED=()
SUCCEEDED=()

for K in "${K_VALUES[@]}"; do
    TARGET_PATH="${BASE_PATH}_k${K}"

    echo "────────────────────────────────────────────"
    echo "Converting: k=$K → $TARGET_PATH"
    echo "────────────────────────────────────────────"

    # Check if target already exists
    if [[ -d "$TARGET_PATH" ]]; then
        echo "  ⚠  Target already exists: $TARGET_PATH"
        echo "     Skipping. Delete it first if you want to regenerate."
        echo ""
        SUCCEEDED+=("k${K} (already existed)")
        continue
    fi

    # Build the conversion command — use full paths as repo IDs (no --root)
    if [[ "$K" -eq 1 ]]; then
        # k=1: use --use-final-action-only (the single executed action)
        CMD=(
            python -m lerobot.scripts.convert_multi_action_dataset
            --source-repo-id "$SOURCE_PATH"
            --target-repo-id "$TARGET_PATH"
            --use-final-action-only
        )
    else
        # k>1: use --approved-actions-per-critical-state
        CMD=(
            python -m lerobot.scripts.convert_multi_action_dataset
            --source-repo-id "$SOURCE_PATH"
            --target-repo-id "$TARGET_PATH"
            --approved-actions-per-critical-state "$K"
        )
    fi

    echo "  Running: ${CMD[*]}"
    echo ""

    if "${CMD[@]}"; then
        echo ""
        echo "  ✓ k=$K complete → $TARGET_PATH"
        SUCCEEDED+=("k${K}")
    else
        echo ""
        echo "  ✗ k=$K FAILED"
        FAILED+=("k${K}")
    fi
    echo ""
done

echo "============================================"
echo "Summary"
echo "============================================"
if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo "  Succeeded: ${SUCCEEDED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED:    ${FAILED[*]}"
    exit 1
fi
echo ""
echo "Output datasets:"
for K in "${K_VALUES[@]}"; do
    echo "  ${BASE_PATH}_k${K}"
done
echo "============================================"
