#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Create mixed (crowd + trajectory) dataset variants for multiple k values.
#
# This script takes pre-existing crowd k-variant datasets (output of
# convert_dcp_variants.sh) and a trajectory/expert dataset, then merges
# each crowd variant with the trajectory dataset using
# merge_flexible_datasets.py.
#
# Naming convention:
#   Crowd input:  <base>_k1, <base>_k2, <base>_k5, <base>_k10
#   Expert input: <expert_path>
#   Output:       <base>_mixed_k1, <base>_mixed_k2, ...
#
# Usage:
#   ./scripts/create_mixed_variants.sh <crowd_base_path> <expert_path> [k values...]
#
# Examples:
#   # Default: generates mixed_k1, mixed_k2, mixed_k5, mixed_k10
#   ./scripts/create_mixed_variants.sh \
#     ~/.cache/huggingface/lerobot/insertion/insertion_c20_mar13 \
#     ~/.cache/huggingface/lerobot/insertion/teleop_100_mar14
#
#   # Custom k values
#   ./scripts/create_mixed_variants.sh \
#     ~/.cache/huggingface/lerobot/insertion/insertion_c20_mar13 \
#     ~/.cache/huggingface/lerobot/insertion/teleop_100_mar14 \
#     1 3 5
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <crowd_base_path> <expert_path> [k values...]"
    echo ""
    echo "  <crowd_base_path>  Base path for crowd k-variants (without _kN suffix)"
    echo "  <expert_path>      Full path to the trajectory/expert dataset"
    echo "  [k values]         Optional list of k values (default: 1 2 5 10)"
    echo ""
    echo "The script expects crowd variants to already exist at <crowd_base_path>_k1, etc."
    echo "Output datasets will be created at <crowd_base_path>_mixed_k1, etc."
    echo ""
    echo "Examples:"
    echo "  $0 ~/.cache/huggingface/lerobot/insertion/c20_mar13 \\"
    echo "     ~/.cache/huggingface/lerobot/insertion/teleop_100_mar14"
    echo ""
    echo "  $0 ~/.cache/huggingface/lerobot/insertion/c20_mar13 \\"
    echo "     ~/.cache/huggingface/lerobot/insertion/teleop_100_mar14 1 3 5"
    exit 1
fi

CROWD_BASE="$(realpath "$1")"
EXPERT_PATH="$(realpath "$2")"
shift 2

# Default k values
if [[ $# -gt 0 ]]; then
    K_VALUES=("$@")
else
    K_VALUES=(1 2 5 10)
fi

# Validate expert dataset exists
if [[ ! -d "$EXPERT_PATH" ]]; then
    echo "ERROR: Expert/trajectory dataset not found: $EXPERT_PATH"
    exit 1
fi

echo "============================================"
echo "Mixed Dataset Variant Creator"
echo "============================================"
echo "  Crowd base:  $CROWD_BASE"
echo "  Expert:      $EXPERT_PATH"
echo "  K values:    ${K_VALUES[*]}"
echo "============================================"
echo ""

# ---- Activate the right Python environment ----
export PYTHONPATH="$PROJECT_ROOT/external/lerobot_trossen:${PYTHONPATH:-}"

FAILED=()
SUCCEEDED=()

for K in "${K_VALUES[@]}"; do
    CROWD_K_PATH="${CROWD_BASE}_k${K}"
    TARGET_PATH="${CROWD_BASE}_mixed_k${K}"

    echo "────────────────────────────────────────────"
    echo "Merging: k=$K → $TARGET_PATH"
    echo "────────────────────────────────────────────"

    # Validate crowd k-variant exists
    if [[ ! -d "$CROWD_K_PATH" ]]; then
        echo "  ✗ Crowd variant not found: $CROWD_K_PATH"
        echo "    Run convert_dcp_variants.sh first to generate it."
        FAILED+=("k${K} (crowd variant missing)")
        echo ""
        continue
    fi

    # Skip if target already exists
    if [[ -d "$TARGET_PATH" ]]; then
        echo "  ⚠  Target already exists: $TARGET_PATH"
        echo "     Skipping. Delete it first if you want to regenerate."
        echo ""
        SUCCEEDED+=("k${K} (already existed)")
        continue
    fi

    # Run merge_flexible_datasets.py
    CMD=(
        python "$SCRIPT_DIR/merge_flexible_datasets.py"
        --crowd-repo-id "$CROWD_K_PATH"
        --expert-repo-id "$EXPERT_PATH"
        --target-repo-id "$TARGET_PATH"
    )

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
    echo "  ${CROWD_BASE}_mixed_k${K}"
done
echo "============================================"
