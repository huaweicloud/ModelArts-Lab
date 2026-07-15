#!/bin/bash
# Unit test execution script for x_diffusers on Ascend NPU
# Generated: $(date +%Y-%m-%d\ %H:%M:%S)

set -e

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/unit_test_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "x_diffusers Unit Test Suite"
echo "========================================"
echo "Script Dir: ${SCRIPT_DIR}"
echo "Log File:   ${LOG_FILE}"
echo "Start Time: $(date)"
echo "========================================"

# ============================================================================
# Environment Setup
# ============================================================================
cd "${SCRIPT_DIR}"

# Activate environment if needed
if [ -d "/opt/conda/envs/pytorch" ]; then
    source /opt/conda/envs/pytorch/bin/activate
fi

# Add project to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ============================================================================
# Run Tests
# ============================================================================
echo ""
echo "[Step 1] Checking test dependencies..."
pip list | grep -E "pytest|pytest-cov|pytest-mock" || {
    echo "Installing test dependencies..."
    pip install pytest>=7.0.0 pytest-cov>=4.0.0 pytest-mock>=3.10.0
}

echo ""
echo "[Step 2] Running unit tests..."
echo ""

# Run pytest with verbose output
pytest tests/ \
    -v \
    --tb=short \
    --strict-markers \
    -m "not slow and not integration" \
    --color=yes \
    2>&1 | tee "${LOG_FILE}"

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
echo "Test Execution Summary"
echo "========================================"
echo "Exit Code:    ${TEST_EXIT_CODE}"
echo "End Time:     $(date)"
echo "Log File:     ${LOG_FILE}"

if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "Status:       PASSED ✓"
else
    echo "Status:       FAILED ✗"
fi
echo "========================================"

exit ${TEST_EXIT_CODE}
