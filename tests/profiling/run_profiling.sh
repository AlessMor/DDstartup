#!/bin/bash
# Profiling script for DD Startup analysis
# Runs profiling with temporary output folders and saves results to tests/profiling/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILING_OUTPUT="$SCRIPT_DIR/results"

echo "================================================================================"
echo "DD STARTUP PROFILING SUITE"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Profiling output: $PROFILING_OUTPUT"
echo ""

# Create profiling output directory
mkdir -p "$PROFILING_OUTPUT"

# Clean up temporary directories on exit
cleanup() {
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf /tmp/ddstartup_profiling_*
    echo "✅ Cleanup complete"
}
trap cleanup EXIT

cd "$PROJECT_ROOT"

# Profile T-seeded analysis
echo "================================================================================"
echo "PROFILING T-SEEDED ANALYSIS (10,000 combinations)"
echo "================================================================================"
python -m ddstartup.utils.profiling \
    tests/fixtures/params_test.yaml \
    tests/fixtures/parametric_tseeded.yaml \
    2>&1 | tee "$PROFILING_OUTPUT/tseeded_profile.txt"

echo ""
echo "================================================================================"
echo "PROFILING LUMP ANALYSIS (10,000 combinations)"
echo "================================================================================"
python -m ddstartup.utils.profiling \
    tests/fixtures/params_test.yaml \
    tests/fixtures/parametric_lump.yaml \
    2>&1 | tee "$PROFILING_OUTPUT/lump_profile.txt"

echo ""
echo "================================================================================"
echo "PROFILING COMPLETE"
echo "================================================================================"
echo "Results saved to: $PROFILING_OUTPUT"
echo ""
ls -lh "$PROFILING_OUTPUT"
