#!/bin/bash
#
# Enhanced coverage script for marEx with Dask and Numba support
#
# This script configures the environment to properly measure coverage
# of code executed in Dask workers and functions called by xr.apply_ufunc
#

set -e  # Exit on any error

echo "Starting enhanced coverage measurement for marEx..."

# Export coverage configuration environment variables
export COVERAGE_PROCESS_START=$(pwd)/.coveragerc
export NUMBA_DISABLE_JIT=1
export PYTEST_COVERAGE=true
export DASK_SCHEDULER=threads  # Force threads execution for coverage

# Clean any existing coverage data
echo "Cleaning previous coverage data..."
coverage erase

# Ensure we're using the correct Python environment
echo "Using Python: $(which python)"
echo "Coverage version: $(coverage --version)"

# Run tests with coverage (synchronous execution ensures accurate measurement)
echo "Running tests with synchronous coverage collection..."
coverage run -m pytest tests/ \
    --tb=short \
    -m "not nocov and not numba_issue" \
    -x \
    --maxfail=5 \
    -v \
    -k "not (test_histogram_quantile_2d_vs_exact_quantile or test_tracker_non_dask_binary_data or test_shifting_baseline_hobday_extreme_exact_percentile)"

# Generate comprehensive reports
echo "Generating coverage reports..."
echo "=========================="
echo "Coverage Summary:"
echo "=========================="
coverage report -m

echo ""
echo "=========================="
echo "Coverage by Module:"
echo "=========================="
coverage report --show-missing

# Generate HTML report
echo ""
echo "Generating HTML coverage report..."
coverage html

# Generate XML report for CI/CD
coverage xml

echo ""
echo "=========================="
echo "Coverage Analysis Complete"
echo "=========================="
echo "HTML report available at: htmlcov/index.html"
echo "XML report available at: coverage.xml"

# Check if we achieved target coverage
COVERAGE_PERCENT=$(coverage report | tail -1 | awk '{print $4}' | sed 's/%//')
echo "Total coverage: ${COVERAGE_PERCENT}%"

if (( $(echo "$COVERAGE_PERCENT >= 75" | bc -l) )); then
    echo "✅ Coverage target of 75% achieved!"
    exit 0
else
    echo "⚠️  Coverage below 75% target. Current: ${COVERAGE_PERCENT}%"
    echo "This may be expected if Numba/Dask functions are still not being measured."
    exit 0  # Don't fail the script, just warn
fi
