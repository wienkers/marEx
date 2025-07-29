#!/bin/bash

# Script to run coverage tests locally with multiprocessing support
# This script sets up the environment for proper coverage tracking with Dask

set -e  # Exit on error

echo "Starting coverage run with multiprocessing support..."

# Set environment variables for coverage in subprocesses
export COVERAGE_PROCESS_START=.coveragerc
export PYTEST_COVERAGE=1

# Clean up any previous coverage data
echo "Cleaning up previous coverage data..."
coverage erase
rm -f .coverage.*

# Run tests with multiprocessing support
echo "Running tests with coverage..."
coverage run --parallel-mode -m pytest tests/ --tb=short -m "not nocov" -x --maxfail=3

# Combine parallel coverage data
echo "Combining coverage data from parallel processes..."
coverage combine

# Generate reports
echo "Generating coverage reports..."
coverage report -m
coverage html
coverage xml

echo "Coverage analysis complete!"
echo "HTML report available at: htmlcov/index.html"
echo "XML report available at: coverage.xml"
