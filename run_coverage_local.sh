#!/bin/bash
#
# Local Coverage Script for MarEx
# Replicates the GitHub Actions coverage job configuration for cluster execution
#

set -e  # Exit on any error

echo "=== MarEx Local Coverage Report ==="
echo "Starting coverage analysis on $(hostname) at $(date)"
echo

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "marEx" ]]; then
    echo "Error: Must be run from the marEx project root directory"
    exit 1
fi

# Load environment variables from .env file if it exists
if [[ -f ".env" ]]; then
    source .env
    echo "Loaded environment variables from .env"
fi

# Create Dask configuration directory
echo "Configuring Dask for conservative execution..."
mkdir -p ~/.dask

# Create Dask configuration (matches GitHub Actions conservative settings)
cat > ~/.dask/config.yaml << 'EOF'
distributed:
  worker:
    memory:
      target: 0.4
      spill: 0.5
      pause: 0.6
      terminate: 0.8
      recent-to-old-time: 10s
    daemon: false
  scheduler:
    allowed-failures: 50
    work-stealing: false
    worker-ttl: 600s
  comm:
    timeouts:
      connect: 300s
      tcp: 300s
    retry:
      count: 15
      delay:
        min: 3s
        max: 30s
  admin:
    log-format: '%(name)s - %(levelname)s - %(message)s'
array:
  chunk-size: 24MiB
  slicing:
    split-large-chunks: false
EOF

echo "Dask configuration created"

# Set environment variables for conservative execution
export DASK_NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONHASHSEED=42
export PYTEST_COVERAGE=true

echo "Environment variables set for single-threaded execution"

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Warning: No virtual environment detected. Consider activating one first."
    echo "Example: conda activate marex-env"
    echo
fi

# Install/upgrade coverage if needed
echo "Checking coverage installation..."
python -c "import coverage; print(f'Coverage version: {coverage.__version__}')" 2>/dev/null || {
    echo "Installing coverage..."
    pip install coverage[toml]
}

# Clean up any previous coverage data
echo "Cleaning up previous coverage data..."
rm -f .coverage .coverage.*
rm -rf htmlcov/

# Generate coverage report
echo "Starting coverage analysis..."
echo "This may take 30-45 minutes depending on your cluster performance..."
echo

# Run coverage with the same settings as GitHub Actions
timeout 2700 coverage run -m pytest tests/ --tb=short -m "not nocov" -x --maxfail=3 || {
    exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
        echo "Error: Coverage run timed out after 45 minutes"
        echo "Consider running with fewer tests or increasing timeout"
    else
        echo "Error: Coverage run failed with exit code $exit_code"
    fi
    exit $exit_code
}

echo
echo "Generating coverage reports..."

# Generate text report
coverage report -m

# Generate HTML report
coverage html
echo "HTML report generated in htmlcov/"

# Generate XML report (useful for tools like VS Code)
coverage xml
echo "XML report generated as coverage.xml"

# Upload to Codecov if token is available
if [[ -n "$CODECOV_TOKEN" ]]; then
    echo
    echo "Uploading coverage to Codecov..."

    # Install codecov if not present
    python -c "import codecov" 2>/dev/null || {
        echo "Installing codecov..."
        pip install codecov
    }

    # Upload coverage
    codecov -t "$CODECOV_TOKEN" -f coverage.xml || {
        echo "Warning: Codecov upload failed, but local reports are still available"
    }
else
    echo
    echo "Note: Set CODECOV_TOKEN environment variable to upload to Codecov"
    echo "export CODECOV_TOKEN=your_token_here"
fi

echo
echo "=== Coverage Analysis Complete ==="
echo "View the HTML report by opening: htmlcov/index.html"
echo "Or run: python -m http.server 8000 -d htmlcov"
echo "Then visit: http://localhost:8000"
echo
