# Contributing to marEx

Thank you for your interest in contributing to marEx! This document provides guidelines for contributing to the Marine Extremes Detection and Tracking package.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Guidelines](#documentation-guidelines)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Familiarity with oceanographic data analysis (helpful but not required)
- Basic understanding of xarray, Dask, and scientific Python ecosystem

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **Feature enhancements**: Improve existing functionality
- **New features**: Add new capabilities for marine extreme detection/tracking
- **Documentation**: Improve documentation, tutorials, or examples
- **Performance improvements**: Optimise algorithms or memory usage
- **Testing**: Add or improve test coverage
- **Examples**: Create/Share new example notebooks or workflows

## Development Environment Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/marEx.git
cd marEx

# Add the upstream repository as a remote
git remote add upstream https://github.com/wienkers/marEx.git
```

### 2. Create a Development Environment

We recommend using conda or mamba for managing dependencies:

```bash
# Create a new environment
conda create -n marex-dev python=3.10
conda activate marex-dev

# Install the package in development mode with all dependencies
pip install -e ".[dev,full]"
```

Alternative with pip and virtual environment:

```bash
# Create and activate virtual environment
python -m venv marex-dev
source marex-dev/bin/activate  # On Windows: marex-dev\Scripts\activate

# Install development dependencies
pip install -e ".[dev,full]"
```

### 3. Install Pre-commit Hooks

Pre-commit hooks ensure code quality and consistency:

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files to check setup
pre-commit run --all-files
```

### 4. Verify Installation

```bash
# Run tests to ensure everything works
pytest

# Check code formatting
black --check marEx/
flake8 marEx/

# Verify imports work
python -c "import marEx; print(marEx.__version__)"
```

## Contribution Workflow

### 1. Create a Feature Branch

```bash
# Ensure your main branch is up to date
git checkout main
git pull upstream main

# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

- Follow the [code style guidelines](#code-style-guidelines)
- Write tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

```bash
# Run the full test suite
pytest

# Run specific test categories
pytest -m "not slow"  # Skip slow tests during development
pytest tests/test_gridded_preprocessing.py -v  # Run specific test file

# Check code coverage
coverage run -m pytest
coverage report -m
```

### 4. Update Documentation

- Add docstrings to new functions/classes
- Update relevant documentation files
- Add examples if introducing new features

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Pre-commit hooks will run automatically
git commit -m "Add feature: brief description of changes"

# If pre-commit hooks modify files, stage and commit again
git add .
git commit -m "Apply pre-commit hook fixes"
```

### 6. Push and Create Pull Request

```bash
# Push your branch to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
# Fill out the pull request template completely
```

## Code Style Guidelines

### Code Quality Standards

- **Functions must accept Dask arrays**: All processing functions should validate `is_dask_collection(da.data)` and raise informative errors for non-Dask arrays
- **Memory efficiency**: Strategically use `.persist()` and `wait()` to manage the dask task graph and memory
- **Grid type support**: Ideally, support both structured (3D: time, lat, lon) and unstructured (2D: time, cell) grids; however, focus on structured grids for initial development

### Documentation Standards

- **Docstrings**: All public functions/classes should have comprehensive docstrings
- **Type hints**: Add type hints where practical
- **Examples**: Include usage examples in docstrings for complex functions
- **Parameter documentation**: Document all parameters, their types, and expected values


## Testing Requirements

### Test Structure

marEx uses pytest with the following test organisation:

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── test_gridded_preprocessing.py  # Gridded data preprocessing tests
├── test_gridded_tracking.py       # Gridded data tracking tests
├── test_unstructured_preprocessing.py  # Unstructured data tests
├── test_plotx.py                  # Plotting functionality tests
├── test_integration.py            # Integration tests
└── data/                          # Test data files
```

### Test Categories

Tests are organised using pytest markers:

- `@pytest.mark.slow`: Computationally expensive tests (skip with `-m "not slow"`)
- `@pytest.mark.integration`: End-to-end workflow tests

### Writing Tests

#### Test Requirements

1. **Test new functionality**: All new features must have corresponding tests
2. **Test edge cases**: Include tests for boundary conditions and error cases
3. **Test with both grid types**: Test structured and unstructured data where applicable
4. **Use fixtures**: Leverage existing fixtures for common test data
5. **Mock external dependencies**: Use mocking for expensive operations or external services


### Running Tests

```bash
# Run all tests
pytest

# Run specific test suites (as in CI)
pytest tests/test_gridded_preprocessing.py -v --tb=short
pytest tests/test_unstructured_preprocessing.py -v --tb=short
pytest tests/test_gridded_tracking.py -v --tb=short

# Skip slow tests during development
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run with coverage
coverage run -m pytest
coverage report -m
coverage html  # Generate HTML coverage report
```

### Test Data

Use the existing test fixtures in `conftest.py`:

```python
def test_my_function(dask_client, sample_sst_data):
    """Test using shared fixtures."""
    with dask_client:
        result = my_function(sample_sst_data)
        assert result is not None
```

## Documentation Guidelines

### Documentation Structure

marEx documentation is built with Sphinx and includes:

- **API documentation**: Auto-generated from docstrings
- **User guides**: Step-by-step tutorials
- **Examples**: Jupyter notebooks demonstrating workflows
- **Development documentation**: This file and related developer resources

### Building Documentation

```bash
cd docs/
make html  # Build HTML documentation
make clean  # Clean build artifacts

# View documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

### Documentation Standards

1. **Docstring format**: Use NumPy-style docstrings
2. **Examples in docstrings**: Include working code examples
3. **Cross-references**: Use Sphinx cross-references for linking
4. **Jupyter notebooks**: Keep example notebooks clean and well-documented
5. **User-focused**: Write documentation from the user's perspective

### Adding New Documentation

1. **API changes**: Docstrings are automatically included
2. **New features**: Add examples to relevant user guide sections
3. **Tutorials**: Create new Jupyter notebooks in `docs/tutorials/`

## Release Process

### Version Management

marEx uses `setuptools_scm` for automatic versioning based on git tags:

- Development versions: `0.2.0.dev10+g1234567` (based on commits since last tag)
- Release versions: `0.2.0` (based on git tags)

### Release Workflow

1. **Prepare release**:
   ```bash
   # Update CHANGELOG.md with new version
   # Ensure all tests pass
   pytest

   # Check documentation builds
   cd docs/ && make html
   ```

2. **Create release**:
   ```bash
   # Tag the release
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push upstream v0.2.0
   ```

3. **Build and distribute**:
   ```bash
   # Build package
   python -m build

   # Upload to PyPI (maintainers only)
   twine upload dist/*
   ```

### Release Checklist

- [ ] All tests pass on all supported Python versions
- [ ] Documentation builds without warnings
- [ ] Version tag created and pushed
- [ ] GitHub release created with release notes
- [ ] Package uploaded to PyPI

## Getting Help

### Resources

- **Documentation**: https://marex.readthedocs.io/
- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions or discuss ideas
- **Email**: Contact maintainers for sensitive issues

### Issue Reporting

When reporting issues:

1. Use the provided issue templates
2. Include minimal reproducible examples
3. Provide environment information (Python version, OS, package versions)
4. Include full error messages and tracebacks

### Feature Requests

When requesting features:

1. Describe the scientific use case
2. Explain why existing functionality doesn't meet your needs
3. Provide examples of the desired API
4. Consider contributing the implementation

## Recognition

Contributors are recognised in:

- GitHub contributors list
- Release notes
- Documentation acknowledgments
- Academic publications (for significant contributions)

Thank you for contributing to marEx!
