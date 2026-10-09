# Contributing to marEx

Thank you for your interest in contributing to marEx. This guide covers the development setup, the test commands, and the conventions the code base follows.

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

- Python 3.10 to 3.13
- Git
- Familiarity with xarray, Dask, and the scientific Python ecosystem

### Types of Contributions

- **Bug fixes**: reports with a minimal reproducible example are the most useful starting point
- **Features**: new anomaly or threshold methods, grid support, tracker options
- **Documentation**: guides, application pages, tutorials, docstrings
- **Performance**: memory or wall-time improvements, ideally with a measurement
- **Tests and examples**: extra coverage, notebooks, new datasets

## Development Environment Setup

### 1. Fork and Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/marEx.git
cd marEx
git remote add upstream https://github.com/wienkers/marEx.git
```

### 2. Create a Development Environment

```bash
conda create -n marex-dev python=3.11
conda activate marex-dev
pip install -e ".[dev,full]"
```

Or with a virtual environment:

```bash
python -m venv marex-dev
source marex-dev/bin/activate
pip install -e ".[dev,full]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### 4. Verify the Installation

```bash
python -c "import marEx; print(marEx.__version__)"
pytest tests/test_api_surface.py -q
```

## Contribution Workflow

1. Update your fork and branch from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```
2. Make focused changes, with tests for new behaviour and docstrings for new public functions.
3. Run the relevant tests (see [Testing Requirements](#testing-requirements)) and the pre-commit hooks.
4. Commit, push to your fork, and open a pull request against `main`. Describe the motivation and what you tested.

## Code Style Guidelines

- Formatting and linting: `black`, `isort` and `flake8` (configured in `pyproject.toml`), run by pre-commit and CI.
- Public functions take Dask-backed arrays and raise an informative `DataValidationError` otherwise.
- Support both gridded `(time, lat, lon)` and unstructured `(time, cell)` data wherever the algorithm allows. Fields with an extra dimension such as depth are supported by the anomaly and extremes stages and rejected by the tracker.
- Keep memory bounded. Intermediate results go through the `compute_mode` machinery (`persist`, `lazy`, `streaming`) rather than ad hoc `.persist()` calls.
- Chunking differs by stage: the detect stages need no particular spatial chunking on a lat/lon grid (chunk space on a large unstructured mesh), whereas the tracker needs the spatial dimension whole and is chunked in time.
- Docstrings use NumPy style and name the public import path (`marEx.anomaly.compute`, `marEx.extremes.identify`, `marEx.helper.start_local_cluster`).
- British spelling throughout (visualise, standardise, centred).

## Testing Requirements

Tests use pytest. The files group by area:

| Area | Files |
|---|---|
| Anomaly stage | `tests/test_anomaly_standalone.py`, `tests/test_3d_preprocessing.py`, `tests/test_fixed_baseline_smoothing.py` |
| Extremes stage | `tests/test_low_tail.py`, `tests/test_histogram_quantile.py`, `tests/test_percentile_agreement.py` |
| Full detect pipeline | `tests/test_gridded_preprocessing.py`, `tests/test_unstructured_preprocessing.py`, `tests/test_pipeline_golden.py` |
| Chunking and compute modes | `tests/test_detect_chunk_invariance.py`, `tests/test_compute_mode.py`, `tests/test_compute_mode_equivalence.py`, `tests/test_track_compute_mode.py` |
| Tracking | `tests/test_gridded_tracking.py`, `tests/test_unstructured_tracking.py`, `tests/test_track_golden.py` |
| Public API | `tests/test_api_surface.py` |

### Running Tests

```bash
# Quick check of one area
pytest tests/test_anomaly_standalone.py -q

# Skip slow tests during development
pytest -m "not slow"

# Parallel run on a workstation (small worker counts; each test may start a Dask cluster)
pytest -n 2

# Coverage
coverage run -m pytest
coverage report -m
```

Avoid `pytest -n auto` on a shared or memory-limited machine: several tests start a Dask cluster, and one worker per core can exhaust memory. `tests/test_error_handling.py` is slow, so run the class you need (`pytest tests/test_error_handling.py::TestName`) rather than the whole file.

### Equivalence Tests

Changes to numerics must keep the reference-output tests (`tests/test_pipeline_golden.py`, `tests/test_track_golden.py`) passing at zero tolerance for integer and label outputs. Changes that alter results on purpose need a changelog entry and an explicit update of the references, never a loosened tolerance. Performance work should also keep the chunking- and compute-mode-invariance tests green, and assert chunk structure as well as values, because identical numbers do not rule out an expensive task graph.

### Both Zarr Majors

marEx supports `zarr>=2.18`, which includes zarr-python 2 and 3. Tests that write stores should pass on both. To check the other major locally:

```bash
pip install "zarr<3"   # zarr 2
pytest tests/test_save_roundtrip.py tests/test_store_encoding_writes.py -q
pip install "zarr>=3"  # zarr 3
pytest tests/test_save_roundtrip.py tests/test_store_encoding_writes.py -q
```

### Writing Tests

1. Test new behaviour, boundary conditions and error cases.
2. Cover both grid types where the feature applies.
3. Use the shared fixtures in `tests/conftest.py`.
4. Mark expensive tests `@pytest.mark.slow`; end-to-end workflows use `@pytest.mark.integration`.

## Documentation Guidelines

The documentation is built with Sphinx and hosted on ReadTheDocs.

```bash
cd docs/
make html
xdg-open _build/html/index.html   # open on macOS
```

- API pages are generated from docstrings, so a new public function needs a docstring and an entry in the matching page under `docs/api/`.
- Guides explain choices and worked examples; keep numbers tied to a stated scale and hardware, and describe them as measured on that setup.
- Example notebooks live under `examples/` and are rendered into the tutorials.

## Release Process

marEx uses `setuptools_scm`, so versions come from git tags.

1. Update `CHANGELOG.md` and confirm the tests and the docs build pass.
2. Tag the release (`git tag -a v5.0.0 -m "Release 5.0.0"`) and push the tag.
3. Maintainers build and upload the package (`python -m build`, `twine upload dist/*`).

## Getting Help

- Documentation: https://marex.readthedocs.io/
- GitHub Issues: bugs and feature requests, with a minimal reproducible example, the Python, xarray, dask and zarr versions, and the full traceback
- GitHub Discussions: questions and ideas
- Email the maintainer for anything sensitive

Contributors are recognised in the GitHub contributors list and the release notes.
