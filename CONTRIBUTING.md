# Contributing to marEx

Thank you for your interest in contributing to marEx. This guide covers the development setup, the test commands, and the conventions the code base follows.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Contribution Workflow](#contribution-workflow)
  - [Contributor License Agreement](#contributor-license-agreement)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Continuous Integration](#continuous-integration)
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
pytest tests/test_api.py -q
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
5. Agree to the [Contributor License Agreement](CLA.md) by ticking its box in the pull request template. It is needed once, before your first contribution is merged.

### Contributor License Agreement

marEx is released under the GNU Affero General Public License v3.0 or later. You keep the copyright in your contributions. The [CLA](CLA.md) grants the maintainer a licence to distribute them under the AGPL and under other terms, which keeps the option of commercial licences open. If your employer has rights in your work, make sure you have its permission before contributing.

## Code Style Guidelines

- Formatting and linting: `ruff check` and `ruff format` (configured in `pyproject.toml`), run by pre-commit and CI.
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
| Anomaly stage | `tests/anomaly/` |
| Extremes stage | `tests/extremes/` |
| Full detect pipeline | `tests/detect/` (`test_pipeline_golden.py`, `test_method_matrix.py`, `test_extra_dim.py`) |
| Chunking and compute modes | `tests/detect/test_chunk_layout.py`, `tests/detect/test_compute_modes.py`, `tests/detect/test_compute_mode_equivalence.py`, `tests/track/test_track_compute_mode.py` |
| Tracking | `tests/track/` (`test_tracker_outputs.py` holds the golden run) |
| Plotting | `tests/plotting/` |
| Public API | `tests/test_api.py` |
| End to end (detect, write, track) | `tests/test_end_to_end.py` |
| Errors, logging, clusters | `tests/test_errors.py`, `tests/test_logging.py`, `tests/test_cluster.py` |
| Larger than memory | `tests/test_larger_than_memory.py` |

### Running Tests

```bash
# Quick check of one area
pytest tests/anomaly -q

# Skip slow tests during development
pytest -m "not slow"

# Parallel run on a workstation (small worker counts; each test may start a Dask cluster)
pytest -n 2

# The fast end-to-end subset that CI runs against the built wheel
pytest -m smoke

# Streaming detect and track on inputs larger than the test cluster's memory
pytest -m larger_than_memory

# Coverage
pytest -n 2 --cov=marEx --cov-report=term
```

A plain `pytest` deselects the `larger_than_memory` tests (`addopts` in `pyproject.toml`), and an explicit `-m` on the command line replaces that filter. Each of those tests starts its own one-worker cluster with a fixed memory limit, sized so the governed input exceeds it by at least 1.1x.

The other tests share a Dask cluster whose per-worker `memory_limit` is 60 % of the usable memory (the smallest of physical RAM, the cgroup limit and `SLURM_MEM_PER_NODE`), divided between every worker of every xdist process. Dask does not see a cgroup or SLURM cap by itself, so this keeps the limit honest on a batch node. Set `MAREX_TEST_WORKER_MEMORY` (bytes, or a Dask string such as `2.5GB`) to override it.

Avoid `pytest -n auto` on a shared or memory-limited machine: several tests start a Dask cluster, and one worker per core can exhaust memory.

### Equivalence Tests

Changes to numerics must keep the reference-output tests (`tests/detect/test_pipeline_golden.py`, `tests/track/test_tracker_outputs.py`) passing at zero tolerance for integer and label outputs. Changes that alter results on purpose need a changelog entry and an explicit update of the references, never a loosened tolerance. Performance work should also keep the chunking- and compute-mode-invariance tests green, and assert chunk structure as well as values, because identical numbers do not rule out an expensive task graph.

### Both Zarr Majors

marEx supports `zarr>=2.18`, which includes zarr-python 2 and 3. Tests that write stores should pass on both. To check the other major locally:

```bash
pip install "zarr<3"   # zarr 2
pytest tests/test_end_to_end.py -q
pip install "zarr>=3"  # zarr 3
pytest tests/test_end_to_end.py -q
```

### Writing Tests

1. Test new behaviour, boundary conditions and error cases.
2. Cover both grid types where the feature applies.
3. Use the shared fixtures in `tests/conftest.py`.
4. Mark expensive tests `@pytest.mark.slow`; end-to-end workflows use `@pytest.mark.integration`.
5. Add `@pytest.mark.smoke` only to a fast test that needs nothing beyond the installed wheel, since the smoke jobs run it on macOS and Windows from the wheel alone.

## Continuous Integration

GitHub Actions runs four workflows.

| Workflow | Runs on | What it does |
|---|---|---|
| `ci.yml` | pushes to `main` and the development branches (`v4D_beta`, `ci/**`), pull requests against `main`, manual dispatch | Lint (pre-commit and the release-check unit tests), the full suite at `-n 2` on Python 3.12, the larger-than-memory tests, a wheel and sdist build, smoke tests of the installed wheel (Ubuntu with Python 3.10 and 3.13, macOS and Windows with 3.12), and the Sphinx build with warnings as errors. The `CI OK` job passes only when all of them do. |
| `coverage.yml` | pushes to `main`, weekly, manual dispatch | The suite under `pytest-cov` with the numba JIT on (`@jit` functions are excluded from the report), uploaded to Codecov. Informational: it never blocks a merge. Takes about 40 min on a GitHub runner. |
| `compat.yml` | weekly, manual dispatch | The suite at the declared dependency floors (`ci/requirements-minimum.txt`, Python 3.10, zarr 2), and at the newest releases on Python 3.14 with deprecation warnings shown. |
| `release.yml` | a pushed `v*` tag, manual dispatch | The release pipeline below. |

Dependabot keeps the Actions versions current, once a month.

## Documentation Guidelines

The documentation is built with Sphinx and hosted on ReadTheDocs. CI builds it with `-W`, so a new Sphinx warning fails the pull request.

```bash
cd docs/
make html
xdg-open _build/html/index.html   # `open` on macOS
```

- API pages are generated from docstrings, so a new public function needs a docstring and an entry in the matching page under `docs/api/`.
- Guides explain choices and worked examples; keep numbers tied to a stated scale and hardware, and describe them as measured on that setup.
- Example notebooks live under `examples/` and are rendered into the tutorials.

## Release Process

marEx uses `setuptools_scm`, so versions come from git tags, and a pushed tag is the whole release.

1. Add a `## [X.Y.Z]` section to `CHANGELOG.md` and merge it to `main`. That section opens the GitHub release notes, above the list of changes GitHub generates. Without it the release still goes out, with a one-line placeholder.
2. Wait for `CI OK` to pass on that `main` commit.
3. Tag the commit and push the tag:
   ```bash
   git tag -a vX.Y.Z -m "Release X.Y.Z" <sha>
   git push origin vX.Y.Z
   ```

`release.yml` then checks that the tag is a valid release version, waits up to 90 min for a successful `ci.yml` push run on `main` for the tagged commit, builds the wheel and sdist and checks their version against the tag, installs the wheel on Linux, macOS and Windows, publishes to PyPI by trusted publishing (environment `pypi`), and creates the GitHub release. A pre-release tag such as `v5.1.0rc1` is marked as one on GitHub. A tag on a commit that never passed CI on `main` stops at the CI check, before anything is built.

To exercise the version check and the CI check without publishing, dispatch a dry run:

```bash
gh workflow run release.yml --ref <branch> -f sha=<40-character sha> -f version=X.Y.Z
```

## Getting Help

- Documentation: https://marex.readthedocs.io/
- GitHub Issues: bugs and feature requests, with a minimal reproducible example, the Python, xarray, dask and zarr versions, and the full traceback
- GitHub Discussions: questions and ideas
- Email the maintainer for anything sensitive

Contributors are recognised in the GitHub contributors list and the release notes.
