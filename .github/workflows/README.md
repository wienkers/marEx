# GitHub Actions Workflows for marEx

This directory contains comprehensive GitHub Actions workflows for the marEx package that provide automated testing, code quality checks, performance monitoring, and release management.

## Workflows Overview

### 1. Continuous Integration (`ci.yml`)

**Triggers:**
- Push to `main` and `advanced_preproc` branches
- Pull requests to `main` branch

**Features:**
- **Multi-platform testing**: Ubuntu, Windows, and macOS
- **Multi-python testing**: Python 3.10, 3.11, and 3.12
- **Code quality checks**: Black, isort, flake8, bandit, mypy
- **Comprehensive test suite**: Unit tests, integration tests, error handling tests
- **Coverage reporting**: Codecov integration and HTML reports
- **Notebook validation**: Syntax checking and formatting for example notebooks
- **Build testing**: Package building and installation verification
- **Dependency caching**: Pip cache for faster builds

**Jobs:**
1. `code-quality`: Code formatting, linting, and security checks
2. `test`: Main test suite across multiple OS/Python combinations
3. `coverage`: Coverage report generation and upload
4. `test-notebooks`: Notebook syntax and formatting validation
5. `build-test`: Package building and installation testing
6. `all-checks-passed`: Final validation of all workflow jobs

### 2. Release Management (`release.yml`)

**Triggers:**
- Push to version tags (e.g., `v1.0.0`, `v2.1.3`)
- Manual workflow dispatch

**Features:**
- **Version validation**: Semantic version tag validation
- **Full test suite**: Comprehensive testing before release
- **Multi-platform build testing**: Verify installation across platforms
- **PyPI publishing**: Automated package publishing using trusted publishing
- **GitHub release creation**: Automatic release notes and changelog generation
- **Prerelease support**: Automatic detection of alpha/beta/rc versions

**Jobs:**
1. `validate-tag`: Version tag validation and prerelease detection
2. `run-tests`: Full test suite execution
3. `build`: Source and wheel distribution building
4. `test-install`: Installation testing across platforms
5. `publish-pypi`: PyPI package publishing
6. `create-github-release`: GitHub release creation with changelog
7. `announce-release`: Post-release notifications (for stable releases)

## Setup Instructions

### Required Secrets

#### 1. CodeCov Integration (Optional)
- `CODECOV_TOKEN`: Token for uploading coverage reports to Codecov
  - Go to https://codecov.io/ and connect your repository
  - Copy the repository token and add it as a secret

#### 2. PyPI Publishing (Required for releases)
The release workflow uses PyPI's trusted publishing feature, which is more secure than using API tokens.

**Setup Trusted Publishing:**
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher with these details:
   - **PyPI Project Name**: `marEx`
   - **Owner**: `wienkers` (or your GitHub username)
   - **Repository name**: `marEx`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`

**Alternative: Using API Token (less secure)**
If you prefer to use an API token instead:
1. Generate a PyPI API token at https://pypi.org/manage/account/token/
2. Add it as a repository secret named `PYPI_API_TOKEN`
3. Modify the release workflow to use the token instead of trusted publishing

### Environment Configuration

The release workflow uses a GitHub environment named `pypi` for additional security. To set this up:

1. Go to your repository settings
2. Navigate to "Environments"
3. Create a new environment named `pypi`
4. (Optional) Configure environment protection rules:
   - Require reviewers for production releases
   - Restrict to specific branches (e.g., `main`)

## Workflow Customization

### Modifying Test Coverage

To adjust which tests run in different scenarios, modify the test markers in `ci.yml`:

```yaml
# Skip slow tests in CI
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m "integration"

# Run specific test files
pytest tests/test_gridded_preprocessing.py -v
```

### Adding New Platforms

To test on additional platforms, add them to the matrix in `ci.yml`:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest, ubuntu-20.04]
    python-version: ['3.10', '3.11', '3.12']
```

## Workflow Dependencies

The workflows require these system dependencies:

**Ubuntu:**
```bash
libhdf5-dev libnetcdf-dev libproj-dev proj-data proj-bin libgeos-dev
```

**macOS:**
```bash
hdf5 netcdf proj geos
```

**Windows:**
Dependencies are handled through pip packages.

## Troubleshooting

### Common Issues

1. **Build failures on Windows**: Ensure all dependencies are properly specified in `pyproject.toml`
2. **Coverage upload failures**: Check that `CODECOV_TOKEN` is correctly configured
3. **Release workflow failures**: Verify that PyPI trusted publishing is properly configured
4. **Performance test timeouts**: Adjust timeout values in the workflow if needed

### Debugging Tips

1. **Check workflow logs**: Go to Actions tab in your repository to view detailed logs
2. **Local testing**: Run tests locally with the same commands used in CI
3. **Matrix debugging**: Use `strategy.fail-fast: false` to see all platform failures
4. **Artifact inspection**: Download artifacts from failed runs to analyze results

## Contributing

When contributing to marEx, please ensure that:

1. Your changes pass all CI checks
2. New features include appropriate tests
3. Documentation is updated if needed
4. Performance-critical changes include performance tests
