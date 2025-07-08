@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://sphinx-doc.org/
	exit /b 1
)

if "%1" == "help" (
	:help
	%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
	echo.
	echo.Additional marEx documentation targets:
	echo.  clean-all     Clean all build files including autosummary
	echo.  html-open     Build HTML and attempt to open in browser
	echo.  strict        Build with warnings as errors
	echo.  coverage      Generate docstring coverage report
	echo.  check-setup   Check documentation setup and dependencies
	echo.  install-deps  Install documentation dependencies
	echo.  dev           Fast development build
	echo.  production    Comprehensive production build
	echo.  info          Show build information
	goto end
)

if "%1" == "clean-all" (
	echo.Removing everything under '_build' and generated autosummary files...
	if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
	if exist "_autosummary" rmdir /s /q "_autosummary"
	if exist "generated" rmdir /s /q "generated"
	echo.Clean complete.
	goto end
)

if "%1" == "html-open" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
	if not errorlevel 1 (
		echo.Opening documentation in browser...
		start "" "%BUILDDIR%\html\index.html"
	)
	goto end
)

if "%1" == "strict" (
	echo.Building documentation with warnings as errors...
	%SPHINXBUILD% -W -b html %SOURCEDIR% %BUILDDIR%\html-strict %SPHINXOPTS% %O%
	goto end
)

if "%1" == "coverage" (
	echo.Generating docstring coverage report...
	%SPHINXBUILD% -b coverage %SOURCEDIR% %BUILDDIR%\coverage %SPHINXOPTS% %O%
	if exist "%BUILDDIR%\coverage\python.txt" (
		echo.
		echo.Coverage report:
		type "%BUILDDIR%\coverage\python.txt"
	)
	goto end
)

if "%1" == "check-setup" (
	echo.Checking documentation setup...
	python -c "import sphinx; print(f'Sphinx version: {sphinx.__version__}')" 2>NUL || echo.Sphinx not installed
	python -c "import sphinx_rtd_theme; print('sphinx_rtd_theme: OK')" 2>NUL || echo.sphinx_rtd_theme not installed
	python -c "import myst_parser; print('myst_parser: OK')" 2>NUL || echo.myst_parser not installed (optional)
	python -c "import nbsphinx; print('nbsphinx: OK')" 2>NUL || echo.nbsphinx not installed (optional)
	python -c "import sphinx_copybutton; print('sphinx_copybutton: OK')" 2>NUL || echo.sphinx_copybutton not installed (optional)
	echo.Setup check complete.
	goto end
)

if "%1" == "install-deps" (
	echo.Installing documentation dependencies...
	if exist "requirements-docs.txt" (
		pip install -r requirements-docs.txt
		echo.Dependencies installed.
	) else (
		echo.requirements-docs.txt not found. Please create it with documentation dependencies.
	)
	goto end
)

if "%1" == "dev" (
	echo.Development build (minimal checks)...
	%SPHINXBUILD% -E -b html %SOURCEDIR% %BUILDDIR%\html-dev %SPHINXOPTS% %O%
	goto end
)

if "%1" == "production" (
	echo.Production build (comprehensive checks)...
	call :clean-all
	%SPHINXBUILD% -W -b html -d %BUILDDIR%\doctrees %SOURCEDIR% %BUILDDIR%\html %SPHINXOPTS% %O%
	if not errorlevel 1 (
		echo.Running link check...
		%SPHINXBUILD% -b linkcheck -D linkcheck_ignore="^https?://" %SOURCEDIR% %BUILDDIR%\linkcheck-internal %SPHINXOPTS% %O%
	)
	echo.Production build complete. Files are in %BUILDDIR%\html.
	goto end
)

if "%1" == "info" (
	echo.marEx Documentation Build Information
	echo.=====================================
	echo.Source directory: %SOURCEDIR%
	echo.Build directory:  %BUILDDIR%
	echo.Sphinx command:   %SPHINXBUILD%
	python -c "import sys; print(f'Python path: {sys.executable}')" 2>NUL || echo.Python path: Not found
	echo.
	echo.Available targets:
	echo.  html          - Build HTML documentation
	echo.  html-open     - Build and open in browser
	echo.  pdf           - Build PDF documentation (requires LaTeX)
	echo.  epub          - Build EPUB documentation
	echo.  linkcheck     - Check for broken links
	echo.  coverage      - Generate docstring coverage
	echo.  clean-all     - Clean all build files
	echo.  dev           - Fast development build
	echo.  production    - Comprehensive production build
	echo.  strict        - Build with warnings as errors
	echo.  check-setup   - Check documentation setup
	echo.  install-deps  - Install documentation dependencies
	goto end
)

REM Default: pass through to Sphinx
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd
