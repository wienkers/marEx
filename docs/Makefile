# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for marEx documentation

# Clean everything including autosummary generated files
clean-all:
	@echo "Removing everything under '_build' and generated autosummary files..."
	rm -rf $(BUILDDIR)/*
	rm -rf _autosummary
	rm -rf generated

# Build HTML with autosummary generation
html-autosummary:
	@echo "Building HTML documentation with autosummary..."
	@$(SPHINXBUILD) -b html -d $(BUILDDIR)/doctrees $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build HTML and open in browser (Linux/macOS)
html-open:
	@make html
	@echo "Opening documentation in browser..."
	@which xdg-open > /dev/null && xdg-open $(BUILDDIR)/html/index.html || \
	 which open > /dev/null && open $(BUILDDIR)/html/index.html || \
	 echo "Could not detect browser. Please open $(BUILDDIR)/html/index.html manually."

# Live reload development server
livehtml:
	@echo "Starting live reload server..."
	@which sphinx-autobuild > /dev/null || (echo "Please install sphinx-autobuild: pip install sphinx-autobuild" && exit 1)
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html \
		--watch ../marEx \
		--ignore "*.pyc" \
		--ignore "*~" \
		--ignore "*.swp" \
		--ignore "*.swo" \
		--port 8000

# Check for broken links
linkcheck-full:
	@echo "Checking for broken links (including external)..."
	@$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete. Check $(BUILDDIR)/linkcheck/output.txt for results."

# Check only internal links
linkcheck-internal:
	@echo "Checking internal links only..."
	@$(SPHINXBUILD) -b linkcheck -D linkcheck_ignore='^https?://' $(SOURCEDIR) $(BUILDDIR)/linkcheck-internal
	@echo
	@echo "Internal link check complete."

# Build documentation with all warnings as errors
strict:
	@echo "Building documentation with warnings as errors..."
	@$(SPHINXBUILD) -W -b html $(SOURCEDIR) $(BUILDDIR)/html-strict

# Generate coverage report for docstrings
coverage:
	@echo "Generating docstring coverage report..."
	@$(SPHINXBUILD) -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage
	@echo
	@echo "Coverage report generated in $(BUILDDIR)/coverage/"
	@cat $(BUILDDIR)/coverage/python.txt

# Build LaTeX/PDF documentation
pdf:
	@echo "Building PDF documentation..."
	@$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo
	@echo "PDF build finished. The PDF files are in $(BUILDDIR)/latex."

# Build EPUB documentation
epub:
	@echo "Building EPUB documentation..."
	@$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub
	@echo
	@echo "EPUB build finished. The EPUB file is in $(BUILDDIR)/epub."

# Check documentation setup and dependencies
check-setup:
	@echo "Checking documentation setup..."
	@python -c "import sphinx; print(f'Sphinx version: {sphinx.__version__}')" || echo "Sphinx not installed"
	@python -c "import sphinx_rtd_theme; print('sphinx_rtd_theme: OK')" || echo "sphinx_rtd_theme not installed"
	@python -c "import myst_parser; print('myst_parser: OK')" || echo "myst_parser not installed (optional)"
	@python -c "import nbsphinx; print('nbsphinx: OK')" || echo "nbsphinx not installed (optional)"
	@python -c "import sphinx_copybutton; print('sphinx_copybutton: OK')" || echo "sphinx_copybutton not installed (optional)"
	@echo "Setup check complete."

# Install documentation dependencies
install-deps:
	@echo "Installing documentation dependencies..."
	pip install -r requirements-docs.txt
	@echo "Dependencies installed."

# Development build (fast, for testing changes)
dev:
	@echo "Development build (minimal checks)..."
	@$(SPHINXBUILD) -E -b html $(SOURCEDIR) $(BUILDDIR)/html-dev

# Production build (comprehensive, for deployment)
production:
	@echo "Production build (comprehensive checks)..."
	@make clean-all
	@$(SPHINXBUILD) -W -b html -d $(BUILDDIR)/doctrees $(SOURCEDIR) $(BUILDDIR)/html
	@make linkcheck-internal
	@echo
	@echo "Production build complete. Files are in $(BUILDDIR)/html."

# Show build information
info:
	@echo "marEx Documentation Build Information"
	@echo "====================================="
	@echo "Source directory: $(SOURCEDIR)"
	@echo "Build directory:  $(BUILDDIR)"
	@echo "Sphinx command:   $(SPHINXBUILD)"
	@echo "Python path:      $(shell which python)"
	@echo ""
	@echo "Available targets:"
	@echo "  html          - Build HTML documentation"
	@echo "  html-open     - Build and open in browser"
	@echo "  livehtml      - Start live reload server"
	@echo "  pdf           - Build PDF documentation"
	@echo "  epub          - Build EPUB documentation"
	@echo "  linkcheck     - Check for broken links"
	@echo "  coverage      - Generate docstring coverage"
	@echo "  clean-all     - Clean all build files"
	@echo "  dev           - Fast development build"
	@echo "  production    - Comprehensive production build"
	@echo "  strict        - Build with warnings as errors"
	@echo "  check-setup   - Check documentation setup"
	@echo "  install-deps  - Install documentation dependencies"
