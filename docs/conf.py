"""Configuration file for the Sphinx documentation builder.

See https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent

# -- Inline example notebooks ------------------------------------------------
# The canonical Jupyter notebooks live in ``examples/<grid> data/`` (referenced
# by the README and batch-job scripts). nbsphinx can only render notebooks that
# live inside the Sphinx source tree, so we copy them into ``docs/tutorials/``
# with space-free paths at build time. The copied directories are git-ignored
# and regenerated on every build (including on Read the Docs).

# Maps canonical example folder -> clean tutorials sub-directory / URL segment.
TUTORIAL_GRID_DIRS = {
    "gridded data": "gridded",
    "regional data": "regional",
    "unstructured data": "unstructured",
}


def _sync_tutorial_notebooks() -> None:
    """Copy example notebooks into ``docs/tutorials/<grid>/`` before the build."""
    examples_dir = REPO_ROOT / "examples"
    tutorials_dir = DOCS_DIR / "tutorials"
    for src_name, dest_name in TUTORIAL_GRID_DIRS.items():
        src = examples_dir / src_name
        dest = tutorials_dir / dest_name
        if dest.exists():
            shutil.rmtree(dest)
        if not src.is_dir():
            print(f"[conf.py] WARNING: example folder not found: {src}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        notebooks = sorted(src.glob("*.ipynb"))
        if not notebooks:
            print(f"[conf.py] WARNING: no notebooks found in {src}")
        for nb in notebooks:
            shutil.copy2(nb, dest / nb.name)
        print(f"[conf.py] synced {len(notebooks)} notebook(s): {src} -> {dest}")


_sync_tutorial_notebooks()

# -- Project information -----------------------------------------------------

project = "marEx"
copyright = "2024, Aaron Wienkers"
author = "Aaron Wienkers"

try:
    release = version("marEx")
except Exception:
    release = "unknown"

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
    "nbsphinx",
    "sphinxcontrib.video",
    "sphinx_reredirects",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "superpowers",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
pygments_style = "sphinx"

# -- Redirects (old URLs -> new locations) -----------------------------------
# Preserve inbound links after the reorganisation. Anchors cannot be carried by
# a redirect; pages land at the top of their new home.
redirects = {
    "quickstart": "getting_started/quickstart.html",
    "concepts": "guide/concepts.html",
    "user_guide": "guide/index.html",
    "examples": "tutorials/index.html",
    "api": "api/index.html",
    "modules/detect": "../api/detect.html",
    "modules/track": "../api/track.html",
    "modules/plotx": "../api/plotx.html",
    "modules/helper": "../api/helper.html",
}

# -- HTML output (pydata-sphinx-theme) ---------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = project

html_theme_options = {
    "github_url": "https://github.com/wienkers/marEx",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/marEx/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "show_nav_level": 1,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "footer_start": ["copyright", "last-updated"],
    "footer_end": ["sphinx-version"],
}

html_context = {
    "github_user": "wienkers",
    "github_repo": "marEx",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_short_title = project
html_last_updated_fmt = "%b %d, %Y"
html_show_sourcelink = False
htmlhelp_basename = "marExdoc"

# -- autodoc / autosummary ---------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autoclass_content = "both"

autosummary_generate = True
autosummary_generate_overwrite = True

# -- napoleon ----------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True
# Render docstring "Attributes:" sections as :ivar: fields instead of separate
# object descriptions, so dataclass fields (also picked up by autodoc :members:)
# are not documented twice.
napoleon_use_ivar = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
}

# -- nbsphinx ----------------------------------------------------------------
# Notebooks are NOT executed during the build: their datasets live on HPC
# scratch storage and are unavailable on Read the Docs. Committed cell outputs
# are rendered as-is. The banner links back to the canonical notebook in the
# ``examples/`` tree on GitHub (where the notebook can be downloaded/run).

nbsphinx_execute = "never"
nbsphinx_kernel_name = "python3"
nbsphinx_allow_errors = True

nbsphinx_prolog = r"""
{% set parts = env.docname.split('/') %}
{% if parts[0] == 'tutorials' and parts|length == 3 %}
{% set foldermap = {'gridded': 'gridded data', 'regional': 'regional data', 'unstructured': 'unstructured data'} %}
{% set gh_path = 'examples/' + foldermap[parts[1]] + '/' + parts[2] + '.ipynb' %}
{% set gh_url = ('https://github.com/wienkers/marEx/blob/main/' + gh_path)|replace(' ', '%20') %}
{% set raw_url = ('https://github.com/wienkers/marEx/raw/main/' + gh_path)|replace(' ', '%20') %}

.. note::

    This page is generated from the Jupyter notebook
    `{{ gh_path }} <{{ gh_url }}>`__.
    `View on GitHub <{{ gh_url }}>`__ · `Download notebook <{{ raw_url }}>`__
    (the notebook reads datasets that are not bundled with the docs, so it is
    rendered here from its committed outputs rather than re-executed).
{% endif %}
"""

# -- LaTeX / others (HTML is the only built format) --------------------------

latex_elements = {"papersize": "a4paper", "pointsize": "10pt"}
latex_documents = [
    (master_doc, "marEx.tex", "marEx Documentation", "Aaron Wienkers", "manual"),
]
