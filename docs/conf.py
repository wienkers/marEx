"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "marEx"
copyright = "2024, Aaron Wienkers"
author = "Aaron Wienkers"

# The full version, including alpha/beta/rc tags
try:
    release = version("marEx")
except Exception:
    release = "unknown"

# The short X.Y version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    # Third-party extensions for scientific documentation
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
    "sphinxcontrib.video",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": None,
    ".md": "myst-parser",
}

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980b9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS files
html_css_files = []

# The name for this set of Sphinx documents.
html_title = f"{project} v{release}"

# A shorter title for the navigation bar.
html_short_title = project

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "donate.html",
    ]
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {}

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.
html_use_opensearch = ""

# Output file base name for HTML help builder.
htmlhelp_basename = "marExdoc"

# -- Options for autodoc extension -------------------------------------------

# This value selects if automatically documented members are sorted
# alphabetically or by member type.
autodoc_member_order = "bysource"

# This value is a list of autodoc directive flags that should be automatically
# applied to all autodoc directives.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# This value controls how to represent typehints.
autodoc_typehints = "description"

# Include __init__ docstrings in class documentation
autoclass_content = "both"

# -- Options for autosummary extension ---------------------------------------

# Boolean indicating whether to scan all found documents for autosummary
# directives, and to generate stub pages for each.
autosummary_generate = True

# If true, autosummary overwrites existing files by generated stub pages.
autosummary_generate_overwrite = False

# -- Options for napoleon extension ------------------------------------------

# True to parse NumPy style docstrings.
napoleon_numpy_docstring = True

# True to parse Google style docstrings.
napoleon_google_docstring = True

# True to include private members (like _membername) with docstrings in the documentation.
napoleon_include_private_with_doc = False

# True to include special members (like __membername__) with docstrings in the documentation.
napoleon_include_special_with_doc = True

# True to use the .. admonition:: directive for the Example and Examples sections.
napoleon_use_admonition_for_examples = False

# True to use the .. admonition:: directive for the Note and Notes sections.
napoleon_use_admonition_for_notes = False

# True to use the .. admonition:: directive for the References section.
napoleon_use_admonition_for_references = False

# True to use the :ivar: role for instance variables.
napoleon_use_ivar = False

# True to use a :param: role for each function parameter.
napoleon_use_param = True

# True to use a :keyword: role for each function keyword argument.
napoleon_use_keyword = True

# True to use the :rtype: role for the return type.
napoleon_use_rtype = True

# -- Options for intersphinx extension ---------------------------------------

# Links to other project's documentation
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

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for coverage extension ------------------------------------------

# List of modules to be skipped when collecting coverage information.
coverage_skip_undoc_in_source = True

# -- Options for nbsphinx extension ------------------------------------------

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::
        This page was generated from `{{ docname }}`__.
        Interactive online version:
        :raw-html:`<a href="https://mybinder.org/v2/gh/wienkers/marEx/{{ env.config.release|e }}?urlpath=lab/tree/{{ docname }}">`
        :raw-html:`<img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

    __ https://github.com/wienkers/marEx/blob/{{ env.config.release|e }}/{{ docname }}
"""

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. raw:: html

        <div class="sphx-glr-download-link-note admonition note">
        <p class="admonition-title">Note</p>
        <p>This page was generated from
        <a class="reference external" href="https://github.com/wienkers/marEx/blob/{{ env.config.release|e }}/{{ docname }}">
        {{ docname }}</a>.
        Interactive online version:
        <a href="https://mybinder.org/v2/gh/wienkers/marEx/{{ env.config.release|e }}?urlpath=lab/tree/{{ docname }}">
        <img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>
        </p>
        </div>
"""

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = "never"

# Use this kernel instead of the one stored in the notebook metadata:
nbsphinx_kernel_name = "python3"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": "",
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "marEx.tex", "marEx Documentation", "Aaron Wienkers", "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = False

# If true, show page references after internal links.
latex_show_pagerefs = False

# If true, show URL addresses after external links.
latex_show_urls = False

# Documents to append as an appendix to all manuals.
latex_appendices = []

# It false, will not define \strong, \code, 	itleref, \crossref ... but only
# \sphinxstrong, ..., \sphinxtitleref, ... To help avoid clash with user added
# packages.
latex_keep_old_macro_names = True

# If false, no module index is generated.
latex_domain_indices = True

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "marex", "marEx Documentation", [author], 1)]

# If true, show URL addresses after external links.
man_show_urls = False

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "marEx",
        "marEx Documentation",
        author,
        "marEx",
        "Marine Extremes Detection and Tracking",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
texinfo_appendices = []

# If false, no module index is generated.
texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
texinfo_show_urls = "footnote"

# If true, do not generate a @detailmenu in the "Top" node's menu.
texinfo_no_detailmenu = False
