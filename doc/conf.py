# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Make the package importable for autodoc when building from a source checkout
# (on ReadTheDocs the package is pip-installed, so this is just a convenience).
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = 'gsf'
copyright = '2018–2026, Aura Obreja and the gsf contributors'
author = 'Aura Obreja and the gsf contributors'
release = '2.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_mdinclude",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

# Let autodoc import `gsf` without the compiled Fortran extension being built
# (e.g. on ReadTheDocs), so the API pages can be generated from the docstrings.
autodoc_mock_imports = ["_twobody"]

# The demo notebook runs the full pipeline (needs the compiled extension and a
# few minutes), so it is NOT executed at build time. Run it once locally and
# commit it with its outputs; nbsphinx then renders those committed outputs.
nbsphinx_execute = "never"

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
