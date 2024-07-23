# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme
import os
import sys

# Temporary fix to ensure that the stgraph codebase can be accessed
sys.path.insert(0, os.path.abspath("../python"))

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/STGraph_docs_logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

html_css_files = ["custom.css"]

project = "STGraph"
copyright = "2024, Joel Mathew Cherian, Nithin Puthalath Manoj"
author = "Joel Mathew Cherian, Nithin Puthalath Manoj"
release = "1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
