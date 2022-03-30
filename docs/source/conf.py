# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import datetime

import shifthappens


autodoc_mock_imports = ["numpy", "torch", "torchvision"]



def get_years(start_year=2021):
    year = datetime.datetime.now().year
    if year > start_year:
        return f"{start_year} - {year}"
    else:
        return f"{year}"


# -- Project information -----------------------------------------------------

project = "Shift Happens (ICML 2022)"
author = "Julian Bitterwolf, Evgenia Rusak, Steffen Schneider, Roland S. Zimmermann"

copyright = f"{get_years(2021)}, {author} and contributors. Released under an Apache 2.0 License"
release = shifthappens.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

coverage_show_missing_items = True

# Config is documented here: https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
autodoc_member_order = "bysource"

templates_path = ["_templates"]

exclude_patterns = []

html_theme = "pydata_sphinx_theme"

# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html
html_theme_options = {
    "nosidebar": True,
    "icon_links": [

        {
            "name": "Github",
            "url": "https://github.com/shift-happens-benchmark/iclr-2022",
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/shifthappens/",
            "icon": "fab fa-python",
        },
        {
            "name": "Contact us!",
            "url": "mailto:shifthappens@bethgelab.org",
            "icon": "fas fa-envelope",
        },

    ],
    "external_links": [
        {"name": "Home", "url": "https://shift-happens-benchmark.github.io/"}

    ],
    # "external_links": [{"name": "ICML 2022", "url": "https://icml.cc/"}],
    "collapse_navigation": False,
    "navigation_depth": 4,
    "navbar_align": "content",
    "show_prev_next": False,
}

html_logo = None

# Remove the search field for now
html_sidebars = {"**": ["sidebar-nav-bs.html"]}

# Disable links for embedded images
html_scaled_image_link = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
