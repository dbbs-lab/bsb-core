import os
import sys

# Fetch the `__version__`
bsb_folder = os.path.join(os.path.dirname(__file__), "..", "bsb")
bsb_init_file = os.path.join(bsb_folder, "__init__.py")
_findver = "__version__ = "
with open(bsb_init_file, "r") as f:
    for line in f:
        if "__version__ = " in line:
            f = line.find(_findver)
            __version__ = eval(line[line.find(_findver) + len(_findver) :])
            break
    else:
        raise Exception(f"No `__version__` found in '{bsb_init_file}'.")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -- Project information -----------------------------------------------------

project = "Brain Scaffold Builder"
copyright = "2022, DBBS University of Pavia"
author = "Robin De Schepper"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxemoji.sphinxemoji",
    "sphinx_design",
    "sphinx_copybutton",
    "bsbdocs",
]

autodoc_mock_imports = [
    "glia",
    "patch",
    "mpi4py",
    "rtree",
    "rtree.index",
    "mpi4py.MPI",
    "dbbs_models",
    "arborize",
    "h5py",
    "joblib",
    "sklearn",
    "scipy",
    "six",
    "plotly",
    "psutil",
    "mpilock",
    "zwembad",
    "arbor",
    "morphio",
    "nrrd",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs/", None),
    "errr": ("https://errr.readthedocs.io/en/latest/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
    "arbor": ("https://docs.arbor-sim.org/en/latest/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "bsb/bsb.simulators*",
    "guides/labels.rst",
    "guides/blender.rst",
    "guides/layer.rst",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_theme_options = {
    "light_logo": "bsb.svg",
    "dark_logo": "bsb_dark.svg",
    "sidebar_hide_name": True,
}

html_favicon = "_static/bsb_ico.svg"

html_context = {
    "maintainer": "Robin De Schepper",
    "project_pretty_name": "BSB",
    "projects": {"DBBS Scaffold": "https://github.com/dbbs/bsb"},
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

todo_include_todos = True
