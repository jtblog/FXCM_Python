# `fxcmpy` Documentation

This is the Git repository containing the **files of the documentation** for the `fxcmpy` Python wrapper package.

First, make sure to have the **required Python packages** installed as well as `Sphinx` with the Notebook extension `nbsphinx`:

    pip install numpy pandas matplotlib ipython jupyter
    pip install sphinx nbsphinx

To **build the documentation** locally, first clone the repository then navigate to the folder of the repository and execute the build file:

    git clone https://github.com/fxcm/fxcmpy-doc.git
    cd fxcmpy-doc
    make html

The build files are then found in `_build/html`.

The Python Quants GmbH \| <team@tpq.io>

