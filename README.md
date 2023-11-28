# Welcome to gsf

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/None/badge/)](https://None.readthedocs.io/)
[![codecov](https://codecov.io/none/None/None/branch/main/graph/badge.svg)](https://codecov.io/none/None/None)

## Installation

The Python package `gsf` can be installed from PyPI:

```
python -m pip install gsf
```

## Development installation

If you want to contribute to the development of `gsf`, we recommend
the following editable installation from this repository:

```
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
