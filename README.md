# gsf — GalacticStructureFinder

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- TODO (Day 2): after you connect the repo to Zenodo and tag a release,
     paste the DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
     Remove this comment once done. -->

**GalacticStructureFinder (gsf)** decomposes simulated galaxies into their
constituent stellar components based on stellar kinematics. Given the star, gas,
and dark matter particles of an isolated, centered galaxy, it separates the
stellar particles into a chosen number of components by fitting Gaussian
Mixture Models in a user-defined feature space (e.g. circularities and normalized
binding energy).  

The method is described in:

> Obreja, Macciò, Moster et al., *"Introducing galactic structure finder: the
> multiple stellar kinematic structures of a Milky Way mass galaxy"*,
> MNRAS **477**, 4915 (2018).
> [2018MNRAS.477.4915O](https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.4915O)

<!-- TODO: if the paper currently in proofs should also be cited as the
     reference for this release/version, add it here once it has a DOI. -->

## Requirements

`gsf` builds a small Fortran extension (`twobody.f95`) at install time, so in
addition to Python you need a working build toolchain **before** installing:

- Python ≥ 3.8
- A Fortran compiler (`gfortran`)
- `cmake`
- An OpenMP runtime (e.g. `libgomp`)

On Debian/Ubuntu these can be installed with:

```bash
sudo apt install gfortran cmake libgomp1
```

Python dependencies (`numpy`, `scipy`, `matplotlib`, `scikit-learn`, `click`,
`pillow`) are installed automatically by pip.

## Installation

<!-- TODO (name pending): confirm the exact registered PyPI distribution name
     with your colleague, then update the command below. If the published name
     is "gsf2", this line becomes: python -m pip install gsf2
     (The import name in your code stays `import gsf` regardless.) -->

Once released on PyPI:

```bash
python -m pip install PACKAGE_NAME_PENDING
```

Or install the latest version directly from source:

```bash
git clone https://github.com/aobr/gsf2.git
cd gsf2
python -m pip install .
```

## Quick start

### Python API

```python
from gsf import gsf

# Three input files: star, gas, and dark matter particle properties.
# Expected units: Msun (mass), kpc (positions), km/s (velocities).
gsf(
    "sim1.halo_1.star.dat",
    "sim1.halo_1.gas.dat",
    "sim1.halo_1.dark.dat",
    varlist="jzjc,jpjc,e",     # features used for the clustering
    number_of_clusters=3,      # number of galaxy components
    out_dir="output/",
    plot=True,
)
```

To search for the optimal number of components instead of fixing it, use
`gsf_loop`, which scans a range of component counts and produces the
log-likelihood diagnostics used for model selection:

```python
from gsf import gsf_loop

gsf_loop(
    "sim1.halo_1.star.dat",
    "sim1.halo_1.gas.dat",
    "sim1.halo_1.dark.dat",
    varlist="jzjc,jpjc,e",
    out_dir="output/",
)
```

### Command line

The same functionality is available as the `gsf` command:

```bash
gsf sim1.halo_1.star.dat sim1.halo_1.gas.dat sim1.halo_1.dark.dat \
    --number_of_clusters 3 --plot
```

See `gsf --help` for the full list of options.

## Documentation

<!-- TODO: enable the ReadTheDocs integration and link it here, e.g.
     https://gsf2.readthedocs.io  — then add its badge at the top. -->
API documentation and an example notebook are available in the `doc/` and
`notebooks/` directories.

## Development

For a development (editable) install with the test dependencies:

```bash
python -m pip install --editable ".[tests]"
python -m pytest
```

## Citation

If you use `gsf` in your work, please cite the MNRAS paper above
([2018MNRAS.477.4915O](https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.4915O)).
<!-- TODO (Day 2): once you have a Zenodo concept DOI for the software,
     add it here so the code itself is citable, and consider adding a
     CITATION.cff file to the repo root for GitHub's "Cite this repository". -->

## License

`gsf` is released under the GNU General Public License v3.0.
See [LICENSE.md](LICENSE.md) and [COPYING.md](COPYING.md).

## Acknowledgments

Some routines are adapted from the [pynbody](https://github.com/pynbody/pynbody)
package (attributed in the source where used). The repository skeleton was
generated with the
[SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
