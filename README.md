# gsf — GalacticStructureFinder

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- TODO after you connect the repo to Zenodo and tag a release,
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

The Python dependencies (`numpy`, `scipy`, `matplotlib`, `scikit-learn`,
`click`, `pillow`) are installed automatically by pip.

## Installation

### Recommended: a clean conda/mamba environment

Because `gsf` compiles a Fortran extension at install time, the most reliable
and reproducible way to build it is inside a fresh conda/mamba environment that
provides both Python and the compiler toolchain from conda-forge. (Everything
below also works with `conda` — just replace `mamba` with `conda`.)

```bash
# 1. create and activate a clean environment
mamba create -n gsf python=3.12
mamba activate gsf

# 2. install the build toolchain from conda-forge
#    (fortran-compiler pulls gfortran + the OpenMP runtime for your platform)
mamba install -c conda-forge fortran-compiler cmake
```

With the environment active, install `gsf` from source:

```bash
git clone https://github.com/aobr/gsf2.git
cd gsf2
python -m pip install .
```

The install step compiles `gsf/twobody.f95` into the `_twobody` extension
through `cmake` and `gfortran` automatically — there is no separate manual
compilation step.

> **Tip.** Run `gsf` (and, if you use it, `ipython`/`jupyter`) from *inside*
> this environment. If `import gsf` works under `python` but fails under
> `ipython` with `ModuleNotFoundError: No module named '_twobody'`, your
> `ipython` is being picked up from a *different* environment — install it in
> this one (`mamba install ipython`) or launch it as `python -m IPython`.

### Alternative: system compilers

If you prefer to use your system Python, install the toolchain with your package
manager first. On Debian/Ubuntu:

```bash
sudo apt install gfortran cmake libgomp1
python -m pip install .
```

### From PyPI

<!-- TODO (name pending): confirm the exact registered PyPI distribution name
     and then update the command below. If the published name
     is "gsf2", this line becomes: python -m pip install gsf2
     (The import name in your code stays `import gsf` regardless.) -->

Once released on PyPI:

```bash
python -m pip install PACKAGE_NAME_PENDING
```

## Quick start

`gsf` exposes three functions, each usable both from Python and from the command
line. In all cases the three required inputs are the star, gas, and dark matter
particle files (in that order), with units of Msun (mass), kpc (positions), and
km/s (velocities).

The first step of any run is also the most expensive: `gsf` computes the
gravitational potential at every stellar particle by direct two-body summation.
This part is OpenMP-parallelized, so exporting `OMP_NUM_THREADS` (up to the
number of cores on your machine) speeds it up substantially:

```bash
export OMP_NUM_THREADS=8   # number of threads for the potential calculation
```

### `gsf` — decompose with a fixed number of components

Runs a single Gaussian-mixture decomposition for a given `number_of_clusters`,
produces the moment maps, and names the components.

**Python:**

```python
from gsf import gsf

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

**Command line:**

```bash
gsf sim1.halo_1.star.dat sim1.halo_1.gas.dat sim1.halo_1.dark.dat \
    --number_of_clusters 3 --out_dir output/ --plot
```

### `gsf_loop` — find the optimal number of components

Scans the number of components from 1 to 15, applies the `st` (elbow) and
modified-ICL criteria, and automatically selects and names the optimal model.
`Kmin`/`Kmax` bound the search, and `order_by` (`'st'` or `'mICL'`) decides how
the two selections are combined.

**Python:**

```python
from gsf import gsf_loop

gsf_loop(
    "sim1.halo_1.star.dat",
    "sim1.halo_1.gas.dat",
    "sim1.halo_1.dark.dat",
    varlist="jzjc,jpjc,e",
    out_dir="output/",
    Kmin=2, Kmax=10, order_by="st",
)
```

**Command line:** use the `--doloop` flag on the same `gsf` command:

```bash
gsf sim1.halo_1.star.dat sim1.halo_1.gas.dat sim1.halo_1.dark.dat \
    --doloop --Kmin 2 --Kmax 10 --order_by st --out_dir output/
```

### `tag_components` — (re)name the components of a decomposition

`gsf` and `gsf_loop` already name the components automatically; `tag_components`
is exposed separately so an existing decomposition can be re-named (for example
with a different field of view) without re-running the clustering. Its two inputs
are the temporary per-particle file and the decomposition file produced by a run.
It writes the component tags, a LaTeX summary table, and (unless disabled) the
moment-map and bar/phi diagnostic figures.

**Python:**

```python
from gsf import tag_components

tag_components(
    "output/sim1.halo_1.star.tmp",           # per-particle properties
    "output/sim1.halo_1.star.gmm_on_jzjcjpjce.scikit_gmm_full_3clusters_white.dat",  # a GMM decomposition
    fov=80.,
    make_plots=True,
)
```

**Command line:**

```bash
gsf-tag output/sim1.halo_1.star.tmp \
        output/sim1.halo_1.star.gmm_on_jzjcjpjce.scikit_gmm_full_3clusters_white.dat --fov 80
```

See `gsf --help` and `gsf-tag --help` for the full list of options.

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
