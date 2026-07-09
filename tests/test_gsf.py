# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end smoke test of the Python API on the bundled example galaxy."""
import glob
import os

import pytest


def test_gsf_default_nk2(galaxy_files, tmp_path):
    """Run the default, simplest decomposition (2 components) end to end and
    check that the expected products are written: the GMM decomposition, the
    component tags, the LaTeX summary table, and the per-model summary."""
    gsf = pytest.importorskip(
        "gsf", reason="gsf (with the compiled _twobody extension) is not installed"
    )

    file_star, file_gas, file_dark = galaxy_files
    out_dir = str(tmp_path) + os.sep

    gsf.gsf(
        file_star, file_gas, file_dark,
        number_of_clusters=2,     # the default, simplest case
        out_dir=out_dir,
        plot=False,               # skip the GMM diagnostic scatter to stay lean
        verbose=False,
    )

    # A 2-component decomposition and its naming products must exist.
    assert glob.glob(out_dir + "*2clusters*.dat"), "no 2-cluster decomposition written"
    assert glob.glob(out_dir + "*_tags.dat"), "components were not named"
    assert glob.glob(out_dir + "*table.tex"), "no LaTeX summary table written"
    assert glob.glob(out_dir + "*summary.dat"), "no per-model summary written"
