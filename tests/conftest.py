# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pytest fixtures for the gsf test-suite."""
import os
# Ensure matplotlib never tries to open a window during the tests.
os.environ.setdefault("MPLBACKEND", "Agg")

import pathlib
import pytest

SIM1 = pathlib.Path(__file__).parent / "sim1"


@pytest.fixture(scope="session")
def galaxy_files():
    """Absolute paths to the bundled example galaxy (star, gas, dark)."""
    star = SIM1 / "sim1.halo_1.star.dat"
    gas = SIM1 / "sim1.halo_1.gas.dat"
    dark = SIM1 / "sim1.halo_1.dark.dat"
    for f in (star, gas, dark):
        assert f.exists(), f"missing example galaxy file: {f}"
    return str(star), str(gas), str(dark)
