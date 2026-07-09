# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end smoke test of the command-line interface on the example galaxy."""
import glob
import os

import pytest


def test_gsf_cli_default_nk2(galaxy_files, tmp_path):
    """Invoke the `gsf` command with its defaults (2 components) and check it
    exits cleanly and produces a named 2-component decomposition."""
    pytest.importorskip(
        "gsf", reason="gsf (with the compiled _twobody extension) is not installed"
    )
    from click.testing import CliRunner
    from gsf.__main__ import main

    file_star, file_gas, file_dark = galaxy_files
    out_dir = str(tmp_path) + os.sep

    runner = CliRunner()
    result = runner.invoke(
        main,
        [file_star, file_gas, file_dark, "--out_dir", out_dir, "--verbose", "False"],
    )

    assert result.exit_code == 0, result.output
    assert glob.glob(out_dir + "*2clusters*.dat"), "no 2-cluster decomposition written"
    assert glob.glob(out_dir + "*_tags.dat"), "components were not named"
