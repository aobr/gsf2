from gsf.__main__ import main

from click.testing import CliRunner


def test_gsf_cli():
    runner = CliRunner()
    result = runner.invoke(main, ['./sim1/sim1.halo_1.star.dat', './sim1/sim1.halo_1.gas.dat', './sim1/sim1.halo_1.dark.dat'])
    assert result.exit_code == 0
