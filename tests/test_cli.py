from click.testing import CliRunner
from razorback.cli import cli


def test_empty_main():
    runner = CliRunner()
    res = runner.invoke(cli, [])
    assert res.exit_code == 2
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0


def test_path():
    runner = CliRunner()
    res = runner.invoke(cli, ["path"])
    assert res.exit_code == 2
    res = runner.invoke(cli, ["path", "base"])
    assert res.exit_code == 0


def test_version():
    runner = CliRunner()
    res = runner.invoke(cli, ["version"])
    assert res.exit_code == 0
