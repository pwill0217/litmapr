from typer.testing import CliRunner

from litmapr.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "serve" in result.stdout


def test_ingest_runs():
    result = runner.invoke(app, ["ingest", "test.cbz"])
    assert result.exit_code == 0
    assert "test.cbz" in result.stdout