from typer.testing import CliRunner

from litmapr.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "serve" in result.stdout


def test_ingest_counts_pages(sample_cbz):
    result = runner.invoke(app, ["ingest", str(sample_cbz)])
    assert result.exit_code == 0
    assert "found 3 pages" in result.stdout


def test_ingest_missing_file_fails_cleanly():
    result = runner.invoke(app, ["ingest", "does_not_exist.cbz"])
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_ingest_pdf(sample_pdf):
    result = runner.invoke(app, ["ingest", str(sample_pdf)])
    assert result.exit_code == 0
    assert "found 2 pages" in result.stdout
    assert "1 with text" in result.stdout


def test_ingest_unsupported_type(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a comic")
    result = runner.invoke(app, ["ingest", str(notes)])
    assert result.exit_code == 1
    assert "Unsupported file type" in result.stdout
