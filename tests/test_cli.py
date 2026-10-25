import json
from pathlib import Path

from conftest import FakeExtractor
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


def test_ingest_writes_json(sample_pdf):
    result = runner.invoke(app, ["ingest", str(sample_pdf)])
    assert result.exit_code == 0
    saved = json.loads(Path("data/maps/sample.json").read_text(encoding="utf-8"))
    assert saved["page_count"] == 2
    assert "Reed Richards" in saved["pages"][0]["text"]
    assert saved["pages"][1]["text"] is None
    assert saved["pages"][1]["source"] == "none"


def test_ingest_suggests_ocr_for_image_pages(sample_cbz):
    result = runner.invoke(app, ["ingest", str(sample_cbz)])
    assert "add --ocr" in result.stdout


def test_ingest_ocr_reads_every_page(sample_cbz, fake_extractor):
    result = runner.invoke(app, ["ingest", str(sample_cbz), "--ocr"])
    assert result.exit_code == 0
    saved = json.loads(Path("data/maps/sample.json").read_text(encoding="utf-8"))
    assert [p["text"] for p in saved["pages"]] == ["BOOM!", "BOOM!", "BOOM!"]
    assert saved["model"] == "qwen2.5vl:3b"
    assert "page 3: model" in result.stdout


def test_ingest_ocr_second_run_hits_cache(sample_cbz, fake_extractor):
    runner.invoke(app, ["ingest", str(sample_cbz), "--ocr"])
    fake_extractor.calls = 0
    result = runner.invoke(app, ["ingest", str(sample_cbz), "--ocr"])
    assert fake_extractor.calls == 0
    assert "page 1: cache" in result.stdout


def test_ingest_custom_output_path(sample_pdf, tmp_path):
    out = tmp_path / "somewhere" / "book.json"
    result = runner.invoke(app, ["ingest", str(sample_pdf), "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_ingest_ocr_when_ollama_is_not_running(sample_cbz, monkeypatch):
    class Down(FakeExtractor):
        def transcribe(self, image):
            raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr("litmapr.cli.make_extractor", lambda model: Down())
    result = runner.invoke(app, ["ingest", str(sample_cbz), "--ocr"])
    assert result.exit_code == 1
    assert "re-run to continue" in result.stdout
