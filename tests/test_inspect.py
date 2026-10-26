from typer.testing import CliRunner

from litmapr.cli import app

runner = CliRunner()


def test_inspect_shows_page_text(sample_pdf):
    result = runner.invoke(app, ["inspect", str(sample_pdf)])
    assert result.exit_code == 0
    assert "Reed Richards" in result.stdout


def test_inspect_page_without_text(sample_pdf):
    result = runner.invoke(app, ["inspect", str(sample_pdf), "--page", "2"])
    assert result.exit_code == 0
    assert "no text layer" in result.stdout


def test_inspect_page_past_the_end(sample_pdf):
    result = runner.invoke(app, ["inspect", str(sample_pdf), "--page", "5"])
    assert result.exit_code == 1
    assert "doesn't exist" in result.stdout


def test_inspect_page_zero_rejected(sample_pdf):
    result = runner.invoke(app, ["inspect", str(sample_pdf), "--page", "0"])
    assert result.exit_code != 0


def test_inspect_saves_image(sample_cbz, tmp_path):
    out = tmp_path / "page1.png"
    result = runner.invoke(app, ["inspect", str(sample_cbz), "--save-image", str(out)])
    assert result.exit_code == 0
    assert out.read_bytes().startswith(b"\x89PNG")


def test_inspect_saves_image_into_new_folder(sample_cbz, tmp_path):
    out = tmp_path / "new_folder" / "page1.png"
    result = runner.invoke(app, ["inspect", str(sample_cbz), "--save-image", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_inspect_without_ocr_suggests_it(sample_cbz):
    result = runner.invoke(app, ["inspect", str(sample_cbz)])
    assert "add --ocr" in result.stdout


def test_inspect_ocr_prints_text_and_scene(sample_cbz, fake_extractor):
    result = runner.invoke(app, ["inspect", str(sample_cbz), "--ocr"])
    assert result.exit_code == 0
    assert "BOOM!" in result.stdout
    assert "A test scene." in result.stdout


def test_inspect_ocr_page_with_no_text(sample_cbz, fake_extractor):
    fake_extractor.text = None
    result = runner.invoke(app, ["inspect", str(sample_cbz), "--ocr"])
    assert "(no text on this page)" in result.stdout


def test_inspect_ocr_when_ollama_is_not_running(sample_cbz, monkeypatch):
    def not_running(model):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr("litmapr.cli.make_extractor", not_running)
    result = runner.invoke(app, ["inspect", str(sample_cbz), "--ocr"])
    assert result.exit_code == 1
    assert "Is it running?" in result.stdout


def test_ingest_carries_on_past_an_unreadable_page(sample_cbz, monkeypatch):
    from conftest import FakeExtractor, tiny_png

    from litmapr.extract.base import PageReadError

    class StuckOnGreen(FakeExtractor):
        def transcribe(self, image):
            if image == tiny_png("green"):
                raise PageReadError("stuck")
            return super().transcribe(image)

    monkeypatch.setattr("litmapr.cli.make_extractor", lambda model: StuckOnGreen())
    result = runner.invoke(app, ["ingest", str(sample_cbz), "--ocr"])
    assert result.exit_code == 0, result.stdout
    assert "page 2: failed, skipped for now" in result.stdout
    assert "page 3: model" in result.stdout
    assert "1 page(s) couldn't be read (2)" in result.stdout
