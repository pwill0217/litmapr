import io

import pymupdf
import pytest
from cbz import ComicInfo, PageInfo
from PIL import Image

from litmapr.extract.base import TextExtractor


def tiny_png(color: str) -> bytes:
    """Make a 10x10 image in memory and return its PNG bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_cbz(tmp_path):
    """Build a 3-page CBZ in a temporary folder and return its path."""
    pages = [PageInfo.loads(tiny_png(color)) for color in ("red", "green", "blue")]
    comic = ComicInfo.from_pages(pages=pages)
    path = tmp_path / "sample.cbz"
    comic.save(path)
    return path


@pytest.fixture
def sample_pdf(tmp_path):
    """Build a 2-page PDF: page 1 has text, page 2 is blank (like a scan)."""
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Reed Richards builds the Bridge.")
    document.new_page()
    path = tmp_path / "sample.pdf"
    document.save(path)
    document.close()
    return path


class FakeExtractor(TextExtractor):
    """Stands in for the real vision model in tests: no Ollama, no GPU."""

    def __init__(self, text: str | None = "BOOM!", scene: str = "A test scene."):
        self.text = text
        self.scene = scene
        self.calls = 0  # how many times the "model" was asked

    def transcribe(self, image: bytes) -> str | None:
        self.calls += 1
        return self.text

    def describe(self, image: bytes) -> str:
        self.calls += 1
        return self.scene


@pytest.fixture
def fake_extractor(monkeypatch):
    """Make `litmapr inspect --ocr` use a FakeExtractor instead of Ollama."""
    fake = FakeExtractor()
    monkeypatch.setattr("litmapr.cli.make_extractor", lambda model: fake)
    return fake


@pytest.fixture(autouse=True)
def run_in_temp_folder(tmp_path, monkeypatch):
    """Run every test inside its own temp folder.

    Commands write to data/cache and data/maps relative to the current folder;
    this keeps tests from filling the real project's data/ folder.
    """
    monkeypatch.chdir(tmp_path)
