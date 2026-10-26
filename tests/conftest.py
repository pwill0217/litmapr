import io
import json

import pymupdf
import pytest
from cbz import ComicInfo, PageInfo
from PIL import Image

from litmapr.extract.base import TextExtractor
from litmapr.llm.base import Embedder, TextLLM


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


class FakeLLM(TextLLM):
    """Stands in for the text model: returns a canned JSON answer."""

    model = "fake-llm"

    def __init__(self, answer: dict | str):
        self.answer = answer if isinstance(answer, str) else json.dumps(answer)
        self.prompts: list[str] = []  # every prompt it was sent

    def complete_json(self, prompt: str, schema: dict) -> str:
        self.prompts.append(prompt)
        return self.answer

    def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


ARMAGEDDON_ANSWER = {
    "entities": [
        {
            "name": "Thaddeus Ross",
            "kind": "character",
            "aliases": ["Red Hulk"],
            "description": "Former general who took over Latveria.",
        },
        {"name": "Latveria", "kind": "location"},
        {"name": "Carol Danvers", "kind": "character", "aliases": ["Captain Marvel"]},
    ],
    "mentions": [
        {"entity": "Red Hulk", "page": 1, "summary": "Takes over Latveria."},
        {"entity": "Latveria", "page": 1, "summary": "Renamed New America."},
        {
            "entity": "Carol Danvers",
            "page": 2,
            "summary": "Learns her powers are diminished.",
            "quote": "HOW DIMINISHED?",
        },
    ],
    "relations": [
        {
            "source": "Thaddeus Ross",
            "target": "Latveria",
            "kind": "other",
            "label": "conquers and renames it",
            "sentiment": -0.6,
            "page": 1,
        }
    ],
}


@pytest.fixture
def pages_json(tmp_path):
    """A small file shaped like `ingest` output."""
    document = {
        "format_version": 1,
        "source_file": "Avengers_Armageddon005.cbz",
        "source_hash": "abc123",
        "model": "qwen2.5vl:3b",
        "page_count": 3,
        "pages": [
            {
                "number": 1,
                "text": "Ross took over Latveria.",
                "scene": None,
                "source": "model",
            },
            {
                "number": 2,
                "text": "HOW DIMINISHED?",
                "scene": "Carol frowns.",
                "source": "model",
            },
            {"number": 3, "text": None, "scene": None, "source": "none"},
        ],
    }
    path = tmp_path / "pages.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def fake_llm(monkeypatch):
    """Make `litmapr extract` use a FakeLLM instead of Ollama."""
    fake = FakeLLM(ARMAGEDDON_ANSWER)
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: fake)
    return fake


class FakeEmbedder(Embedder):
    """Stand-in embeddings: one number per keyword, 1 if the text contains it.

    Texts that share keywords point the same way, so they look "similar",
    which is all the duplicate and callback finders need.
    """

    model = "fake-embed"

    def __init__(self, keywords: list[str]):
        self.keywords = [k.lower() for k in keywords]
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0 if k in t.lower() else 0.0 for k in self.keywords] for t in texts]


class ScriptedLLM(FakeLLM):
    """Answers each prompt with the next answer in a list."""

    def __init__(self, answers: list):
        super().__init__("{}")
        self.answers = [a if isinstance(a, str) else json.dumps(a) for a in answers]

    def complete_json(self, prompt: str, schema: dict) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)

    def complete_text(self, prompt: str) -> str:
        return self.complete_json(prompt, {})


@pytest.fixture
def tagged_cbz(tmp_path):
    """A 2-page CBZ whose ComicInfo.xml says FF #5, March 2012."""
    pages = [PageInfo.loads(tiny_png(color)) for color in ("red", "green")]
    comic = ComicInfo.from_pages(
        pages=pages, series="FF", number="5", year=2012, month=3
    )
    path = tmp_path / "some_download_name.cbz"
    comic.save(path)
    return path


@pytest.fixture
def omnibus_pdf(tmp_path):
    """A 6-page 'omnibus': FF #1 on pages 1-3, FF #2 on pages 4-6, with bookmarks."""
    document = pymupdf.open()
    for n in range(1, 7):
        document.new_page().insert_text((72, 72), f"Omnibus page {n}")
    document.set_toc([[1, "Introduction", 1], [1, "FF #1", 1], [1, "FF #2", 4]])
    path = tmp_path / "Hickman Omnibus.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def omnibus_cbz(tmp_path):
    """A 4-page CBZ with ComicRack-style bookmarks on pages 1 and 3."""
    pages = [PageInfo.loads(tiny_png(c)) for c in ("red", "green", "blue", "white")]
    pages[0].bookmark = "Fantastic Four #570"
    pages[2].bookmark = "Fantastic Four #571"
    path = tmp_path / "FF Omnibus.cbz"
    ComicInfo.from_pages(pages=pages).save(path)
    return path


@pytest.fixture(autouse=True)
def never_touch_real_ollama_models(monkeypatch):
    """Tests must not unload models in a real Ollama running on this computer."""
    monkeypatch.setattr("litmapr.cli.ollama.generate", lambda **kwargs: None)
