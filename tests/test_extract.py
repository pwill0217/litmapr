import io

from conftest import FakeExtractor, tiny_png
from PIL import Image

from litmapr.extract.ollama_vision import OllamaVisionExtractor, shrink
from litmapr.ingest.base import Page


def test_extract_fills_text_and_scene():
    page = Page(number=1, image=tiny_png("red"), text=None)
    result = FakeExtractor(text="BOOM!", scene="Carol walks out.").extract(page)
    assert result.text == "BOOM!"
    assert result.scene == "Carol walks out."


def test_extract_returns_a_copy():
    page = Page(number=1, image=tiny_png("red"), text=None)
    FakeExtractor().extract(page)
    assert page.text is None  # the original page is untouched


def test_extract_skips_pages_that_already_have_text():
    page = Page(number=1, image=tiny_png("red"), text="Real text layer")
    fake = FakeExtractor()
    result = fake.extract(page)
    assert result.text == "Real text layer"
    assert fake.calls == 0  # the model was never asked


def test_shrink_scales_tall_pages_down():
    tall = io.BytesIO()
    Image.new("RGB", (1000, 3000), "white").save(tall, format="PNG")
    small = Image.open(io.BytesIO(shrink(tall.getvalue(), max_height=1600)))
    assert small.size == (533, 1600)  # same shape, 1600 tall


def test_shrink_leaves_small_pages_alone():
    small = Image.open(io.BytesIO(shrink(tiny_png("blue"), max_height=1600)))
    assert small.size == (10, 10)


def test_no_text_answer_becomes_none(monkeypatch):
    extractor = OllamaVisionExtractor()
    monkeypatch.setattr(extractor, "_ask", lambda prompt, image: "NO TEXT")
    assert extractor.transcribe(b"ignored") is None
