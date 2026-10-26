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


def test_shrink_keeps_double_page_spreads_within_budget():
    from litmapr.extract.ollama_vision import MAX_PIXELS

    spread = io.BytesIO()
    Image.new("RGB", (4000, 3000), "white").save(spread, format="PNG")
    small = Image.open(io.BytesIO(shrink(spread.getvalue(), max_height=1600)))
    assert small.width * small.height <= MAX_PIXELS * 1.01
    assert abs(small.width / small.height - 4000 / 3000) < 0.01  # same shape


def test_normal_page_is_only_limited_by_height():
    page = io.BytesIO()
    Image.new("RGB", (1988, 3056), "white").save(page, format="PNG")
    small = Image.open(io.BytesIO(shrink(page.getvalue(), max_height=1600)))
    assert small.height == 1600


def test_repeating_model_gets_a_second_try(monkeypatch):
    import ollama

    from litmapr.extract.ollama_vision import OllamaVisionExtractor

    calls = []

    def fake_chat(**request):
        calls.append(request["options"])
        if len(calls) == 1:
            raise ollama.ResponseError("prediction aborted, token repeat limit reached")
        return type("R", (), {"message": type("M", (), {"content": " WHAM! "})()})()

    monkeypatch.setattr("litmapr.extract.ollama_vision.ollama.chat", fake_chat)
    assert OllamaVisionExtractor().transcribe(tiny_png("red")) == "WHAM!"
    assert calls[0]["temperature"] == 0
    assert calls[1]["repeat_penalty"] > 1  # the retry discourages repeating


def test_page_that_keeps_repeating_raises_page_read_error(monkeypatch):
    import ollama
    import pytest

    from litmapr.extract.base import PageReadError
    from litmapr.extract.ollama_vision import OllamaVisionExtractor

    def always_stuck(**request):
        raise ollama.ResponseError("prediction aborted, token repeat limit reached")

    monkeypatch.setattr("litmapr.extract.ollama_vision.ollama.chat", always_stuck)
    monkeypatch.setattr(
        "litmapr.extract.ollama_vision.ollama.generate", lambda **k: None
    )
    with pytest.raises(PageReadError):
        OllamaVisionExtractor().transcribe(tiny_png("red"))


def test_stuck_model_is_reset_and_page_retried(monkeypatch):
    import ollama

    from litmapr.extract.ollama_vision import ATTEMPTS, OllamaVisionExtractor

    state = {"fresh": False, "unloads": 0}

    def chat(**request):
        if not state["fresh"]:
            raise ollama.ResponseError("prediction aborted, token repeat limit reached")
        return type("R", (), {"message": type("M", (), {"content": "FIXED"})()})()

    def generate(model, keep_alive):
        assert keep_alive == 0  # this is how Ollama is told to unload a model
        state["unloads"] += 1
        state["fresh"] = True

    monkeypatch.setattr("litmapr.extract.ollama_vision.ollama.chat", chat)
    monkeypatch.setattr("litmapr.extract.ollama_vision.ollama.generate", generate)
    assert OllamaVisionExtractor().transcribe(tiny_png("red")) == "FIXED"
    assert state["unloads"] == 1
    assert len(ATTEMPTS) == 3


def test_other_ollama_errors_still_stop_the_run(monkeypatch):
    import ollama
    import pytest

    from litmapr.extract.ollama_vision import OllamaVisionExtractor

    def no_model(**request):
        raise ollama.ResponseError("model 'nope' not found")

    monkeypatch.setattr("litmapr.extract.ollama_vision.ollama.chat", no_model)
    with pytest.raises(ollama.ResponseError):
        OllamaVisionExtractor().transcribe(tiny_png("red"))


def test_first_try_keeps_ollamas_default_context(monkeypatch):
    # Raising num_ctx made qwen2.5vl loop on full-size pages (measured on a real page).
    from litmapr.extract.ollama_vision import OllamaVisionExtractor

    seen = []

    def fake_chat(**request):
        seen.append(request["options"])
        return type("R", (), {"message": type("M", (), {"content": "OK"})()})()

    monkeypatch.setattr("litmapr.extract.ollama_vision.ollama.chat", fake_chat)
    OllamaVisionExtractor().transcribe(tiny_png("red"))
    assert "num_ctx" not in seen[0]


def test_last_resort_sends_a_smaller_page(monkeypatch):
    import ollama

    from litmapr.extract.ollama_vision import OllamaVisionExtractor

    heights = []

    def stuck_until_small(**request):
        picture = Image.open(io.BytesIO(request["messages"][0]["images"][0]))
        heights.append(picture.height)
        if len(heights) < 3:
            raise ollama.ResponseError("prediction aborted, token repeat limit reached")
        return type("R", (), {"message": type("M", (), {"content": "BOOM"})()})()

    monkeypatch.setattr("litmapr.extract.ollama_vision.ollama.chat", stuck_until_small)
    monkeypatch.setattr(
        "litmapr.extract.ollama_vision.ollama.generate", lambda **k: None
    )
    tall = io.BytesIO()
    Image.new("RGB", (1988, 3056), "white").save(tall, format="PNG")
    assert OllamaVisionExtractor().transcribe(tall.getvalue()) == "BOOM"
    assert heights == [1600, 1600, 1200]
