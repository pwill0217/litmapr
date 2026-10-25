from conftest import FakeExtractor, tiny_png

from litmapr.cache import PageCache
from litmapr.ingest.base import Page
from litmapr.pipeline import read_pages


def make_pages():
    return [
        Page(number=1, image=tiny_png("red"), text=None),
        Page(number=2, image=tiny_png("green"), text="Already has text"),
    ]


def test_model_reads_pages_without_text(tmp_path):
    fake = FakeExtractor(text="BOOM!", scene="A scene.")
    cache = PageCache("abc", "fake", cache_dir=tmp_path)
    results = list(read_pages(make_pages(), fake, cache))
    assert [source for _, source in results] == ["model", "text_layer"]
    assert results[0][0].text == "BOOM!"


def test_second_run_uses_cache_not_model(tmp_path):
    first = FakeExtractor()
    list(read_pages(make_pages(), first, PageCache("abc", "fake", cache_dir=tmp_path)))

    second = FakeExtractor()
    results = list(
        read_pages(make_pages(), second, PageCache("abc", "fake", cache_dir=tmp_path))
    )
    assert second.calls == 0  # nothing re-read
    assert results[0][1] == "cache"
    assert results[0][0].scene == "A test scene."


def test_crash_keeps_finished_pages(tmp_path):
    class CrashesOnPageTwo(FakeExtractor):
        def transcribe(self, image):
            if image == tiny_png("green"):
                raise ConnectionError("Ollama went away")
            return super().transcribe(image)

    pages = [
        Page(number=1, image=tiny_png("red"), text=None),
        Page(number=2, image=tiny_png("green"), text=None),
    ]
    cache = PageCache("abc", "fake", cache_dir=tmp_path)
    try:
        list(read_pages(pages, CrashesOnPageTwo(), cache))
    except ConnectionError:
        pass

    reopened = PageCache("abc", "fake", cache_dir=tmp_path)
    assert reopened.get(1) is not None  # page 1 survived the crash
    assert reopened.get(2) is None
