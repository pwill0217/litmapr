from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace

from litmapr.cache import PageCache
from litmapr.extract.base import PageReadError, TextExtractor
from litmapr.ingest.base import Page


def read_pages(
    pages: Iterable[Page],
    extractor: TextExtractor,
    cache: PageCache,
    wanted: Callable[[int], bool] | None = None,
) -> Iterator[tuple[Page, str]]:
    """Fill in text and scene for every page, using the cache when possible.

    Yields (page, source) pairs. source says where the words came from:
    "text_layer" (already in the file), "cache" (read before), "model" (just read),
    or "failed" (the model couldn't read it; not cached, so the next run retries it).
    Pages `wanted` says no to come back as "skipped", without asking the model.
    """
    for page in pages:
        if wanted is not None and not wanted(page.number):
            yield page, "skipped"
            continue
        if page.text:
            yield page, "text_layer"
            continue

        cached = cache.get(page.number)
        if cached is not None:
            yield replace(page, text=cached["text"], scene=cached["scene"]), "cache"
            continue

        try:
            page = extractor.extract(page)
        except PageReadError:
            yield page, "failed"
            continue
        cache.put(page.number, page.text, page.scene)
        yield page, "model"
