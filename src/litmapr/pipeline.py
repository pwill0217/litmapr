from collections.abc import Iterable, Iterator
from dataclasses import replace

from litmapr.cache import PageCache
from litmapr.extract.base import TextExtractor
from litmapr.ingest.base import Page


def read_pages(
    pages: Iterable[Page],
    extractor: TextExtractor,
    cache: PageCache,
) -> Iterator[tuple[Page, str]]:
    """Fill in text and scene for every page, using the cache when possible.

    Yields (page, source) pairs. source says where the words came from:
    "text_layer" (already in the file), "cache" (read before), or "model" (just read).
    """
    for page in pages:
        if page.text:
            yield page, "text_layer"
            continue

        cached = cache.get(page.number)
        if cached is not None:
            yield replace(page, text=cached["text"], scene=cached["scene"]), "cache"
            continue

        page = extractor.extract(page)
        cache.put(page.number, page.text, page.scene)
        yield page, "model"
