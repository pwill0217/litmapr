from abc import ABC, abstractmethod
from dataclasses import replace

from litmapr.ingest.base import Page


class TextExtractor(ABC):
    """Turns a page image into words: what it says, and what happens."""

    @abstractmethod
    def transcribe(self, image: bytes) -> str | None:
        """Return the exact text on the page, or None if there is none."""

    @abstractmethod
    def describe(self, image: bytes) -> str:
        """Return a short description of what happens on the page."""

    def extract(self, page: Page) -> Page:
        """Return a copy of the page with text and scene filled in.

        Pages that already have text (a PDF's text layer) are left alone:
        that text is exact, so there's nothing for a vision model to add.
        """
        if page.text:
            return page
        return replace(
            page,
            text=self.transcribe(page.image),
            scene=self.describe(page.image),
        )
