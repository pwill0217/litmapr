from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    number: int
    image: bytes
    text: str | None
    scene: str | None = None
    path: str | None = None


class Reader(ABC):
    def __init__(self, path: str | Path):
        self.path = path

    @abstractmethod
    def pages(self):
        """Yield Page objects, one per page."""
