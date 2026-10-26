from abc import ABC, abstractmethod
from collections.abc import Iterator


class TextLLM(ABC):
    """A language model that answers prompts."""

    model: str

    @abstractmethod
    def complete_json(self, prompt: str, schema: dict) -> str:
        """Return the model's answer as a JSON string shaped like `schema`."""

    @abstractmethod
    def complete_text(self, prompt: str) -> str:
        """Return the model's answer as plain text (used by the narrator)."""

    def stream_text(self, prompt: str) -> Iterator[str]:
        """Yield the answer in pieces as it is written. Models that can't
        stream just give the whole answer as one piece."""
        yield self.complete_text(prompt)


class Embedder(ABC):
    """Turns text into vectors: lists of numbers where similar meanings
    end up close together. Used to find duplicates and callbacks."""

    model: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in the same order."""


def cosine(a: list[float], b: list[float]) -> float:
    """How similar two vectors point: 1.0 = same direction, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
