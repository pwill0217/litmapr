import hashlib
import json
from collections.abc import Iterator

import ollama

from litmapr.cache import ResponseCache
from litmapr.llm.base import Embedder, TextLLM

DEFAULT_MODEL = "qwen3.6:35b-a3b-coding"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
# The narrator only retells facts already in its prompt, so a small, fast model
# that fits on the GPU is enough. Extraction and merging keep the big one.
# An "instruct" model: it answers straight away. Plain qwen3:4b is a thinking
# model that writes out its reasoning first, which the narrator would read aloud.
DEFAULT_NARRATOR_MODEL = "qwen3:4b-instruct"
CONTEXT_WINDOW = 16384  # tokens
MAX_ANSWER = 6144  # tokens the answer may use


class OllamaTextLLM(TextLLM):
    """A text model running locally in Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def complete_json(self, prompt: str, schema: dict) -> str:
        # Structured output: Ollama only lets the model produce JSON that
        # fits this schema. Pydantic still checks it afterwards.
        return self._chat(prompt, format=schema)

    def complete_text(self, prompt: str) -> str:
        return self._chat(prompt).strip()

    def stream_text(self, prompt: str) -> Iterator[str]:
        request = self._request(prompt, stream=True)
        request["options"]["temperature"] = 0.6  # a narrator can vary its wording
        try:
            parts = ollama.chat(**request, think=False)
            first = next(parts, None)  # errors about thinking show up here
        except ollama.ResponseError as error:
            if "think" not in str(error).lower():
                raise
            parts = ollama.chat(**request)
            first = next(parts, None)
        if first is not None:
            yield first.message.content
        for part in parts:
            yield part.message.content

    def _request(self, prompt: str, **extra) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0,
                # How much text the model can hold at once: prompt + answer.
                # Ollama's default is small; long answers got cut off mid-JSON.
                "num_ctx": CONTEXT_WINDOW,
                "num_predict": MAX_ANSWER,
            },
            **extra,
        }

    def _chat(self, prompt: str, **extra) -> str:
        request = self._request(prompt, **extra)
        try:
            # Thinking makes answers slower and isn't needed here.
            response = ollama.chat(**request, think=False)
        except ollama.ResponseError as error:
            if "think" not in str(error).lower():
                raise
            # This model has no thinking switch at all; ask without it.
            response = ollama.chat(**request)
        return response.message.content


class OllamaEmbedder(Embedder):
    """Embeddings from a small Ollama model, cached so each text is embedded once."""

    def __init__(
        self, model: str = DEFAULT_EMBED_MODEL, cache: ResponseCache | None = None
    ):
        self.model = model
        self.cache = cache or ResponseCache(f"embed-{model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(text.encode()).hexdigest()[:20] for text in texts]
        missing = [i for i, key in enumerate(keys) if self.cache.get(key) is None]
        if missing:
            response = ollama.embed(model=self.model, input=[texts[i] for i in missing])
            for i, vector in zip(missing, response.embeddings, strict=True):
                self.cache.put(keys[i], json.dumps(vector))
        return [json.loads(self.cache.get(key)) for key in keys]
