import io

import ollama
from PIL import Image

from litmapr.extract.base import TextExtractor

DEFAULT_MODEL = "qwen2.5vl:3b"

# Tested on real comic pages: 1200px lost small text, 1600px read it correctly.
MAX_HEIGHT = 1600

# One job per prompt. Asking a small model to transcribe AND describe
# in one call made it worse at both.
TRANSCRIBE_PROMPT = (
    "Transcribe all text on this comic page exactly as written, in reading order. "
    "Include captions, speech balloons and sound effects. "
    "Do not describe the art. Output plain text only, no Markdown. "
    "If there is no text, reply with: NO TEXT."
)

SCENE_PROMPT = (
    "Describe what happens on this comic page in two or three plain sentences. "
    "Focus on actions and events, including any important action that has "
    "no dialogue. Mention visible identifying details of characters, such as "
    "costumes, logos, or distinctive features. "
    "Do not transcribe the speech balloons. No Markdown."
)

NO_TEXT = "NO TEXT"


def shrink(image_bytes: bytes, max_height: int = MAX_HEIGHT) -> bytes:
    """Scale a page down to max_height (keeping its shape) and return JPEG bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image.height > max_height:
        width = round(image.width * max_height / image.height)
        image = image.resize((width, max_height))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class OllamaVisionExtractor(TextExtractor):
    """Reads page images with a vision model running locally in Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL, max_height: int = MAX_HEIGHT):
        self.model = model
        self.max_height = max_height

    def transcribe(self, image: bytes) -> str | None:
        answer = self._ask(TRANSCRIBE_PROMPT, image)
        return None if answer.upper() == NO_TEXT else answer

    def describe(self, image: bytes) -> str:
        return self._ask(SCENE_PROMPT, image)

    def _ask(self, prompt: str, image: bytes) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [shrink(image, self.max_height)],
                }
            ],
            options={"temperature": 0},  # same page in -> same text out
        )
        return response.message.content.strip()
