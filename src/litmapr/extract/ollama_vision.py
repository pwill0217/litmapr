import io

import ollama
from PIL import Image

from litmapr.extract.base import PageReadError, TextExtractor

DEFAULT_MODEL = "qwen2.5vl:3b"

# Tested on real comic pages: 1200px lost small text, 1600px read it correctly.
MAX_HEIGHT = 1600
# A 1600px-tall page is about 1.7 million pixels. Capping the total keeps a
# double-page spread inside the model's context window (vision models turn
# every ~28x28 pixel patch into one token).
MAX_PIXELS = 1_800_000
# What to try for each page, in order. Measured on real pages with a 6 GB GPU:
# 1. full size, exact: reads small lettering correctly (about 30s a page).
#    Ollama's default context window is kept on purpose: raising it to 8192
#    made the model loop on full-size pages within two seconds.
# 2. if the model gets stuck repeating itself: a little randomness and a
#    penalty for repeats, which broke the loop in testing.
# 3. last resort: a smaller page. Faster, but it misreads more.
ATTEMPTS = [
    (MAX_HEIGHT, {"temperature": 0}),
    (MAX_HEIGHT, {"temperature": 0.4, "repeat_penalty": 1.2}),
    (1200, {"temperature": 0}),
]

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
    """Scale a page down (keeping its shape) and return JPEG bytes.

    Two limits: the height (so small lettering stays readable), and the total
    pixels (so a double-page spread, twice as wide, doesn't turn into more
    image tokens than the model can hold).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = min(
        1.0,
        max_height / image.height,
        (MAX_PIXELS / (image.width * image.height)) ** 0.5,
    )
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
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
        answer = self._try_all(prompt, image)
        if answer is not None:
            return answer
        # Every attempt looped. On long runs that usually means Ollama's model
        # process has gone bad (seen after ~60 pages: every page failing the
        # same way). Unloading the model makes Ollama start it fresh.
        self.reset()
        answer = self._try_all(prompt, image)
        if answer is None:
            raise PageReadError("the model kept repeating itself, even after a reset")
        return answer

    def reset(self) -> None:
        """Unload the model from memory; Ollama reloads it on the next request."""
        try:
            ollama.generate(model=self.model, keep_alive=0)
        except ollama.ResponseError:
            pass  # nothing loaded, or already unloading: fine either way

    def _try_all(self, prompt: str, image: bytes) -> str | None:
        """Try each setting in ATTEMPTS. None if the model looped every time."""
        for height, options in ATTEMPTS:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [shrink(image, min(height, self.max_height))],
                        }
                    ],
                    options=options,
                )
            except ollama.ResponseError as error:
                if "repeat" not in str(error).lower():
                    raise  # a real problem (wrong model name...), not this page's fault
                continue
            return response.message.content.strip()
        return None
