import hashlib
import json
from pathlib import Path

DEFAULT_CACHE_DIR = Path("data") / "cache"


def file_hash(path: str | Path) -> str:
    """Fingerprint a file by its contents (not its name).

    Renaming or moving a comic keeps the same hash, so its cache still matches.
    Editing the file changes the hash, so stale results are never reused.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        # Read in 1 MB chunks so huge files don't have to fit in memory.
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class PageCache:
    """Remembers what the model said about each page of one file.

    One JSON file per (file, model) pair, saved after every page,
    so a crash on page 18 keeps pages 1-17.
    """

    def __init__(
        self, source_hash: str, model: str, cache_dir: Path = DEFAULT_CACHE_DIR
    ):
        safe_model = model.replace(":", "_").replace("/", "_")
        self.path = Path(cache_dir) / f"{source_hash}-{safe_model}.json"
        self._entries = self._load()

    def get(self, number: int) -> dict | None:
        """Return {"text": ..., "scene": ...} for a page, or None if not cached."""
        return self._entries.get(str(number))  # JSON keys are always strings

    def put(self, number: int, text: str | None, scene: str | None) -> None:
        self._entries[str(number)] = {"text": text, "scene": scene}
        self._save()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file, then swap it in. If the program dies
        # mid-write, the old cache file is still intact.
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")
        temporary.replace(self.path)
