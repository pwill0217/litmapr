import json
from datetime import UTC, datetime
from pathlib import Path

from litmapr.ingest.base import Page

DEFAULT_OUTPUT_DIR = Path("data") / "maps"

# Bump this when the JSON layout changes, so later phases can tell old files apart.
FORMAT_VERSION = 1


def default_output_path(source: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{source.stem}.json"


def write_pages_json(
    output: Path,
    source: Path,
    source_hash: str,
    model: str | None,
    pages: list[tuple[Page, str]],
    metadata: dict | None = None,
) -> None:
    """Save every page's words to a JSON file. Images are left out on purpose.

    Each page's "source" says where its words came from:
    "text_layer", "model", "cache", or "none" (not read yet).
    """
    document = {
        "format_version": FORMAT_VERSION,
        "source_file": source.name,
        "source_hash": source_hash,
        "source_path": str(source.resolve()),
        "model": model,
        # What the file says about itself (ComicInfo.xml): series, issue, year, month.
        "metadata": metadata or {},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "page_count": len(pages),
        "pages": [
            {
                "number": page.number,
                "text": page.text,
                "scene": page.scene,
                "source": source_label,
            }
            for page, source_label in pages
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
