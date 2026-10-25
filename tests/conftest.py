import io

import pymupdf
import pytest
from cbz import ComicInfo, PageInfo
from PIL import Image


def tiny_png(color: str) -> bytes:
    """Make a 10x10 image in memory and return its PNG bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_cbz(tmp_path):
    """Build a 3-page CBZ in a temporary folder and return its path."""
    pages = [PageInfo.loads(tiny_png(color)) for color in ("red", "green", "blue")]
    comic = ComicInfo.from_pages(pages=pages)
    path = tmp_path / "sample.cbz"
    comic.save(path)
    return path


@pytest.fixture
def sample_pdf(tmp_path):
    """Build a 2-page PDF: page 1 has text, page 2 is blank (like a scan)."""
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Reed Richards builds the Bridge.")
    document.new_page()
    path = tmp_path / "sample.pdf"
    document.save(path)
    document.close()
    return path
