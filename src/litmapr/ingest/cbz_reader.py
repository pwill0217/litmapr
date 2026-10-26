from cbz import ComicInfo

from litmapr.ingest.base import Page, Reader


class CbzReader(Reader):
    def pages(self):
        # 1. load the comic from self.path
        comic = ComicInfo.from_cbz(self.path)
        # 2. loop over its pages, keeping track of the page number
        for i, page in enumerate(comic.pages):
            # 2a. read the image bytes
            # 2b. yield a Page object with the number, image bytes, and no text or path
            yield Page(number=i + 1, image=page.content, text=None, path=None)

    def metadata(self) -> dict:
        """Read ComicInfo.xml, if the file has one. Empty fields are left out."""
        comic = ComicInfo.from_cbz(self.path)
        found = {
            "series": comic.series or None,
            "issue": _issue_text(comic.number),
            "year": comic.year or None,
            "month": comic.month if comic.month and 1 <= comic.month <= 12 else None,
            "title": comic.title or None,
        }
        return {key: value for key, value in found.items() if value is not None}


def _issue_text(number) -> str | None:
    """5 -> "5", 5.0 -> "5", 0.1 -> "0.1", None -> None."""
    if number is None or number == "":
        return None
    if isinstance(number, float) and number.is_integer():
        number = int(number)
    return str(number)
