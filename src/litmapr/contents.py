"""Omnibus support: say where each issue starts inside one big file.

A contents file is plain text, one issue per line:

    # first page, series, issue, year, month
    5,   Fantastic Four, 570, 2009, 10
    29,  Fantastic Four, 571, 2009, 11

Lines starting with # are ignored. Year and month are optional.
Each issue runs until the page before the next one starts. Pages before
the first issue are left out, and a line like `118, skip` leaves out
everything from that page until the next issue (a cover gallery, say).
"""

import csv
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

from cbz import ComicInfo

from litmapr.extraction import guess_series_and_issue

HEADER = "# first page, series, issue, year, month   (or: first page, skip)"


class ContentsError(ValueError):
    """A problem in a contents file, with the line it's on."""


@dataclass
class Section:
    first_page: int
    series: str
    issue: str
    year: int | None = None
    month: int | None = None
    last_page: int | None = None  # filled in by with_last_pages()
    skip: bool = False  # a `skip` line: pages to leave out, not an issue


def contents_path_for(comic: Path) -> Path:
    """Where litmapr looks for an omnibus's contents file: right beside it.
    'Hickman Omnibus.cbz' -> 'Hickman Omnibus.contents.txt'."""
    return comic.with_name(f"{comic.stem}.contents.txt")


def parse_contents(text: str) -> list[Section]:
    sections: list[Section] = []
    lines = text.splitlines()
    for line_number, row in enumerate(
        csv.reader(lines, skipinitialspace=True), start=1
    ):
        if not row or not "".join(row).strip() or row[0].lstrip().startswith("#"):
            continue
        where = f"line {line_number}"
        if len(row) >= 2 and row[1].strip().lower() == "skip":
            try:
                first = int(row[0])
            except ValueError:
                raise ContentsError(f"{where}: the page must be a number") from None
            if sections and first <= sections[-1].first_page:
                raise ContentsError(f"{where}: first pages must go up, line by line")
            sections.append(Section(first, "", "", skip=True))
            continue
        if len(row) < 3:
            raise ContentsError(f"{where}: needs at least first page, series and issue")
        try:
            first = int(row[0])
            year = int(row[3]) if len(row) > 3 and row[3].strip() else None
            month = int(row[4]) if len(row) > 4 and row[4].strip() else None
        except ValueError:
            raise ContentsError(
                f"{where}: page, year and month must be numbers"
            ) from None
        if first < 1:
            raise ContentsError(f"{where}: pages start at 1")
        if month is not None and not 1 <= month <= 12:
            raise ContentsError(f"{where}: month must be 1-12")
        if sections and first <= sections[-1].first_page:
            raise ContentsError(f"{where}: first pages must go up, line by line")
        series, issue = row[1].strip(), row[2].strip()
        if not series or not issue:
            raise ContentsError(f"{where}: series and issue can't be empty")
        sections.append(Section(first, series, issue, year, month))
    if not any(not section.skip for section in sections):
        raise ContentsError("no issues listed")
    return sections


def issues(sections: list[Section]) -> list[Section]:
    """The sections that are real issues (not `skip` lines)."""
    return [section for section in sections if not section.skip]


def page_filter(sections: list[Section]):
    """A test for 'is this page part of an issue?', so skipped pages and
    front matter are never sent to the vision model."""
    starts = [(s.first_page, s.skip) for s in sections]

    def wanted(page: int) -> bool:
        inside = [skip for first, skip in starts if first <= page]
        return bool(inside) and not inside[-1]

    return wanted


def with_last_pages(sections: list[Section], total_pages: int) -> list[Section]:
    """Each issue ends where the next begins; the last one ends with the file."""
    if sections[-1].first_page > total_pages:
        raise ContentsError(
            f"an issue starts on page {sections[-1].first_page}, "
            f"but the file only has {total_pages} pages"
        )
    for current, following in itertools.pairwise(sections):
        current.last_page = following.first_page - 1
    sections[-1].last_page = total_pages
    return sections


def read_contents(path: Path, total_pages: int) -> list[Section]:
    try:
        sections = parse_contents(path.read_text(encoding="utf-8"))
        return with_last_pages(sections, total_pages)
    except ContentsError as error:
        raise ContentsError(f"{path.name}: {error}") from None


# --- Suggesting a contents file from bookmarks ---------------------------------------


def bookmarks(path: Path) -> list[tuple[int, str]]:
    """(page, title) pairs from a PDF's table of contents or a CBZ's
    ComicInfo page bookmarks. Empty if the file has none."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(path) as document:
            # get_toc() gives [level, title, page]; level 1 = top-level chapters.
            return [
                (page, title) for level, title, page in document.get_toc() if level == 1
            ]
    if suffix == ".cbz":
        comic = ComicInfo.from_cbz(path)
        return [
            (number, page.bookmark)
            for number, page in enumerate(comic.pages, start=1)
            if page.bookmark
        ]
    return []


def suggest_contents(path: Path) -> str:
    """A draft contents file built from bookmarks, for you to check and fix."""
    marks = bookmarks(path)
    lines = [
        f"# Suggested from the bookmarks in {path.name}. Check every line against the book,",
        "# fix series names and add year and month: they set the reading order.",
        "# Lines starting with # are ignored. Bookmarks with no issue number are commented out.",
        HEADER,
    ]
    if not marks:
        lines.append(
            "# (no bookmarks found: fill this in from the book's table of contents)"
        )
        lines.append("# 1, Series Name, 1, 2012, 1")
    for page, title in sorted(marks):
        series, issue = guess_series_and_issue(f"{title.strip()}.x")
        has_number = re.search(r"\d", title) is not None
        prefix = "" if has_number else "# "
        lines.append(f"{prefix}{page}, {series}, {issue}, , ")
    return "\n".join(lines) + "\n"
