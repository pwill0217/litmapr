import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from litmapr.cli import app
from litmapr.contents import (
    ContentsError,
    bookmarks,
    contents_path_for,
    parse_contents,
    suggest_contents,
    with_last_pages,
)
from litmapr.model import Work
from litmapr.store import SCHEMA_VERSION, StoryStore

runner = CliRunner()

GOOD = """
# first page, series, issue, year, month
1, FF, 1, 2011, 3
4, FF, 2, 2011, 4
"""


# --- reading contents files --------------------------------------------------------


def test_parse_contents_skips_comments_and_blank_lines():
    sections = parse_contents(GOOD)
    assert [(s.first_page, s.series, s.issue, s.year, s.month) for s in sections] == [
        (1, "FF", "1", 2011, 3),
        (4, "FF", "2", 2011, 4),
    ]


def test_year_and_month_are_optional():
    [section] = parse_contents("10, Fantastic Four, 570")
    assert (section.year, section.month) == (None, None)


def test_series_with_comma_needs_quotes():
    [section] = parse_contents('1, "Fantastic Four, Vol. 3", 570')
    assert section.series == "Fantastic Four, Vol. 3"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("5, FF, 1\n3, FF, 2", "must go up"),
        ("1, FF, 1, 2011, 13", "month"),
        ("1, FF", "at least"),
        ("one, FF, 1", "numbers"),
        ("# only comments", "no issues"),
    ],
)
def test_contents_mistakes_are_explained(text, problem):
    with pytest.raises(ContentsError, match=problem):
        parse_contents(text)


def test_last_pages_run_to_the_next_issue():
    sections = with_last_pages(parse_contents(GOOD), total_pages=6)
    assert [(s.first_page, s.last_page) for s in sections] == [(1, 3), (4, 6)]


def test_issue_past_the_end_of_the_file():
    with pytest.raises(ContentsError, match="only has 3 pages"):
        with_last_pages(parse_contents(GOOD), total_pages=3)


def test_local_page():
    assert Work(series="FF", issue="2", first_page=4).local_page(6) == 3
    assert Work(series="FF", issue="2").local_page(6) == 6


# --- bookmarks --------------------------------------------------------------------------


def test_pdf_bookmarks(omnibus_pdf):
    assert bookmarks(omnibus_pdf) == [(1, "Introduction"), (1, "FF #1"), (4, "FF #2")]


def test_cbz_bookmarks(omnibus_cbz):
    assert bookmarks(omnibus_cbz) == [
        (1, "Fantastic Four #570"),
        (3, "Fantastic Four #571"),
    ]


def test_suggestion_parses_and_comments_out_non_issues(omnibus_pdf):
    draft = suggest_contents(omnibus_pdf)
    assert "# 1, Introduction" in draft
    sections = parse_contents(draft)
    assert [(s.first_page, s.series, s.issue) for s in sections] == [
        (1, "FF", "1"),
        (4, "FF", "2"),
    ]


def test_contents_command_writes_draft_beside_file(omnibus_cbz):
    result = runner.invoke(app, ["contents", str(omnibus_cbz)])
    assert result.exit_code == 0, result.stdout
    target = contents_path_for(omnibus_cbz)
    assert target.name == "FF Omnibus.contents.txt"
    assert "3, Fantastic Four, 571" in target.read_text(encoding="utf-8")


def test_contents_command_wont_overwrite_your_edits(omnibus_cbz):
    runner.invoke(app, ["contents", str(omnibus_cbz)])
    result = runner.invoke(app, ["contents", str(omnibus_cbz)])
    assert result.exit_code == 1
    assert "--force" in result.stdout


# --- extracting an omnibus ---------------------------------------------------------------


def omnibus_pages_json(tmp_path):
    document = {
        "source_file": "Hickman Omnibus.pdf",
        "source_hash": "omni1",
        "pages": [
            {"number": n, "text": f"page {n}", "scene": None} for n in range(1, 7)
        ],
    }
    path = tmp_path / "omni.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def mention_everyone_answer():
    return {
        "entities": [{"name": "Reed Richards", "kind": "character"}],
        "mentions": [
            {"entity": "Reed Richards", "page": n, "summary": f"On page {n}."}
            for n in range(1, 7)
        ],
    }


@pytest.fixture
def every_page_llm(monkeypatch):
    from conftest import FakeLLM

    fake = FakeLLM(mention_everyone_answer())
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: fake)
    return fake


def test_omnibus_becomes_one_work_per_issue(tmp_path, every_page_llm):
    contents = tmp_path / "omni.contents.txt"
    contents.write_text(GOOD, encoding="utf-8")
    result = runner.invoke(
        app, ["extract", str(omnibus_pages_json(tmp_path)), "--contents", str(contents)]
    )
    assert result.exit_code == 0, result.stdout
    assert "2 issues from omni.contents.txt" in result.stdout

    with StoryStore("data/litmapr.db") as store:
        story = store.load()
    works = story.reading_order()
    assert [(w.issue, w.first_page, w.last_page, w.month) for w in works] == [
        ("1", 1, 3, 3),
        ("2", 4, 6, 4),
    ]
    # Each issue only got the mentions from its own pages.
    by_work = {
        w.issue: sorted(m.page for m in story.mentions if m.work_id == w.id)
        for w in works
    }
    assert by_work == {"1": [1, 2, 3], "2": [4, 5, 6]}


def test_show_uses_page_numbers_within_the_issue(tmp_path, every_page_llm):
    contents = tmp_path / "omni.contents.txt"
    contents.write_text(GOOD, encoding="utf-8")
    runner.invoke(
        app, ["extract", str(omnibus_pages_json(tmp_path)), "--contents", str(contents)]
    )
    result = runner.invoke(app, ["show", "Reed Richards"])
    assert "FF #2 p1: On page 4." in result.stdout  # omnibus page 4 = FF #2, page 1


def test_changing_contents_replaces_old_issues(tmp_path, every_page_llm):
    json_path = omnibus_pages_json(tmp_path)
    runner.invoke(app, ["extract", str(json_path)])  # first: treated as one big work
    contents = tmp_path / "omni.contents.txt"
    contents.write_text(GOOD, encoding="utf-8")
    runner.invoke(app, ["extract", str(json_path), "--contents", str(contents)])
    with StoryStore("data/litmapr.db") as store:
        story = store.load()
    assert sorted(w.issue for w in story.works) == [
        "1",
        "2",
    ]  # the whole-file work is gone
    assert len(story.mentions) == 6  # no duplicates


def test_bad_contents_file_is_reported(tmp_path, every_page_llm):
    contents = tmp_path / "omni.contents.txt"
    contents.write_text("1, FF, 1\n99, FF, 2\n", encoding="utf-8")
    result = runner.invoke(
        app, ["extract", str(omnibus_pages_json(tmp_path)), "--contents", str(contents)]
    )
    assert result.exit_code == 1
    assert "only has 6 pages" in result.stdout


def test_add_finds_contents_file_beside_the_omnibus(
    omnibus_pdf, fake_extractor, every_page_llm
):
    contents_path_for(omnibus_pdf).write_text(GOOD, encoding="utf-8")
    result = runner.invoke(app, ["add", str(omnibus_pdf)])
    assert result.exit_code == 0, result.stdout
    assert "omnibus: using Hickman Omnibus.contents.txt" in result.stdout
    with StoryStore("data/litmapr.db") as store:
        assert len(store.load().works) == 2


# --- upgrading an old database ------------------------------------------------------------


def test_old_database_gets_new_columns(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE works (id TEXT PRIMARY KEY, series TEXT NOT NULL, issue TEXT NOT NULL, "
        "issue_number REAL NOT NULL, title TEXT, year INTEGER, month INTEGER, "
        "source_hash TEXT, order_override REAL)"
    )
    db.execute(
        "INSERT INTO works VALUES ('w1', 'FF', '1', 1.0, NULL, 2011, 3, 'h', NULL)"
    )
    db.commit()
    db.close()

    with StoryStore(path) as store:
        [work] = store.load().works
        version = store.db.execute("PRAGMA user_version").fetchone()[0]
    assert work.first_page is None
    assert version == SCHEMA_VERSION


# --- skip lines ----------------------------------------------------------------------------

WITH_SKIP = """
6, Fantastic Four, 570, 2009, 10
29, Fantastic Four, 571, 2009, 11
40, skip
"""


def test_skip_line_parses():
    from litmapr.contents import issues

    sections = parse_contents(WITH_SKIP)
    assert [s.skip for s in sections] == [False, False, True]
    assert [s.issue for s in issues(sections)] == ["570", "571"]


def test_skip_ends_the_issue_before_it():
    from litmapr.contents import issues

    sections = issues(with_last_pages(parse_contents(WITH_SKIP), total_pages=50))
    assert [(s.first_page, s.last_page) for s in sections] == [(6, 28), (29, 39)]


def test_page_filter_leaves_out_front_matter_and_skipped_pages():
    from litmapr.contents import page_filter

    wanted = page_filter(parse_contents(WITH_SKIP))
    assert [p for p in range(1, 51) if wanted(p)] == list(range(6, 40))


def test_only_skip_lines_is_an_error():
    with pytest.raises(ContentsError, match="no issues"):
        parse_contents("5, skip")


def test_ingest_with_contents_skips_pages(omnibus_pdf, fake_extractor, tmp_path):
    contents = tmp_path / "c.txt"
    contents.write_text("2, FF, 1\n5, skip\n", encoding="utf-8")
    result = runner.invoke(
        app, ["ingest", str(omnibus_pdf), "--ocr", "--contents", str(contents)]
    )
    assert result.exit_code == 0, result.stdout
    assert "3 page(s) outside the listed issues were not read" in result.stdout
    saved = json.loads(
        Path("data/maps/Hickman Omnibus.json").read_text(encoding="utf-8")
    )
    assert [p["source"] for p in saved["pages"]] == [
        "skipped",
        "text_layer",
        "text_layer",
        "text_layer",
        "skipped",
        "skipped",
    ]
