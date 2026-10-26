import io

import pytest
from cbz import ComicInfo, PageInfo
from PIL import Image
from typer.testing import CliRunner

from litmapr.cli import app
from litmapr.model import EntityKind, Mention, Story, Work
from litmapr.portraits import (
    CharacterLocator,
    best_pages,
    crop,
    inherit_portrait,
    make_portrait,
    portrait_path,
)
from litmapr.store import StoryStore

runner = CliRunner()


def big_png(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 600), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeLocator(CharacterLocator):
    """Finds the character only on pages of one colour, always in the same box."""

    def __init__(self, on_color=(0, 0, 255), box=(100, 100, 300, 400), right=True):
        self.on_color = on_color
        self.box = list(box)
        self.right = right  # does the crop show the right one?
        self.asked = 0
        self.kinds = []

    def locate(self, image, description, kind=EntityKind.CHARACTER):
        self.asked += 1
        self.kinds.append(kind)
        pixel = Image.open(io.BytesIO(image)).convert("RGB").getpixel((5, 5))
        return self.box if pixel == self.on_color else None

    def confirm(self, picture, description, kind):
        return self.right


def keep_background(picture):
    return picture, False


@pytest.fixture
def comic(tmp_path):
    """Three pages: red, green, blue. The character is only 'visible' on blue."""
    pages = [PageInfo.loads(big_png(c)) for c in ("red", "green", "blue")]
    path = tmp_path / "issue.cbz"
    ComicInfo.from_pages(pages=pages).save(path)
    return path


@pytest.fixture
def story(comic):
    s = Story()
    work = Work(series="FF", issue="1", year=2011, month=1, source_path=str(comic))
    s.works.append(work)
    reed = s.add_entity("Reed Richards", EntityKind.CHARACTER)
    for page in (1, 2, 3):
        s.mentions.append(
            Mention(entity_id=reed.id, work_id=work.id, page=page, summary="x")
        )
    return s


def test_crop_adds_padding_and_shrinks():
    picture = crop(big_png("blue"), [100, 100, 300, 400])
    assert picture is not None
    assert max(picture.size) <= 512
    assert picture.size[0] > 200  # padding made it wider than the box


def test_crop_rejects_tiny_or_offpage_boxes():
    assert crop(big_png("blue"), [10, 10, 15, 15]) is None
    assert crop(big_png("blue"), [5000, 5000, 6000, 6000]) is None


def test_portrait_from_the_page_where_they_are_found(story, tmp_path):
    reed = story.entities[0]
    locator = FakeLocator()
    saved, note = make_portrait(
        story, reed, locator, folder=tmp_path / "p", background_remover=keep_background
    )
    assert saved == portrait_path(reed.id, tmp_path / "p")
    assert saved.exists()
    assert locator.asked == 3  # red and green pages had no match
    assert note.startswith("FF #1 p3")
    assert "background kept" in note


def test_gives_up_after_three_pages(story, tmp_path):
    reed = story.entities[0]
    locator = FakeLocator(on_color=(1, 2, 3))  # never matches
    saved, note = make_portrait(story, reed, locator, folder=tmp_path / "p", tries=2)
    assert saved is None
    assert locator.asked == 2
    assert "not found on 2 page(s)" in note


def test_missing_comic_file_is_explained(story, tmp_path):
    story.works[0].source_path = str(tmp_path / "moved.cbz")
    saved, note = make_portrait(
        story, story.entities[0], FakeLocator(), folder=tmp_path / "p"
    )
    assert saved is None
    assert "moved or deleted" in note


def test_portraits_command(story, monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    monkeypatch.setattr("litmapr.cli.make_locator", lambda model: FakeLocator())
    monkeypatch.setattr("litmapr.portraits.remove_background", keep_background)
    result = runner.invoke(app, ["portraits"])
    assert result.exit_code == 0, result.stdout
    assert "saved: Reed Richards" in result.stdout
    assert portrait_path(story.entities[0].id).exists()

    again = runner.invoke(app, ["portraits"])
    assert "already has a picture" in again.stdout


def test_portrait_from_your_own_picture(story, tmp_path, monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    monkeypatch.setattr("litmapr.portraits.remove_background", keep_background)
    picture = tmp_path / "reed.png"
    picture.write_bytes(big_png("purple"))
    result = runner.invoke(
        app, ["portraits", "--name", "Reed Richards", "--image", str(picture)]
    )
    assert result.exit_code == 0, result.stdout
    assert portrait_path(story.entities[0].id).exists()


def test_export_leaves_out_your_file_paths(story, tmp_path):
    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    out = tmp_path / "shared.json"
    runner.invoke(app, ["export", str(out)])
    assert "issue.cbz" not in out.read_text(encoding="utf-8")


def test_source_path_is_saved_and_loaded(story, tmp_path):
    with StoryStore(tmp_path / "t.db") as store:
        store.save(story)
        assert store.load().works[0].source_path.endswith("issue.cbz")


def test_portrait_from_a_page_you_pick(story, monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    monkeypatch.setattr("litmapr.cli.make_locator", lambda model: FakeLocator())
    monkeypatch.setattr("litmapr.portraits.remove_background", keep_background)
    wrong = runner.invoke(app, ["portraits", "--name", "Reed Richards", "--page", "1"])
    assert wrong.exit_code == 1
    assert "not found on FF #1 p1" in wrong.stdout
    right = runner.invoke(app, ["portraits", "--name", "Reed Richards", "--page", "3"])
    assert right.exit_code == 0, right.stdout
    assert "saved: Reed Richards: FF #1 p3" in right.stdout
    assert portrait_path(story.entities[0].id).exists()


def test_page_needs_an_issue_when_there_are_several(story, monkeypatch):
    story.works.append(Work(series="FF", issue="2", year=2011, month=2))
    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    result = runner.invoke(app, ["portraits", "--name", "Reed Richards", "--page", "3"])
    assert result.exit_code == 1
    assert "Add --issue" in result.stdout


def test_a_crop_of_someone_else_is_refused(story, tmp_path):
    reed = story.entities[0]
    saved, note = make_portrait(
        story, reed, FakeLocator(right=False), folder=tmp_path / "p"
    )
    assert saved is None
    assert "showed someone or something else" in note


def test_a_crowd_is_not_a_character_portrait(story, tmp_path):
    reed = story.entities[0]
    whole_page = FakeLocator(box=(0, 0, 400, 600))
    saved, _ = make_portrait(story, reed, whole_page, folder=tmp_path / "p")
    assert saved is None


def test_places_get_a_panel_with_its_background(story, tmp_path):
    work = story.works[0]
    harbor = story.add_entity("Star Harbor", EntityKind.LOCATION)
    story.mentions.append(
        Mention(entity_id=harbor.id, work_id=work.id, page=3, summary="x")
    )
    removed = []
    locator = FakeLocator(box=(0, 0, 400, 600))  # a whole-page panel is fine here

    def remover(picture):
        removed.append(True)
        return picture, True

    saved, note = make_portrait(
        story, harbor, locator, folder=tmp_path / "p", background_remover=remover
    )
    assert saved is not None
    assert note == "FF #1 p3"
    assert locator.kinds == [EntityKind.LOCATION]
    assert not removed


def test_ideas_get_no_picture(story, tmp_path):
    idea = story.add_entity("Solve Everything", EntityKind.CONCEPT)
    saved, note = make_portrait(story, idea, FakeLocator(), folder=tmp_path / "p")
    assert saved is None
    assert "no picture" in note


def test_busiest_page_is_tried_first(story):
    reed, work = story.entities[0], story.works[0]
    story.mentions.append(
        Mention(entity_id=reed.id, work_id=work.id, page=3, summary="y")
    )
    assert [m.page for m in best_pages(story, reed)] == [3, 1, 2]


def test_merged_entity_keeps_a_picture(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    portrait_path("drop", folder).write_bytes(b"png")
    inherit_portrait("keep", "drop", folder)
    assert portrait_path("keep", folder).read_bytes() == b"png"
    assert not portrait_path("drop", folder).exists()

    portrait_path("other", folder).write_bytes(b"second")
    inherit_portrait("keep", "other", folder)  # keep already has one: it stays
    assert portrait_path("keep", folder).read_bytes() == b"png"
    assert not portrait_path("other", folder).exists()


def test_remove_a_wrong_picture(story, monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    reed = story.entities[0]
    path = portrait_path(reed.id)
    path.parent.mkdir(parents=True)
    path.write_bytes(big_png("red"))
    result = runner.invoke(app, ["portraits", "--name", "Reed Richards", "--remove"])
    assert result.exit_code == 0, result.stdout
    assert "removed" in result.stdout
    assert not path.exists()
