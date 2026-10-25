import sqlite3

import pytest

from litmapr.model import EntityKind, Mention, Story, Work
from litmapr.store import StoryStore


@pytest.fixture
def store(tmp_path):
    with StoryStore(tmp_path / "test.db") as s:
        yield s  # the test runs here; the store closes afterwards


@pytest.fixture
def two_issue_story():
    """Red Hulk shows up in #4 and #5; the works are added out of order on purpose."""
    s = Story()
    issue5 = Work(series="Avengers: Armageddon", issue="5", year=2026, month=5)
    issue4 = Work(series="Avengers: Armageddon", issue="4", year=2026, month=4)
    s.works = [issue5, issue4]
    ross = s.add_entity("Thaddeus Ross", EntityKind.CHARACTER, aliases=["Red Hulk"])
    s.mentions = [
        Mention(entity_id=ross.id, work_id=issue5.id, page=3, summary="Explodes."),
        Mention(
            entity_id=ross.id,
            work_id=issue4.id,
            page=12,
            summary="Declares New America.",
            quote="Latveria is NEW AMERICA now.",
        ),
        Mention(
            entity_id=ross.id, work_id=issue4.id, page=2, summary="Takes Latveria."
        ),
    ]
    return s


def test_save_and_load_round_trip(store, two_issue_story):
    store.save(two_issue_story)
    loaded = store.load()
    assert {w.id for w in loaded.works} == {w.id for w in two_issue_story.works}
    assert loaded.entities == two_issue_story.entities
    assert sorted(loaded.mentions, key=lambda m: m.id) == sorted(
        two_issue_story.mentions, key=lambda m: m.id
    )


def test_quote_is_kept(store, two_issue_story):
    store.save(two_issue_story)
    quotes = [m.quote for m in store.load().mentions if m.quote]
    assert quotes == ["Latveria is NEW AMERICA now."]


def test_saving_twice_does_not_duplicate(store, two_issue_story):
    store.save(two_issue_story)
    store.save(two_issue_story)
    assert len(store.load().mentions) == 3


def test_saving_an_update_changes_the_row(store, two_issue_story):
    store.save(two_issue_story)
    two_issue_story.entities[0].description = "Former general, now Red Hulk."
    store.save(two_issue_story)
    [ross] = store.load().entities
    assert ross.description == "Former general, now Red Hulk."
    assert len(store.load().mentions) == 3  # update didn't wipe mentions


def test_find_by_alias_ignores_case(store, two_issue_story):
    store.save(two_issue_story)
    [ross] = store.find_entities("RED HULK")
    assert ross.name == "Thaddeus Ross"


def test_mentions_come_back_in_reading_order(store, two_issue_story):
    store.save(two_issue_story)
    ross = two_issue_story.entities[0]
    summaries = [m.summary for m in store.mentions_in_order(ross.id)]
    assert summaries == ["Takes Latveria.", "Declares New America.", "Explodes."]


def test_first_appearance_store_and_model_agree(store, two_issue_story):
    store.save(two_issue_story)
    ross = two_issue_story.entities[0]
    from_db = store.first_appearance(ross.id)
    from_model = two_issue_story.first_appearance(ross.id)
    assert from_db == from_model
    assert from_db.summary == "Takes Latveria."  # issue #4, page 2


def test_first_appearance_moves_back_when_earlier_issue_is_added(
    store, two_issue_story
):
    ross = two_issue_story.entities[0]
    issue1 = Work(series="Avengers: Armageddon", issue="1", year=2026, month=1)
    two_issue_story.works.append(issue1)
    two_issue_story.mentions.append(
        Mention(entity_id=ross.id, work_id=issue1.id, page=5, summary="Gives a speech.")
    )
    store.save(two_issue_story)
    assert store.first_appearance(ross.id).summary == "Gives a speech."


def test_database_refuses_mention_of_unknown_entity(store, two_issue_story):
    store.save(two_issue_story)
    bad = Story(
        works=two_issue_story.works,
        mentions=[
            Mention(
                entity_id="nobody",
                work_id=two_issue_story.works[0].id,
                page=1,
                summary="x",
            )
        ],
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.save(bad)


def test_failed_save_changes_nothing(store, two_issue_story):
    # The bad mention comes AFTER a valid new entity in the same save.
    two_issue_story.add_entity("Carol Danvers", EntityKind.CHARACTER)
    two_issue_story.mentions.append(
        Mention(
            entity_id="nobody", work_id=two_issue_story.works[0].id, page=1, summary="x"
        )
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.save(two_issue_story)
    assert store.load().entities == []  # all-or-nothing: Carol wasn't saved either


def test_data_survives_closing_the_file(tmp_path, two_issue_story):
    path = tmp_path / "litmapr.db"
    with StoryStore(path) as store:
        store.save(two_issue_story)
    with StoryStore(path) as reopened:
        assert len(reopened.load().mentions) == 3
