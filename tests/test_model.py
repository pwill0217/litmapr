import pytest
from pydantic import ValidationError

from litmapr.model import (
    Callback,
    Entity,
    EntityKind,
    Mention,
    Relation,
    RelationKind,
    Story,
    Work,
    slugify,
)


@pytest.fixture
def story():
    """A tiny hand-made story built from Avengers: Armageddon #5."""
    s = Story()
    issue5 = Work(series="Avengers: Armageddon", issue="5", year=2026, month=5)
    s.works.append(issue5)
    ross = s.add_entity(
        "Thaddeus Ross", EntityKind.CHARACTER, aliases=["Red Hulk", "Thunderbolt Ross"]
    )
    carol = s.add_entity(
        "Carol Danvers", EntityKind.CHARACTER, aliases=["Captain Marvel"]
    )
    reed = s.add_entity(
        "Reed Richards", EntityKind.CHARACTER, aliases=["Mr. Fantastic"]
    )
    s.mentions.append(
        Mention(
            entity_id=ross.id, work_id=issue5.id, page=3, summary="Explodes in Russia."
        )
    )
    s.relations.append(
        Relation(
            source_id=reed.id,
            target_id=carol.id,
            kind=RelationKind.ALLY,
            label="tells Carol her powers are down to ten percent",
            sentiment=0.4,
            work_id=issue5.id,
            page=8,
        )
    )
    return s


# --- names, slugs and aliases -------------------------------------------------


def test_slugify():
    assert slugify("Mr. Fantastic") == "mr-fantastic"


def test_one_entity_many_names(story):
    [ross] = story.find_entities("red hulk")
    assert ross.name == "Thaddeus Ross"
    assert story.find_entities("Thaddeus Ross") == [ross]


def test_name_clash_gets_numbered_slug():
    s = Story()
    steve = s.add_entity("Captain America", EntityKind.CHARACTER)
    colton = s.add_entity("Captain America", EntityKind.CHARACTER)
    assert steve.slug == "captain-america"
    assert colton.slug == "captain-america-2"
    assert steve.id != colton.id  # the real identity is the random id


def test_shared_alias_finds_everyone():
    s = Story()
    s.add_entity("Steve Rogers", EntityKind.CHARACTER, aliases=["Captain America"])
    s.add_entity("David Colton", EntityKind.CHARACTER, aliases=["Captain America"])
    assert len(s.find_entities("captain america")) == 2


def test_duplicate_aliases_removed():
    entity = Entity(
        name="Ross", kind="character", aliases=["Red Hulk", "red hulk ", "Red Hulk"]
    )
    assert entity.aliases == ["Red Hulk"]


# --- validation: what Pydantic refuses ----------------------------------------


def test_unknown_entity_kind_rejected():
    with pytest.raises(ValidationError):
        Entity(name="Bridge", kind="gadget")  # not in EntityKind


def test_kind_from_plain_text_accepted():
    # An LLM sends plain strings; Pydantic turns them into the enum.
    entity = Entity(name="The Bridge", kind="device")
    assert entity.kind is EntityKind.DEVICE


def test_sentiment_out_of_range_rejected(story):
    with pytest.raises(ValidationError):
        Relation(
            source_id="a",
            target_id="b",
            kind="ally",
            label="x",
            sentiment=5.0,
            work_id="w",
            page=1,
        )


def test_relation_with_itself_rejected():
    with pytest.raises(ValidationError, match="itself"):
        Relation(
            source_id="a", target_id="a", kind="ally", label="x", work_id="w", page=1
        )


def test_page_zero_rejected():
    with pytest.raises(ValidationError):
        Mention(entity_id="e", work_id="w", page=0, summary="x")


def test_llm_style_dict_is_validated():
    # This is what Phase 3 will do with model output.
    raw = {"name": "The Bridge", "kind": "device", "aliases": ["the Bridge"]}
    entity = Entity.model_validate(raw)
    assert entity.name == "The Bridge"


# --- reading order -------------------------------------------------------------


def test_reading_order_crosses_series_by_date():
    s = Story()
    s.works = [
        Work(series="Secret Wars", issue="1", year=2015, month=5),
        Work(series="New Avengers", issue="1", year=2013, month=2),
        Work(series="Avengers", issue="1", year=2012, month=12),
        Work(series="Avengers", issue="2", year=2013, month=1),
    ]
    order = [(w.series, w.issue) for w in s.reading_order()]
    assert order == [
        ("Avengers", "1"),
        ("Avengers", "2"),
        ("New Avengers", "1"),
        ("Secret Wars", "1"),
    ]


def test_same_month_falls_back_to_series_then_issue():
    s = Story()
    s.works = [
        Work(series="New Avengers", issue="3", year=2013, month=3),
        Work(series="Avengers", issue="7", year=2013, month=3),
        Work(series="Avengers", issue="6", year=2013, month=3),
    ]
    assert [w.issue for w in s.reading_order()] == ["6", "7", "3"]


def test_override_moves_an_issue():
    s = Story()
    a = Work(series="Avengers", issue="1", year=2013, month=1)
    b = Work(
        series="New Avengers", issue="1", year=2013, month=2, order_override=201300.5
    )
    s.works = [a, b]
    assert s.reading_order() == [b, a]  # b now reads before a


def test_works_without_dates_go_last():
    s = Story()
    dated = Work(series="Avengers", issue="1", year=2013, month=1)
    undated = Work(series="Avengers", issue="Annual 1")
    s.works = [undated, dated]
    assert s.reading_order() == [dated, undated]


def test_issue_number_parsing():
    assert Work(series="X", issue="0.1").issue_number == 0.1
    assert Work(series="X", issue="Annual 2").issue_number == 2.0


# --- references and saving -----------------------------------------------------


def test_clean_story_has_no_broken_references(story):
    assert story.broken_references() == []


def test_broken_reference_is_reported(story):
    story.callbacks.append(
        Callback(
            from_mention_id=story.mentions[0].id, to_mention_id="missing", reason="x"
        )
    )
    problems = story.broken_references()
    assert len(problems) == 1
    assert "missing" in problems[0]


def test_save_and_load_round_trip(story, tmp_path):
    path = tmp_path / "story.json"
    story.save(path)
    assert Story.load(path) == story


# --- quotes and first appearances ------------------------------------------------


def test_mention_quote_is_optional():
    visual_only = Mention(
        entity_id="e", work_id="w", page=8, summary="Carol walks out."
    )
    assert visual_only.quote is None


def test_first_appearance_uses_reading_order_not_insert_order():
    s = Story()
    later = Work(series="Avengers", issue="2", year=2013, month=1)
    earlier = Work(series="Avengers", issue="1", year=2012, month=12)
    s.works = [later, earlier]
    carol = s.add_entity("Carol Danvers", EntityKind.CHARACTER)
    s.mentions = [
        Mention(entity_id=carol.id, work_id=later.id, page=1, summary="second"),
        Mention(entity_id=carol.id, work_id=earlier.id, page=9, summary="first"),
    ]
    assert s.first_appearance(carol.id).summary == "first"


def test_first_appearance_of_unmentioned_entity_is_none():
    s = Story()
    ghost = s.add_entity("Nobody", EntityKind.CHARACTER)
    assert s.first_appearance(ghost.id) is None
