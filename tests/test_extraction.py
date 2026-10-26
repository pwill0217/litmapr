import json

import pytest
from conftest import ARMAGEDDON_ANSWER, FakeLLM

from litmapr.cache import ResponseCache
from litmapr.extraction import (
    ChunkExtraction,
    build_prompt,
    chunk_pages,
    extract_work,
    guess_series_and_issue,
    merge_into_story,
    parse_extraction,
)
from litmapr.model import EntityKind, Story, Work


@pytest.fixture
def work():
    return Work(series="Avengers Armageddon", issue="5", source_hash="abc123")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Avengers_Armageddon005.cbz", ("Avengers Armageddon", "5")),
        ("Fantastic Four #12.cbz", ("Fantastic Four", "12")),
        ("Secret_Wars_009.pdf", ("Secret Wars", "9")),
        ("New Avengers 0.1.cbz", ("New Avengers", "0.1")),
        ("Watchmen.pdf", ("Watchmen", "1")),
    ],
)
def test_guess_series_and_issue(filename, expected):
    assert guess_series_and_issue(filename) == expected


def test_chunks_skip_pages_without_words():
    pages = [
        {"number": n, "text": "x" if n != 3 else None, "scene": None}
        for n in range(1, 7)
    ]
    chunks = chunk_pages(pages, size=2)
    assert [[p["number"] for p in c] for c in chunks] == [[1, 2], [4, 5], [6]]


def test_prompt_lists_known_entities_and_pages(work):
    story = Story()
    story.add_entity("Thaddeus Ross", EntityKind.CHARACTER, aliases=["Red Hulk"])
    prompt = build_prompt(work, [{"number": 7, "text": "SMASH", "scene": None}], story)
    assert "Thaddeus Ross (character), also called: Red Hulk" in prompt
    assert "--- PAGE 7 ---" in prompt
    assert "SMASH" in prompt


# --- parsing the model's answer -------------------------------------------------------


def test_parse_good_answer():
    extraction, problems = parse_extraction(json.dumps(ARMAGEDDON_ANSWER))
    assert problems == []
    assert len(extraction.entities) == 3
    assert extraction.entities[1].kind is EntityKind.LOCATION


def test_parse_keeps_good_items_when_one_is_bad():
    answer = {
        "relations": [
            {"source": "A", "target": "B", "kind": "ally", "label": "x", "page": 1},
            {
                "source": "A",
                "target": "B",
                "kind": "frenemies",
                "label": "x",
                "page": 1,
            },
        ]
    }
    extraction, problems = parse_extraction(json.dumps(answer))
    assert len(extraction.relations) == 1
    assert len(problems) == 1
    assert "relations" in problems[0]


def test_parse_not_json():
    extraction, problems = parse_extraction("Sure! Here are the entities:")
    assert extraction == ChunkExtraction()
    assert "not valid JSON" in problems[0]


# --- merging into the story -----------------------------------------------------------


def test_merge_creates_entities_and_resolves_aliases(work):
    story = Story(works=[work])
    extraction, _ = parse_extraction(json.dumps(ARMAGEDDON_ANSWER))
    report = merge_into_story(story, work, extraction, pages={1, 2})
    assert report.new_entities == 3
    assert report.mentions == 3  # "Red Hulk" found Thaddeus Ross through his alias
    assert report.relations == 1
    [ross] = story.find_entities("Red Hulk")
    assert story.first_appearance(ross.id).summary == "Takes over Latveria."


def test_merge_reuses_known_entity_and_learns_new_alias(work):
    story = Story(works=[work])
    ross = story.add_entity("Thaddeus Ross", EntityKind.CHARACTER)
    answer = {
        "entities": [
            {"name": "Thaddeus Ross", "kind": "character", "aliases": ["Red Hulk"]}
        ]
    }
    extraction, _ = parse_extraction(json.dumps(answer))
    report = merge_into_story(story, work, extraction, pages={1})
    assert report.new_entities == 0
    assert len(story.entities) == 1
    assert "Red Hulk" in ross.aliases


def test_new_entry_never_matched_or_renamed_through_an_alias(work):
    story = Story(works=[work])
    story.add_entity("Thaddeus Ross", EntityKind.CHARACTER)
    answer = {
        "entities": [
            {"name": "Red Hulk", "kind": "character", "aliases": ["Thaddeus Ross"]}
        ]
    }
    extraction, _ = parse_extraction(json.dumps(answer))
    merge_into_story(story, work, extraction, pages={1})
    [red] = [e for e in story.entities if e.name == "Red Hulk"]
    assert len(story.entities) == 2  # left for `litmapr merge` to confirm
    assert red.aliases == []  # can't take another entity's name as an alias


def test_undescribed_name_becomes_a_character_and_wrong_page_is_skipped(work):
    story = Story(works=[work])
    answer = {
        "entities": [{"name": "Carol Danvers", "kind": "character"}],
        "mentions": [
            {"entity": "Galactus", "page": 1, "summary": "x"},
            {"entity": "Carol Danvers", "page": 99, "summary": "x"},
        ],
    }
    extraction, _ = parse_extraction(json.dumps(answer))
    report = merge_into_story(story, work, extraction, pages={1, 2})
    assert report.mentions == 1
    [galactus] = story.find_entities("Galactus")
    assert galactus.kind is EntityKind.CHARACTER
    assert any("page 99" in p for p in report.problems)


def test_merge_rejects_relation_with_itself(work):
    story = Story(works=[work])
    answer = {
        "entities": [
            {"name": "Thaddeus Ross", "kind": "character", "aliases": ["Red Hulk"]}
        ],
        "relations": [
            {
                "source": "Red Hulk",
                "target": "Thaddeus Ross",
                "kind": "rival",
                "label": "x",
                "page": 1,
            }
        ],
    }
    extraction, _ = parse_extraction(json.dumps(answer))
    report = merge_into_story(story, work, extraction, pages={1})
    assert report.relations == 0
    assert "itself" in report.problems[0]


# --- running a whole work -------------------------------------------------------------


def test_extract_work_caches_answers(work, tmp_path):
    pages = [
        {"number": 1, "text": "a", "scene": None},
        {"number": 2, "text": "b", "scene": None},
    ]
    cache = ResponseCache("test", cache_dir=tmp_path)

    first = FakeLLM(ARMAGEDDON_ANSWER)
    list(
        extract_work(Story(works=[work]), work, pages, first, cache, pages_per_chunk=2)
    )
    assert len(first.prompts) == 1

    second = FakeLLM(ARMAGEDDON_ANSWER)
    results = list(
        extract_work(Story(works=[work]), work, pages, second, cache, pages_per_chunk=2)
    )
    assert second.prompts == []  # answered from the cache
    assert results[0][2] is True


def test_second_chunk_prompt_knows_first_chunk_entities(work, tmp_path):
    pages = [{"number": n, "text": "x", "scene": None} for n in (1, 2)]
    fake = FakeLLM(ARMAGEDDON_ANSWER)
    story = Story(works=[work])
    list(
        extract_work(
            story, work, pages, fake, ResponseCache("t", tmp_path), pages_per_chunk=1
        )
    )
    assert "(none yet)" in fake.prompts[0]
    assert "Thaddeus Ross (character)" in fake.prompts[1]


# --- fixes from the first real run -------------------------------------------------------


class FlakyLLM(FakeLLM):
    """Answers with cut-off JSON first, then a good answer."""

    def __init__(self, answers):
        super().__init__("{}")
        self.answers = list(answers)

    def complete_json(self, prompt, schema):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_cut_off_answer_is_retried_and_not_cached(work, tmp_path):
    pages = [{"number": 1, "text": "x", "scene": None}]
    cache = ResponseCache("t", tmp_path)
    flaky = FlakyLLM(['{"entities": [{"name": "Ro', json.dumps(ARMAGEDDON_ANSWER)])
    story = Story(works=[work])
    [(_, report, _)] = list(
        extract_work(story, work, pages, flaky, cache, pages_per_chunk=4)
    )
    assert len(flaky.prompts) == 2  # asked twice
    assert report.new_entities == 3  # the second, good answer was used
    assert json.loads(
        cache.get(f"v1:{work.source_hash}:1")
    )  # and that's what got cached


def test_broken_answer_in_old_cache_is_asked_again(work, tmp_path):
    pages = [{"number": 1, "text": "x", "scene": None}]
    cache = ResponseCache("t", tmp_path)
    cache.put(f"v1:{work.source_hash}:1", '{"entities": [{"name": "Ro')
    fake = FakeLLM(ARMAGEDDON_ANSWER)
    list(extract_work(Story(works=[work]), work, pages, fake, cache, pages_per_chunk=4))
    assert len(fake.prompts) == 1


def test_short_name_resolves_when_unique(work):
    story = Story(works=[work])
    ross = story.add_entity("Thaddeus Ross", EntityKind.CHARACTER, aliases=["Red Hulk"])
    answer = {"mentions": [{"entity": "Ross", "page": 1, "summary": "Explodes."}]}
    extraction, _ = parse_extraction(json.dumps(answer))
    report = merge_into_story(story, work, extraction, pages={1})
    assert report.mentions == 1
    assert story.mentions[0].entity_id == ross.id
    assert "Ross" not in ross.aliases  # not learned as a permanent alias


def test_short_name_skipped_when_ambiguous(work):
    story = Story(works=[work])
    story.add_entity("Thaddeus Ross", EntityKind.CHARACTER)
    story.add_entity("Betty Ross", EntityKind.CHARACTER)
    answer = {"mentions": [{"entity": "Ross", "page": 1, "summary": "?"}]}
    extraction, _ = parse_extraction(json.dumps(answer))
    report = merge_into_story(story, work, extraction, pages={1})
    assert report.mentions == 0
    assert "Ross" in report.problems[0]


def test_short_name_must_be_whole_words(work):
    story = Story(works=[work])
    stone = story.add_entity("Rossetta Stone", EntityKind.DEVICE)
    answer = {"mentions": [{"entity": "Ross", "page": 1, "summary": "?"}]}
    extraction, _ = parse_extraction(json.dumps(answer))
    merge_into_story(story, work, extraction, pages={1})
    assert all(m.entity_id != stone.id for m in story.mentions)


def test_shared_alias_does_not_glue_characters_together(work):
    # The real bug: the Wizard (Bentley) was also "Ben", so Ben Grimm got merged into him.
    story = Story(works=[work])
    answer = {
        "entities": [
            {
                "name": "Bentley Wittman",
                "kind": "character",
                "aliases": ["The Wizard", "Ben"],
            },
            {"name": "Ben Grimm", "kind": "character", "aliases": ["Ben", "Thing"]},
        ],
        "mentions": [
            {"entity": "Thing", "page": 1, "summary": "Smashes a robot."},
            {"entity": "Ben", "page": 1, "summary": "Which Ben?"},
        ],
    }
    extraction, _ = parse_extraction(json.dumps(answer))
    report = merge_into_story(story, work, extraction, pages={1})
    wizard = story.find_entities("The Wizard")[0]
    grimm = story.find_entities("Ben Grimm")[0]
    assert wizard.id != grimm.id
    assert "Thing" not in wizard.aliases
    assert [m.entity_id for m in story.mentions] == [grimm.id]  # "Ben" was ambiguous
    assert any("unclear who" in p for p in report.problems)


def test_kind_label_is_stripped_from_names(work):
    story = Story(works=[work])
    spidey = story.add_entity("Spider-Man", EntityKind.CHARACTER)
    answer = {
        "mentions": [
            {"entity": "Spider-Man (character)", "page": 1, "summary": "Thwip."}
        ]
    }
    extraction, _ = parse_extraction(json.dumps(answer))
    merge_into_story(story, work, extraction, pages={1})
    assert len(story.entities) == 1
    assert story.mentions[0].entity_id == spidey.id


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Mr. Fantastic", True),
        ("Victor von Doom", True),
        ("Thirty-Two", True),
        ("character in a black suit", False),
        ("barrier", False),
        ("Unknown child", False),
    ],
)
def test_looks_like_a_name(name, expected):
    from litmapr.extraction import looks_like_a_name

    assert looks_like_a_name(name) is expected
