from conftest import FakeEmbedder, FakeLLM, ScriptedLLM
from typer.testing import CliRunner

from litmapr.cli import app
from litmapr.model import EntityKind, Mention, Relation, Story, Work
from litmapr.resolve import find_duplicate_candidates, pick_keeper
from litmapr.store import StoryStore

runner = CliRunner()
KEYWORDS = ["reed", "fantastic", "stretch", "scientist", "carol", "pilot", "latveria"]


def reed_story() -> Story:
    s = Story()
    work = Work(series="FF", issue="1", year=2012, month=1)
    s.works.append(work)
    reed = s.add_entity(
        "Reed Richards",
        EntityKind.CHARACTER,
        description="Stretching scientist, Fantastic Four",
    )
    mister = s.add_entity(
        "Mr. Fantastic",
        EntityKind.CHARACTER,
        description="Fantastic Four leader who can stretch",
    )
    carol = s.add_entity("Carol Danvers", EntityKind.CHARACTER, description="Pilot")
    s.add_entity("Latveria", EntityKind.LOCATION)
    s.mentions += [
        Mention(
            entity_id=reed.id, work_id=work.id, page=1, summary="Builds the Bridge."
        ),
        Mention(entity_id=reed.id, work_id=work.id, page=2, summary="Talks to Carol."),
        Mention(entity_id=mister.id, work_id=work.id, page=3, summary="Stretches."),
    ]
    s.relations.append(
        Relation(
            source_id=mister.id,
            target_id=carol.id,
            kind="ally",
            label="treats her",
            work_id=work.id,
            page=3,
        )
    )
    return s


def test_similar_entities_become_candidates():
    story = reed_story()
    candidates = find_duplicate_candidates(story, FakeEmbedder(KEYWORDS), threshold=0.5)
    pairs = [{c.a.name, c.b.name} for c in candidates]
    assert {"Reed Richards", "Mr. Fantastic"} in pairs
    assert all(
        "Latveria" not in pair for pair in pairs
    )  # different kind, never compared


def test_shared_name_is_always_a_candidate():
    s = Story()
    s.add_entity("Steve Rogers", EntityKind.CHARACTER, aliases=["Captain America"])
    s.add_entity("David Colton", EntityKind.CHARACTER, aliases=["Captain America"])
    [candidate] = find_duplicate_candidates(s, FakeEmbedder(["zzz"]), threshold=0.99)
    assert candidate.score == 1.0  # flagged for a check, not merged automatically


def test_keeper_is_the_more_mentioned():
    story = reed_story()
    reed, mister = story.entities[0], story.entities[1]
    keep, drop = pick_keeper(story, mister, reed)
    assert (keep.name, drop.name) == ("Reed Richards", "Mr. Fantastic")


def test_store_merge_moves_everything(tmp_path):
    story = reed_story()
    reed, mister = story.entities[0], story.entities[1]
    with StoryStore(tmp_path / "t.db") as store:
        store.save(story)
        store.merge_entities(reed.id, mister.id)
        merged = store.load()
    assert len(merged.entities) == 3
    [kept] = [e for e in merged.entities if e.id == reed.id]
    assert "Mr. Fantastic" in kept.aliases
    assert len(merged.mentions_in_order(reed.id)) == 3
    assert merged.relations[0].source_id == reed.id


def test_merge_removes_relations_that_point_at_themselves(tmp_path):
    story = reed_story()
    reed, mister = story.entities[0], story.entities[1]
    story.relations.append(
        Relation(
            source_id=reed.id,
            target_id=mister.id,
            kind="other",
            label="same guy",
            work_id=story.works[0].id,
            page=1,
        )
    )
    with StoryStore(tmp_path / "t.db") as store:
        store.save(story)
        store.merge_entities(reed.id, mister.id)
        assert all(r.source_id != r.target_id for r in store.load().relations)


def test_merge_command_asks_llm_and_merges(monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(reed_story())
    monkeypatch.setattr(
        "litmapr.cli.make_embedder", lambda model: FakeEmbedder(KEYWORDS)
    )
    llm = ScriptedLLM(
        [{"same": True, "reason": "Mr. Fantastic is Reed's hero name."}] * 10
    )
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: llm)

    result = runner.invoke(app, ["merge", "--threshold", "0.5"])
    assert result.exit_code == 0, result.stdout
    assert "merged 'Mr. Fantastic' into 'Reed Richards'" in result.stdout
    with StoryStore("data/litmapr.db") as store:
        assert len(store.find_entities("Mr. Fantastic")) == 1


def test_merge_dry_run_changes_nothing(monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(reed_story())
    monkeypatch.setattr(
        "litmapr.cli.make_embedder", lambda model: FakeEmbedder(KEYWORDS)
    )
    llm = ScriptedLLM([{"same": True, "reason": "yes"}] * 10)
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: llm)
    runner.invoke(app, ["merge", "--threshold", "0.5", "--dry-run"])
    with StoryStore("data/litmapr.db") as store:
        assert len(store.load().entities) == 4


def test_merge_respects_a_no(monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(reed_story())
    monkeypatch.setattr(
        "litmapr.cli.make_embedder", lambda model: FakeEmbedder(KEYWORDS)
    )
    llm = ScriptedLLM([{"same": False, "reason": "different people"}] * 10)
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: llm)
    result = runner.invoke(app, ["merge", "--threshold", "0.5"])
    assert "different:" in result.stdout
    with StoryStore("data/litmapr.db") as store:
        assert len(store.load().entities) == 4


def test_manual_merge_entities():
    with StoryStore("data/litmapr.db") as store:
        store.save(reed_story())
    result = runner.invoke(app, ["merge-entities", "Reed Richards", "Mr. Fantastic"])
    assert result.exit_code == 0
    with StoryStore("data/litmapr.db") as store:
        [reed] = store.find_entities("mr. fantastic")
        assert reed.name == "Reed Richards"


def test_manual_merge_unknown_name():
    with StoryStore("data/litmapr.db") as store:
        store.save(reed_story())
    result = runner.invoke(app, ["merge-entities", "Reed Richards", "Galactus"])
    assert result.exit_code == 1


def test_ollama_embedder_embeds_each_text_once(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from litmapr.cache import ResponseCache
    from litmapr.llm.ollama_text import OllamaEmbedder

    sent = []

    def fake_embed(model, input):
        sent.append(list(input))
        return SimpleNamespace(embeddings=[[float(len(text)), 1.0] for text in input])

    monkeypatch.setattr("litmapr.llm.ollama_text.ollama.embed", fake_embed)
    embedder = OllamaEmbedder("fake", cache=ResponseCache("e", tmp_path))
    first = embedder.embed(["Reed", "Carol"])
    second = embedder.embed(["Carol", "Ross"])
    assert sent == [["Reed", "Carol"], ["Ross"]]  # "Carol" came from the cache
    assert second[0] == first[1]


def test_roster_groups_codenames_with_real_names():
    from litmapr.resolve import (
        combine,
        find_duplicate_candidates,
        find_roster_candidates,
    )

    story = Story()
    sue = story.add_entity("Susan Storm", EntityKind.CHARACTER)
    woman = story.add_entity("Invisible Woman", EntityKind.CHARACTER)
    story.add_entity("Val", EntityKind.CHARACTER)
    story.add_entity("Baxter Building", EntityKind.LOCATION)  # alone in its kind
    llm = FakeLLM(
        {
            "groups": [
                {
                    "names": ["Susan Storm", "invisible woman", "Nobody"],
                    "reason": "codename",
                }
            ]
        }
    )
    found = find_roster_candidates(llm, story)
    assert [(c.a.id, c.b.id) for c in found] == [(sue.id, woman.id)]
    assert found[0].why == "roster: codename"
    assert len(llm.prompts) == 1  # the location had nothing to pair with
    assert "Invisible Woman" in llm.prompts[0]

    # The same pair from embeddings isn't checked twice.
    again = find_duplicate_candidates(story, FakeEmbedder(KEYWORDS), threshold=0.0)
    merged = combine(found, again)
    pairs = [frozenset((c.a.id, c.b.id)) for c in merged]
    assert len(pairs) == len(set(pairs))


def test_garbled_roster_answer_is_ignored():
    from litmapr.resolve import find_roster_candidates

    story = Story()
    story.add_entity("A", EntityKind.CHARACTER)
    story.add_entity("B", EntityKind.CHARACTER)
    assert find_roster_candidates(FakeLLM("not json"), story) == []


def test_merge_finds_codenames_embeddings_miss(monkeypatch):
    from litmapr.portraits import portrait_path

    story = Story()
    work = Work(series="FF", issue="570", year=2009, month=10)
    story.works.append(work)
    sue = story.add_entity("Susan Storm", EntityKind.CHARACTER)
    woman = story.add_entity("Invisible Woman", EntityKind.CHARACTER)
    for entity, count in ((sue, 3), (woman, 1)):
        for page in range(count):
            story.mentions.append(
                Mention(
                    entity_id=entity.id, work_id=work.id, page=page + 1, summary="x"
                )
            )
    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    picture = portrait_path(woman.id)
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"png")
    monkeypatch.setattr(
        "litmapr.cli.make_embedder", lambda model: FakeEmbedder(KEYWORDS)
    )
    llm = ScriptedLLM(
        [
            {
                "groups": [
                    {"names": ["Susan Storm", "Invisible Woman"], "reason": "codename"}
                ]
            },
            {"same": True, "reason": "Invisible Woman is Sue Storm."},
        ]
    )
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: llm)

    result = runner.invoke(app, ["merge", "--threshold", "1.0"])
    assert result.exit_code == 0, result.stdout
    assert "roster: codename" in result.stdout
    assert "merged 'Invisible Woman' into 'Susan Storm'" in result.stdout
    assert portrait_path(sue.id).read_bytes() == b"png"  # the picture moved with it
