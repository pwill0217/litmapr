from conftest import FakeLLM
from test_web import two_issue_story
from typer.testing import CliRunner

from litmapr.cache import ResponseCache
from litmapr.cli import app
from litmapr.narrator import build_context, build_prompt, narrate
from litmapr.store import StoryStore

runner = CliRunner()


def ilo(story):
    return story.find_entities("Doctor Ilo")[0].id


def test_context_has_appearances_relations_and_callbacks():
    story = two_issue_story()
    context = build_context(story, ilo(story), upto=None)
    assert "Star Harbor #1 p2: Builds the Lantern." in context
    assert "Captain Rook -> Doctor Ilo (ally): saves him" in context
    assert "refers back to Star Harbor #1 p2" in context


def test_context_stops_at_reading_point():
    story = two_issue_story()
    context = build_context(story, ilo(story), upto=0)
    assert "Captain Rook" not in context  # issue 2 is a spoiler here
    assert "CALLBACKS" not in context


def test_prompt_states_progress_and_rules():
    story = two_issue_story()
    prompt = build_prompt(story, ilo(story), upto=0)
    assert "read up to Star Harbor #1" in prompt
    assert "Use ONLY the facts below" in prompt


def test_narration_is_cached(tmp_path):
    story = two_issue_story()
    cache = ResponseCache("n", tmp_path)
    first = FakeLLM("Once upon a time.")
    narrate(first, story, ilo(story), None, cache)
    second = FakeLLM("Different words.")
    assert narrate(second, story, ilo(story), None, cache) == "Once upon a time."
    assert second.prompts == []


def test_new_facts_mean_new_narration(tmp_path):
    story = two_issue_story()
    cache = ResponseCache("n", tmp_path)
    narrate(FakeLLM("old"), story, ilo(story), None, cache)
    story.entities[0].description = "Now we know more."
    assert narrate(FakeLLM("new"), story, ilo(story), None, cache) == "new"


def test_narrate_command(monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(two_issue_story())
    monkeypatch.setattr(
        "litmapr.cli.make_text_llm", lambda model: FakeLLM("Ilo builds things.")
    )
    result = runner.invoke(app, ["narrate", "Doctor Ilo", "--up-to", "Star Harbor #1"])
    assert result.exit_code == 0, result.stdout
    assert "Ilo builds things." in result.stdout


def test_narrate_up_to_by_number(monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(two_issue_story())
    fake = FakeLLM("ok")
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: fake)
    runner.invoke(app, ["narrate", "Doctor Ilo", "--up-to", "1"])
    assert "Captain Rook" not in fake.prompts[0]


def test_narrate_bad_up_to():
    with StoryStore("data/litmapr.db") as store:
        store.save(two_issue_story())
    result = runner.invoke(app, ["narrate", "Doctor Ilo", "--up-to", "Star Harbor #99"])
    assert result.exit_code == 1


def test_works_command():
    with StoryStore("data/litmapr.db") as store:
        store.save(two_issue_story())
    result = runner.invoke(app, ["works"])
    assert "1. Star Harbor #1  (2020-01)" in result.stdout
    assert "2. Star Harbor #2  (2020-02)" in result.stdout


def test_narrator_writes_for_the_ear():
    story = two_issue_story()
    prompt = build_prompt(story, ilo(story), None)
    assert "spoken aloud" in prompt
    assert "storyteller" in prompt
    assert "intelligence officer" in build_prompt(story, ilo(story), None, "briefing")


def test_streamed_narration_is_cached_whole(tmp_path):
    from litmapr.narrator import narrate_stream

    class Pieces(FakeLLM):
        def stream_text(self, prompt):
            yield from ["Once ", "upon ", "a time."]

    story, cache = two_issue_story(), ResponseCache("narration", tmp_path)
    assert list(narrate_stream(Pieces(""), story, ilo(story), None, cache)) == [
        "Once ",
        "upon ",
        "a time.",
    ]
    assert list(narrate_stream(Pieces(""), story, ilo(story), None, cache)) == [
        "Once upon a time."
    ]


def test_thinking_is_left_out(tmp_path):
    from litmapr.narrator import narrate_stream

    class Thinker(FakeLLM):
        def __init__(self, pieces):
            super().__init__("")
            self.pieces = pieces

        def stream_text(self, prompt):
            yield from self.pieces

    story = two_issue_story()
    tagged = Thinker(
        ["<thi", "nk>Okay, the user wants", " a story.</think>\n\n", "You meet Ilo."]
    )
    assert "".join(narrate_stream(tagged, story, ilo(story), None)) == "You meet Ilo."

    # No opening tag: it streams through, but the saved narration is clean.
    cache = ResponseCache("n", tmp_path)
    bare = Thinker(["Okay, let me think.", "</think>", " You meet Ilo."])
    list(narrate_stream(bare, story, ilo(story), None, cache))
    assert narrate(bare, story, ilo(story), None, cache) == "You meet Ilo."


def test_plain_answers_stream_untouched():
    from litmapr.narrator import narrate_stream

    class Plain(FakeLLM):
        def stream_text(self, prompt):
            yield from ["You ", "meet ", "Ilo."]

    story = two_issue_story()
    assert list(narrate_stream(Plain(""), story, ilo(story), None)) == [
        "You ",
        "meet ",
        "Ilo.",
    ]
