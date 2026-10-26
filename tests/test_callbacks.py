from conftest import FakeEmbedder, ScriptedLLM
from typer.testing import CliRunner

from litmapr.callbacks import find_callback_candidates
from litmapr.cli import app
from litmapr.model import EntityKind, Mention, Story, Work
from litmapr.store import StoryStore

runner = CliRunner()
KEYWORDS = ["bridge", "builds", "breaks", "dinner"]


def bridge_story() -> Story:
    """The Bridge is built in issue 1 and breaks in issue 2. Works added out of order."""
    s = Story()
    issue2 = Work(series="FF", issue="2", year=2012, month=2)
    issue1 = Work(series="FF", issue="1", year=2012, month=1)
    s.works = [issue2, issue1]
    reed = s.add_entity("Reed Richards", EntityKind.CHARACTER)
    bridge = s.add_entity("The Bridge", EntityKind.DEVICE)
    s.mentions = [
        Mention(
            entity_id=bridge.id, work_id=issue2.id, page=4, summary="The Bridge breaks."
        ),
        Mention(
            entity_id=bridge.id,
            work_id=issue1.id,
            page=2,
            summary="Reed builds the Bridge.",
        ),
        Mention(entity_id=reed.id, work_id=issue1.id, page=9, summary="Eats dinner."),
    ]
    return s


def test_candidates_only_point_back_in_reading_order():
    candidates = find_callback_candidates(
        bridge_story(), FakeEmbedder(KEYWORDS), threshold=0.3
    )
    assert candidates
    for c in candidates:
        assert c.later.summary == "The Bridge breaks."
        assert c.earlier.summary == "Reed builds the Bridge."


def test_unrelated_moments_are_not_candidates():
    candidates = find_callback_candidates(
        bridge_story(), FakeEmbedder(KEYWORDS), threshold=0.3
    )
    assert all("dinner" not in c.earlier.summary for c in candidates)


def test_same_issue_is_not_a_callback():
    s = bridge_story()
    issue1 = next(w for w in s.works if w.issue == "1")
    for m in s.mentions:
        m.work_id = issue1.id
    assert find_callback_candidates(s, FakeEmbedder(KEYWORDS), threshold=0.3) == []


def test_link_command_saves_confirmed_callbacks(monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(bridge_story())
    monkeypatch.setattr(
        "litmapr.cli.make_embedder", lambda model: FakeEmbedder(KEYWORDS)
    )
    llm = ScriptedLLM(
        [{"is_callback": True, "reason": "The Bridge Reed built now fails."}]
    )
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: llm)

    result = runner.invoke(app, ["link", "--threshold", "0.3"])
    assert result.exit_code == 0, result.stdout
    assert "1 callbacks saved" in result.stdout
    with StoryStore("data/litmapr.db") as store:
        [callback] = store.load().callbacks
    assert callback.reason == "The Bridge Reed built now fails."


def test_link_remembers_decisions(monkeypatch):
    with StoryStore("data/litmapr.db") as store:
        store.save(bridge_story())
    monkeypatch.setattr(
        "litmapr.cli.make_embedder", lambda model: FakeEmbedder(KEYWORDS)
    )
    first = ScriptedLLM([{"is_callback": False, "reason": "no"}])
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: first)
    runner.invoke(app, ["link", "--threshold", "0.3"])

    second = ScriptedLLM([])
    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: second)
    result = runner.invoke(app, ["link", "--threshold", "0.3"])
    assert "No new possible callbacks" in result.stdout
    assert second.prompts == []
