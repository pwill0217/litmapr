from typer.testing import CliRunner

from litmapr.cli import app
from litmapr.store import StoryStore

runner = CliRunner()


def test_extract_saves_to_database(pages_json, fake_llm):
    result = runner.invoke(
        app, ["extract", str(pages_json), "--year", "2026", "--month", "5"]
    )
    assert result.exit_code == 0, result.stdout
    assert "3 new entities" in result.stdout
    with StoryStore("data/litmapr.db") as store:
        story = store.load()
    [work] = story.works
    assert (work.series, work.issue, work.year) == ("Avengers Armageddon", "5", 2026)
    assert len(story.mentions) == 3


def test_extract_twice_replaces_instead_of_duplicating(pages_json, fake_llm):
    runner.invoke(app, ["extract", str(pages_json)])
    runner.invoke(app, ["extract", str(pages_json)])
    with StoryStore("data/litmapr.db") as store:
        story = store.load()
    assert len(story.works) == 1
    assert len(story.entities) == 3
    assert len(story.mentions) == 3


def test_extract_series_override(pages_json, fake_llm):
    runner.invoke(app, ["extract", str(pages_json), "--series", "Avengers: Armageddon"])
    with StoryStore("data/litmapr.db") as store:
        assert store.load().works[0].series == "Avengers: Armageddon"


def test_show_by_alias(pages_json, fake_llm):
    runner.invoke(app, ["extract", str(pages_json)])
    result = runner.invoke(app, ["show", "red hulk"])
    assert result.exit_code == 0
    assert "Thaddeus Ross (character)" in result.stdout
    assert "first appearance: Avengers Armageddon #5, page 1" in result.stdout
    assert "conquers and renames it" in result.stdout


def test_show_quote(pages_json, fake_llm):
    runner.invoke(app, ["extract", str(pages_json)])
    result = runner.invoke(app, ["show", "Captain Marvel"])
    assert '"HOW DIMINISHED?"' in result.stdout


def test_show_unknown_name(pages_json, fake_llm):
    runner.invoke(app, ["extract", str(pages_json)])
    result = runner.invoke(app, ["show", "Galactus"])
    assert result.exit_code == 1
    assert "Nothing called 'Galactus'" in result.stdout


def test_extract_when_ollama_is_down(pages_json, monkeypatch):
    class Down:
        model = "down"

        def complete_json(self, prompt, schema):
            raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: Down())
    result = runner.invoke(app, ["extract", str(pages_json)])
    assert result.exit_code == 1
    assert "Is it running?" in result.stdout


def test_entities_lists_most_mentioned_first(pages_json, fake_llm):
    runner.invoke(app, ["extract", str(pages_json)])
    result = runner.invoke(app, ["entities"])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if "(" in line]
    assert len(lines) == 3
    assert "aka Red Hulk" in result.stdout
    assert "3 entities" in result.stdout


def test_entities_filter_by_kind(pages_json, fake_llm):
    runner.invoke(app, ["extract", str(pages_json)])
    result = runner.invoke(app, ["entities", "--kind", "location"])
    assert "Latveria" in result.stdout
    assert "Carol" not in result.stdout


def test_entities_empty_database():
    result = runner.invoke(app, ["entities"])
    assert result.exit_code == 1


def test_add_reads_and_extracts_in_one_step(sample_cbz, fake_extractor, fake_llm):
    result = runner.invoke(
        app, ["add", str(sample_cbz), "--year", "2026", "--month", "5"]
    )
    assert result.exit_code == 0, result.stdout
    assert "page 3: model" in result.stdout  # ingest ran with OCR
    assert "new entities" in result.stdout  # extract ran
    with StoryStore("data/litmapr.db") as store:
        [work] = store.load().works
    assert work.year == 2026


def test_add_folder(sample_cbz, sample_pdf, fake_extractor, fake_llm):
    result = runner.invoke(app, ["add", str(sample_cbz.parent)])
    assert result.exit_code == 0, result.stdout
    assert "[1/2]" in result.stdout and "[2/2]" in result.stdout


def test_export_then_import_into_new_database(pages_json, fake_llm, tmp_path):
    runner.invoke(app, ["extract", str(pages_json)])
    shared = tmp_path / "shared.json"
    assert runner.invoke(app, ["export", str(shared)]).exit_code == 0
    result = runner.invoke(
        app, ["import", str(shared), "--db", str(tmp_path / "friend.db")]
    )
    assert result.exit_code == 0
    with StoryStore(tmp_path / "friend.db") as store:
        assert len(store.load().entities) == 3


def test_show_lists_callbacks():
    from test_web import two_issue_story

    with StoryStore("data/litmapr.db") as store:
        store.save(two_issue_story())
    result = runner.invoke(app, ["show", "Doctor Ilo"])
    assert "Star Harbor #2 p3 -> Star Harbor #1 p2: It fails." in result.stdout


def test_add_uses_the_files_own_series_issue_and_date(
    tagged_cbz, fake_extractor, fake_llm
):
    result = runner.invoke(app, ["add", str(tagged_cbz)])
    assert result.exit_code == 0, result.stdout
    with StoryStore("data/litmapr.db") as store:
        [work] = store.load().works
    # Not "some download name": the file's ComicInfo.xml wins over the filename.
    assert (work.series, work.issue, work.year, work.month) == ("FF", "5", 2012, 3)


def test_options_beat_the_files_metadata(tagged_cbz, fake_extractor, fake_llm):
    runner.invoke(
        app, ["add", str(tagged_cbz), "--series", "FF (2011)", "--year", "2013"]
    )
    with StoryStore("data/litmapr.db") as store:
        [work] = store.load().works
    assert (work.series, work.year, work.month) == ("FF (2011)", 2013, 3)


def test_add_folder_refuses_one_date_for_every_issue(sample_cbz):
    result = runner.invoke(app, ["add", str(sample_cbz.parent), "--year", "2012"])
    assert result.exit_code == 1
    assert "each" in result.stdout


def test_add_frees_the_vision_model_before_extracting(
    sample_cbz, fake_extractor, fake_llm, monkeypatch
):
    unloaded = []
    monkeypatch.setattr(
        "litmapr.cli.ollama.generate",
        lambda model, keep_alive: unloaded.append((model, keep_alive)),
    )
    result = runner.invoke(app, ["add", str(sample_cbz)])
    assert result.exit_code == 0, result.stdout
    assert unloaded == [("qwen2.5vl:3b", 0)]


def test_load_timeout_explains_the_fix(pages_json, monkeypatch):
    import ollama

    class SlowToLoad:
        model = "big"

        def complete_json(self, prompt, schema):
            raise ollama.ResponseError("timed out waiting for llama-server to start")

    monkeypatch.setattr("litmapr.cli.make_text_llm", lambda model: SlowToLoad())
    result = runner.invoke(app, ["extract", str(pages_json)])
    assert result.exit_code == 1
    assert "OLLAMA_LOAD_TIMEOUT" in result.stdout


def test_missing_model_says_how_to_get_it(capsys):
    import ollama
    import pytest
    import typer

    from litmapr.cli import friendly_ollama_errors

    with pytest.raises(typer.Exit):
        with friendly_ollama_errors("qwen3.6:35b-a3b-coding"):
            raise ollama.ResponseError("model 'nomic-embed-text' not found", 404)
    out = capsys.readouterr().out
    assert "ollama pull nomic-embed-text" in out
    assert "qwen3.6" not in out
