import pytest
from conftest import FakeLLM
from fastapi.testclient import TestClient

from litmapr.model import Callback, EntityKind, Mention, Relation, Story, Work
from litmapr.store import StoryStore
from litmapr.web.api import create_app


def two_issue_story() -> Story:
    """Issue 1: Ilo builds the Lantern. Issue 2: it fails, and Rook appears."""
    s = Story()
    w1 = Work(series="Star Harbor", issue="1", year=2020, month=1)
    w2 = Work(series="Star Harbor", issue="2", year=2020, month=2)
    s.works = [w2, w1]
    ilo = s.add_entity("Doctor Ilo", EntityKind.CHARACTER)
    lantern = s.add_entity("The Lantern", EntityKind.DEVICE)
    rook = s.add_entity("Captain Rook", EntityKind.CHARACTER)
    built = Mention(
        entity_id=ilo.id, work_id=w1.id, page=2, summary="Builds the Lantern."
    )
    fails = Mention(entity_id=lantern.id, work_id=w2.id, page=3, summary="Fails.")
    s.mentions = [
        built,
        Mention(entity_id=lantern.id, work_id=w1.id, page=2, summary="Switched on."),
        fails,
        Mention(entity_id=rook.id, work_id=w2.id, page=7, summary="Arrives."),
    ]
    s.relations = [
        Relation(
            source_id=ilo.id,
            target_id=lantern.id,
            kind="created",
            label="built it",
            work_id=w1.id,
            page=2,
        ),
        Relation(
            source_id=rook.id,
            target_id=ilo.id,
            kind="ally",
            label="saves him",
            work_id=w2.id,
            page=7,
        ),
    ]
    s.callbacks = [
        Callback(from_mention_id=fails.id, to_mention_id=built.id, reason="It fails.")
    ]
    return s


@pytest.fixture
def story(tmp_path):
    s = two_issue_story()
    with StoryStore(tmp_path / "web.db") as store:
        store.save(s)
    return s


@pytest.fixture
def client(story, tmp_path):
    return TestClient(
        create_app(
            tmp_path / "web.db", llm_factory=lambda: FakeLLM("Ilo is a builder.")
        )
    )


def entity_id(story, name):
    return story.find_entities(name)[0].id


def test_page_and_library_are_served(client):
    assert "litmapr" in client.get("/").text
    assert client.get("/static/vendor/3d-force-graph.min.js").status_code == 200


def test_works_in_reading_order(client):
    labels = [w["label"] for w in client.get("/api/works").json()]
    assert labels == ["Star Harbor #1", "Star Harbor #2"]


def test_full_graph(client):
    data = client.get("/api/graph").json()
    assert {n["name"] for n in data["nodes"]} == {
        "Doctor Ilo",
        "The Lantern",
        "Captain Rook",
    }
    kinds = sorted(k for link in data["links"] for k in link["kinds"])
    assert kinds == ["ally", "callback", "created"]


def test_graph_hides_spoilers(client):
    data = client.get("/api/graph?upto=0").json()  # only issue 1 read
    assert "Captain Rook" not in {n["name"] for n in data["nodes"]}
    assert all("callback" not in link["kinds"] for link in data["links"])


def test_first_appearance_position_for_timeline(client, story):
    nodes = {n["name"]: n for n in client.get("/api/graph").json()["nodes"]}
    assert nodes["Doctor Ilo"]["first"] == 0
    assert nodes["Captain Rook"]["first"] == 1


def test_entity_detail(client, story):
    detail = client.get(f"/api/entity/{entity_id(story, 'Doctor Ilo')}").json()
    assert detail["first_appearance"]["work"] == "Star Harbor #1"
    assert [r["other"] for r in detail["relations"]] == ["The Lantern", "Captain Rook"]
    assert detail["callbacks"][0]["reason"] == "It fails."


def test_entity_detail_hides_spoilers(client, story):
    detail = client.get(f"/api/entity/{entity_id(story, 'Doctor Ilo')}?upto=0").json()
    assert [r["other"] for r in detail["relations"]] == ["The Lantern"]
    assert detail["callbacks"] == []


def test_unknown_entity_is_404(client):
    assert client.get("/api/entity/nope").status_code == 404


def test_narrate(client, story):
    response = client.post(f"/api/narrate/{entity_id(story, 'Doctor Ilo')}")
    assert response.json() == {"text": "Ilo is a builder."}


def test_narrator_can_be_switched_off(story, tmp_path):
    client = TestClient(create_app(tmp_path / "web.db", llm_factory=None))
    assert client.get("/api/config").json()["narrator"] is False
    response = client.post(f"/api/narrate/{entity_id(story, 'Doctor Ilo')}")
    assert response.status_code == 403


def test_narrate_when_ollama_is_down(story, tmp_path):
    class Down(FakeLLM):
        def complete_text(self, prompt):
            raise ConnectionError("down")

    client = TestClient(create_app(tmp_path / "web.db", llm_factory=lambda: Down("")))
    response = client.post(f"/api/narrate/{entity_id(story, 'Doctor Ilo')}")
    assert response.status_code == 503
    assert "Is it running?" in response.json()["detail"]


# --- dossier and portraits -------------------------------------------------------------


def test_dossier_stats_and_presence(client, story):
    detail = client.get(f"/api/entity/{entity_id(story, 'The Lantern')}").json()
    assert detail["slug"] == "the-lantern"
    assert detail["stats"] == {
        "appearances": 2,
        "issues": 2,
        "relations": 1,
        "callbacks": 1,
    }
    assert [(p["label"], p["count"]) for p in detail["presence"]] == [
        ("Star Harbor #1", 1),
        ("Star Harbor #2", 1),
    ]


def test_presence_stops_at_reading_point(client, story):
    detail = client.get(f"/api/entity/{entity_id(story, 'The Lantern')}?upto=0").json()
    assert [p["label"] for p in detail["presence"]] == ["Star Harbor #1"]


def test_portrait_is_served_when_it_exists(story, tmp_path):
    folder = tmp_path / "portraits"
    folder.mkdir()
    ilo = entity_id(story, "Doctor Ilo")
    (folder / f"{ilo}.png").write_bytes(b"\x89PNG fake")
    client = TestClient(create_app(tmp_path / "web.db", portraits=folder))
    nodes = {n["name"]: n for n in client.get("/api/graph").json()["nodes"]}
    assert nodes["Doctor Ilo"]["portrait"] == f"/api/portrait/{ilo}"
    assert nodes["Captain Rook"]["portrait"] is None
    assert client.get(f"/api/portrait/{ilo}").status_code == 200


def test_portraits_can_be_switched_off(story, tmp_path):
    client = TestClient(create_app(tmp_path / "web.db", portraits=None))
    assert client.get("/api/config").json()["portraits"] is False
    assert (
        client.get(f"/api/portrait/{entity_id(story, 'Doctor Ilo')}").status_code == 404
    )


def test_portrait_endpoint_rejects_path_tricks(story, tmp_path):
    client = TestClient(create_app(tmp_path / "web.db", portraits=tmp_path))
    assert client.get("/api/portrait/..%2Fweb").status_code == 404


def test_narration_streams(client, story):
    response = client.get(f"/api/narrate/{entity_id(story, 'Doctor Ilo')}/stream")
    assert response.status_code == 200
    assert response.text == "Ilo is a builder."


def test_narrator_styles(client, story):
    assert client.get("/api/config").json()["styles"] == ["storyteller", "briefing"]
    bad = client.post(f"/api/narrate/{entity_id(story, 'Doctor Ilo')}?style=opera")
    assert bad.status_code == 400


def test_stream_reports_ollama_down(story, tmp_path):
    class Down(FakeLLM):
        def stream_text(self, prompt):
            raise ConnectionError
            yield ""

    client = TestClient(create_app(tmp_path / "web.db", llm_factory=lambda: Down("")))
    response = client.get(f"/api/narrate/{entity_id(story, 'Doctor Ilo')}/stream")
    assert response.status_code == 503


def test_replace_and_remove_a_picture(story, tmp_path, monkeypatch):
    import io

    from PIL import Image

    monkeypatch.setattr("litmapr.portraits.remove_background", lambda p: (p, False))
    folder = tmp_path / "pics"
    client = TestClient(create_app(tmp_path / "web.db", portraits=folder))
    ilo = entity_id(story, "Doctor Ilo")
    buffer = io.BytesIO()
    Image.new("RGB", (50, 80), "purple").save(buffer, format="PNG")

    put = client.put(f"/api/portrait/{ilo}", content=buffer.getvalue())
    assert put.status_code == 200, put.text
    assert put.json()["portrait"] == f"/api/portrait/{ilo}"
    assert client.get(f"/api/portrait/{ilo}").status_code == 200

    assert (
        client.put(f"/api/portrait/{ilo}", content=b"not a picture").status_code == 400
    )
    assert client.delete(f"/api/portrait/{ilo}").json() == {"removed": True}
    assert client.get(f"/api/portrait/{ilo}").status_code == 404


def test_pictures_cant_be_changed_when_turned_off(story, tmp_path):
    client = TestClient(create_app(tmp_path / "web.db", portraits=None))
    ilo = entity_id(story, "Doctor Ilo")
    assert client.delete(f"/api/portrait/{ilo}").status_code == 403
