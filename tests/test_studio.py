"""The Studio: commands from the browser, run in the background."""

import json
import sys
import time

import pytest
from fastapi.testclient import TestClient
from test_web import two_issue_story
from typer.testing import CliRunner

from litmapr.cli import app
from litmapr.store import StoryStore
from litmapr.web.api import create_app
from litmapr.web.jobs import JobRunner

# A stand-in for `litmapr`: prints the arguments it was given, so tests can see
# exactly which command the Studio would have run. "sleep" makes it wait.
ECHO = [
    sys.executable,
    "-u",
    "-c",
    (
        "import sys, time\n"
        "print('args:', sys.argv[1:])\n"
        "if 'sleep' in sys.argv: time.sleep(30)\n"
        "print('finished')\n"
        "sys.exit(3 if 'fail' in sys.argv else 0)"
    ),
]
HEADERS = {"X-Litmapr-Studio": "1", "Host": "127.0.0.1:8000"}


def wait_for(job_id, runner, done=("done", "failed", "stopped"), timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        job = runner.jobs[job_id]
        if job.status in done:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job still {runner.jobs[job_id].status}")


@pytest.fixture
def runner():
    jobs = JobRunner(program=ECHO)
    yield jobs
    jobs.close()


@pytest.fixture
def studio(tmp_path, runner):
    with StoryStore(tmp_path / "web.db") as store:
        store.save(two_issue_story())
    app_ = create_app(tmp_path / "web.db", runner=runner)
    with TestClient(app_, headers=HEADERS) as client:
        yield client


# ---------------------------------------------------------------- the runner


def test_jobs_run_in_order_and_keep_their_output(runner):
    first = runner.submit("one", ["hello"])
    second = runner.submit("two", ["fail"])
    assert wait_for(first.id, runner).status == "done"
    failed = wait_for(second.id, runner)
    assert failed.status == "failed"
    assert failed.code == 3
    assert first.lines == ["args: ['hello']", "finished"]
    assert first.ended_at <= failed.started_at  # one at a time


def test_a_running_job_can_be_stopped(runner):
    job = runner.submit("slow", ["sleep"])
    wait_for(job.id, runner, done=("running",))
    runner.stop(job.id)
    stopped = wait_for(job.id, runner)
    assert stopped.status == "stopped"
    assert "finished" not in stopped.lines


def test_a_queued_job_can_be_stopped_before_it_starts(runner):
    slow = runner.submit("slow", ["sleep"])
    waiting = runner.submit("next", ["hello"])
    runner.stop(waiting.id)
    assert waiting.status == "stopped"
    runner.stop(slow.id)
    wait_for(slow.id, runner)
    assert waiting.lines == []


def test_log_can_be_read_from_where_you_left_off(runner):
    job = runner.submit("one", ["hello"])
    wait_for(job.id, runner)
    assert job.log_from(1) == ["finished"]


# ---------------------------------------------------------------- safety


def test_studio_only_answers_this_computer(tmp_path, runner):
    app_ = create_app(tmp_path / "web.db", runner=runner)
    client = TestClient(app_)
    no_header = client.get("/api/studio/config", headers={"Host": "127.0.0.1:8000"})
    assert no_header.status_code == 403
    elsewhere = client.get(
        "/api/studio/config", headers={**HEADERS, "Host": "evil.example:8000"}
    )
    assert elsewhere.status_code == 403
    ipv6 = client.get("/api/studio/config", headers={**HEADERS, "Host": "[::1]:8000"})
    assert ipv6.status_code == 200


def test_studio_is_off_without_a_runner(tmp_path):
    client = TestClient(create_app(tmp_path / "web.db"), headers=HEADERS)
    assert client.get("/studio").status_code == 404
    assert client.get("/api/studio/config").status_code == 404
    assert client.get("/api/config").json()["studio"] is False


def test_model_names_cant_smuggle_in_options(studio, tmp_path):
    answer = studio.post("/api/studio/run/merge", json={"models": {"text": "--db=x"}})
    assert answer.status_code == 400


# ---------------------------------------------------------------- commands


def run(studio, runner, command, form):
    answer = studio.post(f"/api/studio/run/{command}", json=form)
    assert answer.status_code == 200, answer.text
    job = wait_for(answer.json()["id"], runner)
    return json.loads(job.lines[0].removeprefix("args: ").replace("'", '"'))


def test_add_runs_the_add_command(studio, runner, sample_cbz, tmp_path):
    args = run(
        studio,
        runner,
        "add",
        {
            "path": str(sample_cbz),
            "year": 2009,
            "month": 10,
            "models": {"vision": "qwen2.5vl:3b"},
        },
    )
    db = str(tmp_path / "web.db")
    assert args == [
        "add",
        "--db",
        db,
        "--year",
        "2009",
        "--month",
        "10",
        "--vision-model",
        "qwen2.5vl:3b",
        "--",
        str(sample_cbz.resolve()),
    ]


def test_add_explains_a_bad_path(studio, tmp_path):
    missing = studio.post(
        "/api/studio/run/add", json={"path": str(tmp_path / "nope.cbz")}
    )
    assert missing.status_code == 400
    assert "Nothing at" in missing.json()["detail"]
    (tmp_path / "notes.txt").write_text("x")
    wrong = studio.post(
        "/api/studio/run/add", json={"path": str(tmp_path / "notes.txt")}
    )
    assert "reads .cbz, .pdf" in wrong.json()["detail"]


def test_merge_preview_and_names_that_look_like_options(studio, runner):
    assert run(studio, runner, "merge", {"dry_run": True})[-1] == "--dry-run"
    args = run(studio, runner, "merge-entities", {"keep": "Reed", "drop": "-Mr. F"})
    assert args[-3:] == ["--", "Reed", "-Mr. F"]


def test_pictures_for_one_page(studio, runner):
    args = run(
        studio,
        runner,
        "portraits",
        {"kinds": ["character", "device"], "name": "Ilo", "page": 4, "issue": "2"},
    )
    assert args[args.index("--kind") + 1] == "character"
    assert args.count("--kind") == 2
    assert args[args.index("--page") + 1] == "4"
    assert args[args.index("--issue") + 1] == "2"
    no_name = studio.post("/api/studio/run/portraits", json={"page": 4})
    assert no_name.status_code == 400


def test_job_list_and_log(studio, runner):
    started = studio.post("/api/studio/run/link", json={}).json()
    wait_for(started["id"], runner)
    log = studio.get(f"/api/studio/jobs/{started['id']}?since=1").json()
    assert log["status"] == "done"
    assert log["log"] == ["finished"]
    assert studio.get("/api/studio/jobs").json()[0]["id"] == started["id"]
    assert studio.get("/api/studio/jobs/nope").status_code == 404


# ---------------------------------------------------------------- files


def test_browse_shows_folders_and_comics(studio, tmp_path, sample_cbz):
    (tmp_path / "Hickman FF").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sample.contents.txt").write_text("1, FF, 570\n")
    listing = studio.get("/api/studio/browse", params={"path": str(tmp_path)}).json()
    assert "Hickman FF" in [f["name"] for f in listing["folders"]]
    assert [f["name"] for f in listing["files"]] == ["sample.cbz"]
    assert listing["files"][0]["contents"] is True
    assert listing["parent"] == str(tmp_path.parent)


def test_upload_keeps_the_file_in_the_library(studio, tmp_path):
    answer = studio.put(
        "/api/studio/upload?name=../../FF 570.cbz", content=b"zip bytes"
    )
    assert answer.status_code == 200, answer.text
    saved = tmp_path / "data" / "library" / "FF 570.cbz"
    assert saved.read_bytes() == b"zip bytes"
    assert (
        studio.put("/api/studio/upload?name=run.exe", content=b"x").status_code == 400
    )


def test_contents_file_is_checked_before_saving(studio, sample_cbz):
    empty = studio.get("/api/studio/contents", params={"path": str(sample_cbz)}).json()
    assert empty["exists"] is False
    bad = studio.put(
        "/api/studio/contents", json={"path": str(sample_cbz), "text": "two, FF, 570"}
    )
    assert bad.status_code == 400
    good = studio.put(
        "/api/studio/contents",
        json={"path": str(sample_cbz), "text": "1, FF, 570, 2009, 10\n3, skip\n"},
    )
    assert good.json()["issues"] == 1
    again = studio.get("/api/studio/contents", params={"path": str(sample_cbz)}).json()
    assert again["text"].startswith("1, FF, 570")


# ---------------------------------------------------------------- the map


def test_library_and_removing_an_issue(studio):
    issues = studio.get("/api/studio/library").json()
    assert [i["label"] for i in issues] == ["Star Harbor #1", "Star Harbor #2"]
    assert issues[1]["entities"] == 2
    gone = studio.delete(f"/api/studio/works/{issues[1]['id']}").json()
    assert gone["orphans"] == 1  # Captain Rook only appeared in issue 2
    assert len(studio.get("/api/studio/library").json()) == 1


def test_export_then_import(studio, tmp_path):
    exported = studio.get("/api/studio/export")
    assert "attachment" in exported.headers["content-disposition"]
    other = create_app(tmp_path / "other.db", runner=JobRunner(program=ECHO))
    with TestClient(other, headers=HEADERS) as client:
        answer = client.post("/api/studio/import", content=exported.content)
        assert answer.json() == {"works": 2, "entities": 3}
        assert client.post("/api/studio/import", content=b"{}").status_code == 400
        assert client.post("/api/studio/import", content=b"not json").status_code == 400


# ---------------------------------------------------------------- the CLI


def test_remove_command(tmp_path):
    with StoryStore("data/litmapr.db") as store:
        store.save(two_issue_story())
    cli = CliRunner()
    result = cli.invoke(app, ["remove", "Star Harbor #2"])
    assert result.exit_code == 0, result.stdout
    assert "removed Star Harbor #2 and 1 entities" in result.stdout
    assert cli.invoke(app, ["remove", "Star Harbor #9"]).exit_code == 1


def test_serve_without_a_map_needs_the_studio():
    result = CliRunner().invoke(app, ["serve", "--no-studio", "--no-open"])
    assert result.exit_code == 1
    assert "No map" in result.stdout


def test_studio_refuses_to_run_open_to_the_network():
    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0", "--studio"])
    assert result.exit_code == 1
    assert "only works" in result.stdout
