"""The Studio: every litmapr command, from the browser.

Long commands (reading pages, merging, pictures) run in the background through
a JobRunner, exactly as if you'd typed them, so the page shows their live
output. Quick things (listing folders, editing a contents file, removing an
issue, export and import) are answered directly.

The Studio can read your folders and run commands, so it only answers this
computer: requests must name a local host (which stops "DNS rebinding" tricks)
and carry a custom header that other websites can't add without permission.
"""

import os
import re
import string
from pathlib import Path

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError

from litmapr.contents import ContentsError, contents_path_for, parse_contents
from litmapr.extract.ollama_vision import DEFAULT_MODEL as DEFAULT_VISION_MODEL
from litmapr.llm.ollama_text import DEFAULT_EMBED_MODEL, DEFAULT_NARRATOR_MODEL
from litmapr.llm.ollama_text import DEFAULT_MODEL as DEFAULT_TEXT_MODEL
from litmapr.model import EntityKind, Story
from litmapr.narrator import work_label
from litmapr.store import StoryStore
from litmapr.web.jobs import JobRunner

COMIC_TYPES = (".cbz", ".pdf")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
HEADER = "X-Litmapr-Studio"
MODEL_NAME = re.compile(r"^[A-Za-z0-9][\w.:/-]*$")  # e.g. qwen2.5vl:3b
LIBRARY = Path("data") / "library"  # where dropped-in files are kept


class Models(BaseModel):
    vision: str | None = None
    text: str | None = None
    embed: str | None = None

    def flag(self, option: str, which: str) -> list[str]:
        name = getattr(self, which)
        if not name:
            return []
        if not MODEL_NAME.match(name):
            raise HTTPException(status_code=400, detail=f"'{name}' isn't a model name.")
        return [option, name]


class AddForm(BaseModel):
    path: str
    series: str | None = None
    year: int | None = Field(None, ge=1800, le=2200)
    month: int | None = Field(None, ge=1, le=12)
    models: Models = Models()


class ContentsForm(BaseModel):
    path: str
    force: bool = False


class MergeForm(BaseModel):
    dry_run: bool = True
    models: Models = Models()


class MergeTwoForm(BaseModel):
    keep: str = Field(min_length=1)
    drop: str = Field(min_length=1)


class LinkForm(BaseModel):
    models: Models = Models()


class PicturesForm(BaseModel):
    kinds: list[EntityKind] = []
    top: int = Field(30, ge=1, le=500)
    force: bool = False
    name: str | None = None
    issue: str | None = None
    page: int | None = Field(None, ge=1)
    models: Models = Models()


class ContentsText(BaseModel):
    path: str
    text: str


def existing(path: str, folder_ok: bool = False) -> Path:
    """The comic file (or folder) the form names, or a clear error."""
    found = Path(path.strip().strip('"')).expanduser()
    if not found.exists():
        raise HTTPException(status_code=400, detail=f"Nothing at {found}")
    if found.is_dir() and not folder_ok:
        raise HTTPException(status_code=400, detail=f"{found} is a folder, not a file")
    if found.is_file() and found.suffix.lower() not in COMIC_TYPES:
        kinds = ", ".join(COMIC_TYPES)
        raise HTTPException(status_code=400, detail=f"litmapr reads {kinds} files")
    return found.resolve()


def host_name(header: str) -> str:
    """'127.0.0.1:8000' -> '127.0.0.1', '[::1]:8000' -> '[::1]'."""
    if header.startswith("["):
        return header[: header.find("]") + 1]
    return header.split(":")[0]


def build_router(
    db_path: Path, runner: JobRunner, allowed_hosts: set[str] = LOCAL_HOSTS
) -> APIRouter:
    db = str(db_path)

    def only_this_computer(request: Request) -> None:
        if host_name(request.headers.get("host", "")) not in allowed_hosts:
            raise HTTPException(
                status_code=403, detail="The Studio only works locally."
            )
        if request.headers.get(HEADER) != "1":
            raise HTTPException(status_code=403, detail="Missing the Studio header.")

    router = APIRouter(prefix="/api/studio", dependencies=[Depends(only_this_computer)])

    def start(title: str, args: list[str]) -> dict:
        return runner.submit(title, args).summary()

    # ---------------------------------------------------------------- settings
    @router.get("/config")
    def config() -> dict:
        return {
            "models": {
                "vision": DEFAULT_VISION_MODEL,
                "text": DEFAULT_TEXT_MODEL,
                "embed": DEFAULT_EMBED_MODEL,
                "narrator": DEFAULT_NARRATOR_MODEL,
            },
            "types": list(COMIC_TYPES),
            "kinds": [k.value for k in EntityKind],
            "home": str(Path.home()),
            "library": str(LIBRARY.resolve()),
            "db": str(Path(db).resolve()),
        }

    # ---------------------------------------------------------------- files
    @router.get("/browse")
    def browse(path: str = "") -> dict:
        """Folders and comic files in one folder. No path: drives on Windows,
        your home folder elsewhere."""
        if not path:
            if os.name == "nt":
                drives = [
                    f"{d}:\\"
                    for d in string.ascii_uppercase
                    if Path(f"{d}:\\").exists()
                ]
                return {
                    "path": "",
                    "parent": None,
                    "folders": [{"name": d, "path": d} for d in drives],
                    "files": [],
                }
            path = str(Path.home())
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"{folder} isn't a folder")
        folders, files = [], []
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            raise HTTPException(
                status_code=403, detail=f"Can't open {folder}"
            ) from None
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    folders.append({"name": entry.name, "path": str(entry)})
                elif entry.suffix.lower() in COMIC_TYPES:
                    files.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "size": entry.stat().st_size,
                            "contents": contents_path_for(entry).exists(),
                        }
                    )
            except OSError:
                continue  # a broken link or a file we may not look at
        parent = folder.parent if folder.parent != folder else None
        if parent is None and os.name == "nt":
            parent = ""  # from C:\ go up to the list of drives
        return {
            "path": str(folder),
            "parent": None if parent is None else str(parent),
            "folders": folders,
            "files": files,
        }

    @router.put("/upload")
    async def upload(request: Request, name: str) -> dict:
        """A file dragged onto the page, saved into data/library."""
        clean = Path(name).name  # never a path: no ../ tricks
        if Path(clean).suffix.lower() not in COMIC_TYPES or clean.startswith("."):
            raise HTTPException(
                status_code=400, detail="litmapr reads .cbz and .pdf files"
            )
        LIBRARY.mkdir(parents=True, exist_ok=True)
        target = LIBRARY / clean
        partial = target.with_name(target.name + ".part")
        async with await anyio.open_file(partial, "wb") as out:
            async for chunk in request.stream():
                await out.write(chunk)
        os.replace(partial, target)
        return {"path": str(target.resolve()), "size": target.stat().st_size}

    @router.get("/contents")
    def read_contents(path: str) -> dict:
        comic = existing(path)
        target = contents_path_for(comic)
        text = target.read_text(encoding="utf-8") if target.exists() else ""
        return {"path": str(target), "exists": target.exists(), "text": text}

    @router.put("/contents")
    def write_contents(form: ContentsText) -> dict:
        comic = existing(form.path)
        try:
            sections = parse_contents(form.text)
        except ContentsError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        target = contents_path_for(comic)
        target.write_text(form.text, encoding="utf-8")
        issues = [s for s in sections if not s.skip]
        return {"path": str(target), "issues": len(issues)}

    # ---------------------------------------------------------------- library
    @router.get("/library")
    def library() -> list[dict]:
        with StoryStore(db) as store:
            story = store.load()
        counts: dict[str, set[str]] = {}
        for mention in story.mentions:
            counts.setdefault(mention.work_id, set()).add(mention.entity_id)
        return [
            {
                "id": w.id,
                "label": work_label(w),
                "series": w.series,
                "issue": w.issue,
                "year": w.year,
                "month": w.month,
                # The name only, split by hand: a map made on Windows has
                # backslashes, which Path() on Mac or Linux won't split.
                "file": re.split(r"[\\/]", w.source_path)[-1]
                if w.source_path
                else None,
                "entities": len(counts.get(w.id, ())),
            }
            for w in story.reading_order()
        ]

    @router.delete("/works/{work_id}")
    def remove_work(work_id: str) -> dict:
        if runner.busy():
            raise HTTPException(
                status_code=409, detail="Wait for the running job to finish."
            )
        with StoryStore(db) as store:
            if not any(w.id == work_id for w in store.load().works):
                raise HTTPException(status_code=404, detail="No such issue")
            store.delete_work(work_id)
            orphans = store.remove_orphans()
        return {"removed": work_id, "orphans": orphans}

    @router.get("/export")
    def export() -> Response:
        with StoryStore(db) as store:
            story = store.load()
        for work in story.works:
            work.source_path = None  # your folders are nobody else's business
        return Response(
            story.model_dump_json(indent=1),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="litmapr-map.json"'},
        )

    @router.post("/import")
    async def import_map(request: Request) -> dict:
        if runner.busy():
            raise HTTPException(
                status_code=409, detail="Wait for the running job to finish."
            )
        try:
            story = Story.model_validate_json(await request.body())
        except ValidationError:
            raise HTTPException(
                status_code=400, detail="That isn't a map exported from litmapr."
            ) from None
        if not story.works:
            raise HTTPException(status_code=400, detail="That map has no issues in it.")
        problems = story.broken_references()
        if problems:
            raise HTTPException(
                status_code=400,
                detail=f"Not imported: the map has {len(problems)} broken references.",
            )
        with StoryStore(db) as store:
            store.save(story)
        return {"works": len(story.works), "entities": len(story.entities)}

    # ---------------------------------------------------------------- commands
    @router.post("/run/add")
    def run_add(form: AddForm) -> dict:
        path = existing(form.path, folder_ok=True)
        args = ["add", "--db", db]
        if form.series:
            args += ["--series", form.series]
        if form.year:
            args += ["--year", str(form.year)]
        if form.month:
            args += ["--month", str(form.month)]
        args += form.models.flag("--vision-model", "vision")
        args += form.models.flag("--model", "text")
        return start(f"Add {path.name}", [*args, "--", str(path)])

    @router.post("/run/contents")
    def run_contents(form: ContentsForm) -> dict:
        path = existing(form.path)
        args = ["contents", *(["--force"] if form.force else []), "--", str(path)]
        return start(f"Draft contents for {path.name}", args)

    @router.post("/run/merge")
    def run_merge(form: MergeForm) -> dict:
        args = ["merge", "--db", db, *(["--dry-run"] if form.dry_run else [])]
        args += form.models.flag("--model", "text")
        args += form.models.flag("--embed-model", "embed")
        return start("Preview merges" if form.dry_run else "Merge duplicates", args)

    @router.post("/run/merge-entities")
    def run_merge_two(form: MergeTwoForm) -> dict:
        args = ["merge-entities", "--db", db, "--", form.keep, form.drop]
        return start(f"Merge {form.drop} into {form.keep}", args)

    @router.post("/run/link")
    def run_link(form: LinkForm) -> dict:
        args = ["link", "--db", db]
        args += form.models.flag("--model", "text")
        args += form.models.flag("--embed-model", "embed")
        return start("Find callbacks", args)

    @router.post("/run/portraits")
    def run_portraits(form: PicturesForm) -> dict:
        args = ["portraits", "--db", db, "--top", str(form.top)]
        for kind in form.kinds:
            args += ["--kind", kind.value]
        if form.force:
            args.append("--force")
        if form.name:
            args += ["--name", form.name]
            if form.page:
                args += ["--page", str(form.page)]
                if form.issue:
                    args += ["--issue", form.issue]
        elif form.page:
            raise HTTPException(status_code=400, detail="A page needs a name.")
        args += form.models.flag("--model", "vision")
        title = f"Pictures: {form.name}" if form.name else "Make pictures"
        return start(title, args)

    # ---------------------------------------------------------------- jobs
    @router.get("/jobs")
    def jobs() -> list[dict]:
        return [job.summary() for job in runner.all()]

    @router.get("/jobs/{job_id}")
    def job_log(job_id: str, since: int = 0) -> dict:
        job = runner.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return {**job.summary(), "since": since, "log": job.log_from(since)}

    @router.post("/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict:
        job = runner.stop(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job.summary()

    return router
