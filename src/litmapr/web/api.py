"""Phase 5: the web server behind the 3D explorer.

FastAPI turns Python functions into web addresses (endpoints). The browser
page in static/ calls them with fetch() and draws what comes back.
Every endpoint takes `upto` (a reading position) so nothing past the
reader's progress is ever sent to the browser: spoiler control.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from litmapr.cache import ResponseCache
from litmapr.llm.base import TextLLM
from litmapr.model import Story
from litmapr.narrator import (
    DEFAULT_STYLE,
    STYLES,
    narrate_stream,
    page_label,
    visible_work_ids,
    work_label,
)
from litmapr.portraits import (
    DEFAULT_PORTRAIT_DIR,
    portrait_from_bytes,
    portrait_path,
    remove_portrait,
)
from litmapr.store import StoryStore

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 15 * 1024 * 1024  # bytes


def portrait_url(entity_id: str, folder: Path | None) -> str | None:
    if folder is None or not portrait_path(entity_id, folder).exists():
        return None
    return f"/api/portrait/{entity_id}"


def build_graph(story: Story, upto: int | None, portraits: Path | None = None) -> dict:
    """Nodes and links for the 3D view, limited to what the reader has read."""
    order = {w.id: i for i, w in enumerate(story.reading_order())}
    visible = visible_work_ids(story, upto)
    mentions = [m for m in story.mentions if m.work_id in visible]
    relations = [r for r in story.relations if r.work_id in visible]

    counts = Counter(m.entity_id for m in mentions)
    first: dict[str, int] = {}
    for m in mentions:
        position = order[m.work_id]
        first[m.entity_id] = min(first.get(m.entity_id, position), position)
    linked = {r.source_id for r in relations} | {r.target_id for r in relations}

    nodes = [
        {
            "id": e.id,
            "name": e.name,
            "kind": e.kind,
            "aliases": e.aliases,
            "mentions": counts[e.id],
            "first": first.get(e.id, 0),
            "portrait": portrait_url(e.id, portraits),
        }
        for e in story.entities
        if e.id in counts or e.id in linked
    ]

    # Many relations between the same two entities become one link with a count.
    pairs: dict[tuple[str, str], dict] = {}
    for r in relations:
        key = tuple(sorted((r.source_id, r.target_id)))
        link = pairs.setdefault(
            key,
            {"source": key[0], "target": key[1], "kinds": [], "count": 0, "labels": []},
        )
        link["count"] += 1
        if r.kind not in link["kinds"]:
            link["kinds"].append(r.kind)
        link["labels"].append(r.label)

    by_id = {m.id: m for m in mentions}
    callback_links = []
    for c in story.callbacks:
        later, earlier = by_id.get(c.from_mention_id), by_id.get(c.to_mention_id)
        if later and earlier and later.entity_id != earlier.entity_id:
            callback_links.append(
                {
                    "source": later.entity_id,
                    "target": earlier.entity_id,
                    "kinds": ["callback"],
                    "count": 1,
                    "labels": [c.reason],
                }
            )
    return {"nodes": nodes, "links": list(pairs.values()) + callback_links}


def entity_detail(
    story: Story, entity_id: str, upto: int | None, portraits: Path | None = None
) -> dict:
    entity = next((e for e in story.entities if e.id == entity_id), None)
    if entity is None:
        raise KeyError(entity_id)
    visible = visible_work_ids(story, upto)
    works = {w.id: w for w in story.works}
    order = {w.id: i for i, w in enumerate(story.reading_order())}
    names = {e.id: e.name for e in story.entities}

    mentions = [m for m in story.mentions_in_order(entity_id) if m.work_id in visible]
    appearances = [
        {
            "work": work_label(works[m.work_id]),
            "position": order[m.work_id],
            "page": works[m.work_id].local_page(m.page),
            "summary": m.summary,
            "quote": m.quote,
        }
        for m in mentions
    ]
    relations = [
        {
            "other_id": r.target_id if r.source_id == entity_id else r.source_id,
            "other": names[r.target_id if r.source_id == entity_id else r.source_id],
            "direction": "to" if r.source_id == entity_id else "from",
            "kind": r.kind,
            "label": r.label,
            "sentiment": r.sentiment,
            "work": work_label(works[r.work_id]),
            "page": works[r.work_id].local_page(r.page),
        }
        for r in story.relations
        if entity_id in (r.source_id, r.target_id) and r.work_id in visible
    ]
    mention_ids = {m.id for m in mentions}
    by_id = {m.id: m for m in story.mentions}
    callbacks = []
    for c in story.callbacks:
        later, earlier = by_id[c.from_mention_id], by_id[c.to_mention_id]
        if (later.id in mention_ids or earlier.id in mention_ids) and {
            later.work_id,
            earlier.work_id,
        } <= visible:
            callbacks.append(
                {
                    "later": page_label(works[later.work_id], later.page),
                    "earlier": page_label(works[earlier.work_id], earlier.page),
                    "earlier_entity": names[earlier.entity_id],
                    "reason": c.reason,
                }
            )
    # Presence: how often the entity shows up in each issue the reader has reached,
    # for the dossier's timeline strip.
    per_work = Counter(m.work_id for m in mentions)
    presence = [
        {"position": i, "label": work_label(w), "count": per_work[w.id]}
        for i, w in enumerate(story.reading_order())
        if w.id in visible
    ]
    return {
        "id": entity.id,
        "slug": entity.slug,
        "portrait": portrait_url(entity.id, portraits),
        "stats": {
            "appearances": len(appearances),
            "issues": len(per_work),
            "relations": len(relations),
            "callbacks": len(callbacks),
        },
        "presence": presence,
        "name": entity.name,
        "kind": entity.kind,
        "aliases": entity.aliases,
        "description": entity.description,
        "first_appearance": appearances[0] if appearances else None,
        "appearances": appearances,
        "relations": relations,
        "callbacks": callbacks,
    }


def create_app(
    db_path: Path,
    llm_factory: Callable[[], TextLLM] | None = None,
    portraits: Path | None = DEFAULT_PORTRAIT_DIR,
) -> FastAPI:
    """Build the web app. With no llm_factory the narrator is switched off, and
    with portraits=None no comic artwork is served (both useful for a public demo)."""
    app = FastAPI(title="litmapr")
    narration_cache = ResponseCache("narration")

    def load() -> Story:
        # Loaded per request, so a new `extract` shows up on refresh.
        with StoryStore(db_path) as store:
            return store.load()

    @app.get("/api/works")
    def works() -> list[dict]:
        return [
            {
                "position": i,
                "id": w.id,
                "label": work_label(w),
                "year": w.year,
                "month": w.month,
            }
            for i, w in enumerate(load().reading_order())
        ]

    @app.get("/api/graph")
    def graph(upto: int | None = None) -> dict:
        return build_graph(load(), upto, portraits)

    @app.get("/api/entity/{entity_id}")
    def entity(entity_id: str, upto: int | None = None) -> dict:
        try:
            return entity_detail(load(), entity_id, upto, portraits)
        except KeyError:
            raise HTTPException(status_code=404, detail="No such entity") from None

    @app.get("/api/config")
    def config() -> dict:
        return {
            "narrator": llm_factory is not None,
            "portraits": portraits is not None,
            "styles": list(STYLES),
        }

    @app.get("/api/portrait/{entity_id}")
    def portrait(entity_id: str) -> FileResponse:
        # The id goes through portrait_path, never straight into a file path,
        # and must look like one of our ids: no "../" tricks.
        if portraits is None or not entity_id.isalnum():
            raise HTTPException(status_code=404, detail="No portrait")
        path = portrait_path(entity_id, portraits)
        if not path.exists():
            raise HTTPException(status_code=404, detail="No portrait")
        return FileResponse(path, media_type="image/png")

    def picture_target(entity_id: str):
        if portraits is None or not entity_id.isalnum():
            raise HTTPException(status_code=403, detail="Pictures are turned off.")
        entity = next((e for e in load().entities if e.id == entity_id), None)
        if entity is None:
            raise HTTPException(status_code=404, detail="No such entity")
        return entity

    @app.put("/api/portrait/{entity_id}")
    async def replace_portrait(entity_id: str, request: Request) -> dict:
        """Your own picture, sent as the raw file in the request body."""
        entity = picture_target(entity_id)
        data = await request.body()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="That picture is over 15 MB.")
        try:
            portrait_from_bytes(entity, data, portraits)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        return {"portrait": portrait_url(entity_id, portraits)}

    @app.delete("/api/portrait/{entity_id}")
    def delete_portrait(entity_id: str) -> dict:
        picture_target(entity_id)
        return {"removed": remove_portrait(entity_id, portraits)}

    def start_narration(entity_id: str, upto: int | None, style: str):
        """Start the narrator and wait for its first words, so problems become
        proper error responses instead of a stream that just stops."""
        if llm_factory is None:
            raise HTTPException(status_code=403, detail="The narrator is turned off.")
        story = load()
        if not any(e.id == entity_id for e in story.entities):
            raise HTTPException(status_code=404, detail="No such entity")
        if style not in STYLES:
            raise HTTPException(status_code=400, detail=f"Unknown style '{style}'")
        pieces = narrate_stream(
            llm_factory(), story, entity_id, upto, narration_cache, style
        )
        try:
            first = next(pieces, "")
        except ConnectionError:
            raise HTTPException(
                status_code=503, detail="Can't reach Ollama. Is it running?"
            ) from None
        except ollama.ResponseError as error:
            raise HTTPException(
                status_code=502, detail=f"Ollama error: {error.error}"
            ) from None
        return first, pieces

    @app.post("/api/narrate/{entity_id}")
    def narrate_entity(
        entity_id: str, upto: int | None = None, style: str = DEFAULT_STYLE
    ) -> dict:
        first, rest = start_narration(entity_id, upto, style)
        return {"text": (first + "".join(rest)).strip()}

    @app.get("/api/narrate/{entity_id}/stream")
    def narrate_live(
        entity_id: str, upto: int | None = None, style: str = DEFAULT_STYLE
    ) -> StreamingResponse:
        first, rest = start_narration(entity_id, upto, style)

        def words():
            yield first
            yield from rest

        return StreamingResponse(words(), media_type="text/plain; charset=utf-8")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
