import json
import re
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Annotated

import ollama
import typer

from litmapr import callbacks as callback_finder
from litmapr import narrator, resolve
from litmapr.cache import PageCache, ResponseCache, file_hash
from litmapr.contents import (
    ContentsError,
    Section,
    contents_path_for,
    issues,
    page_filter,
    parse_contents,
    read_contents,
    suggest_contents,
)
from litmapr.extract.base import TextExtractor
from litmapr.extract.ollama_vision import DEFAULT_MODEL, OllamaVisionExtractor
from litmapr.extraction import extract_work, guess_series_and_issue
from litmapr.ingest.base import Reader
from litmapr.ingest.cbz_reader import CbzReader
from litmapr.ingest.pdf_reader import PdfReader
from litmapr.llm.base import Embedder, TextLLM
from litmapr.llm.ollama_text import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_NARRATOR_MODEL,
    OllamaEmbedder,
    OllamaTextLLM,
)
from litmapr.llm.ollama_text import DEFAULT_MODEL as DEFAULT_TEXT_MODEL
from litmapr.model import Callback, Entity, EntityKind, Story, Work
from litmapr.narrator import page_label
from litmapr.output import default_output_path, write_pages_json
from litmapr.pipeline import read_pages
from litmapr.portraits import inherit_portrait
from litmapr.store import StoryStore

DEFAULT_DB = Path("data") / "litmapr.db"

app = typer.Typer()

# File extension -> the reader class that handles it.
# Supporting a new format = write a reader + add one line here.
READERS = {
    ".cbz": CbzReader,
    ".pdf": PdfReader,
}


def get_reader(path: Path) -> Reader:
    # 1. look up the reader class for this file's extension
    reader_class = READERS.get(path.suffix.lower())
    # 2. if there isn't one: print the "Unsupported file type" message and exit
    if reader_class is None:
        supported = ", ".join(READERS)
        print(f"Unsupported file type '{path.suffix}'. Supported: {supported}")
        raise typer.Exit(code=1)
    # 3. otherwise: create the reader with the path and return it
    return reader_class(path)


def make_extractor(model: str) -> TextExtractor:
    """Build the extractor `--ocr` uses. Tests swap this out for a fake."""
    return OllamaVisionExtractor(model=model)


def free_model_memory(model: str) -> None:
    """Ask Ollama to unload a model now. Harmless if it isn't loaded or Ollama is off."""
    try:
        ollama.generate(model=model, keep_alive=0)
    except (ollama.ResponseError, ConnectionError):
        pass


def make_text_llm(model: str) -> TextLLM:
    """Build the LLM `extract` uses. Tests swap this out for a fake."""
    return OllamaTextLLM(model=model)


def make_embedder(model: str) -> Embedder:
    """Build the embedder `merge` and `link` use. Tests swap this out for a fake."""
    return OllamaEmbedder(model=model)


@contextmanager
def friendly_ollama_errors(model: str, after: str = "") -> Iterator[None]:
    """Turn Ollama's exceptions into a clear message and exit code 1."""
    try:
        yield
    except ConnectionError:
        print(
            f"Can't reach Ollama. Is it running? Start the Ollama app and try again.{after}"
        )
        raise typer.Exit(code=1) from None
    except ollama.ResponseError as error:
        missing = re.search(r"model '([^']+)' not found", str(error.error))
        if missing:
            name = missing.group(1)
            print(f"The Ollama model '{name}' isn't installed. Get it with")
            print(f"  ollama pull {name}")
            print(f"then run the command again.{after}")
            raise typer.Exit(code=1) from None
        print(f"Ollama error with model '{model}': {error.error}{after}")
        if "timed out" in str(error.error).lower():
            print("The model took too long to load. Give Ollama more time with")
            print('  setx OLLAMA_LOAD_TIMEOUT "15m"')
            print("then quit and restart Ollama, and run the command again.")
        raise typer.Exit(code=1) from None


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    ocr: Annotated[
        bool, typer.Option(help="Read page images with a local vision model.")
    ] = False,
    model: Annotated[
        str, typer.Option(help="Ollama vision model for --ocr.")
    ] = DEFAULT_MODEL,
    out: Annotated[
        Path | None,
        typer.Option(help="Where to write the JSON. Default: data/maps/<name>.json"),
    ] = None,
    contents: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Omnibus: only read pages that belong to issues.",
        ),
    ] = None,
) -> None:
    """Read every page of a book or comic and save the words to a JSON file."""
    wanted = None
    if contents is not None:
        try:
            wanted = page_filter(parse_contents(contents.read_text(encoding="utf-8")))
        except ContentsError as error:
            print(f"Contents file problem: {contents.name}: {error}")
            raise typer.Exit(code=1) from None
    reader = get_reader(path)
    source_hash = file_hash(path)
    output = out or default_output_path(path)

    if ocr:
        extractor = make_extractor(model)
        cache = PageCache(source_hash, model)
        print(f"reading {path.name} with {model} (cache: {cache.path})")
        results = []
        with friendly_ollama_errors(
            model, after="\nPages read so far are cached; re-run to continue."
        ):
            start = time.perf_counter()
            for page, source in read_pages(reader.pages(), extractor, cache, wanted):
                results.append((page, source))
                if source == "skipped":
                    continue  # outside every issue in the contents file
                elapsed = time.perf_counter() - start
                note = "failed, skipped for now" if source == "failed" else source
                print(f"  page {page.number}: {note} ({elapsed:.1f}s)")
                start = time.perf_counter()
        skipped = sum(1 for _, source in results if source == "skipped")
        if skipped:
            print(f"{skipped} page(s) outside the listed issues were not read.")
        failed = [page.number for page, source in results if source == "failed"]
        if failed:
            shown = ", ".join(map(str, failed[:10])) + (
                "..." if len(failed) > 10 else ""
            )
            print(f"{len(failed)} page(s) couldn't be read ({shown}). Run the same")
            print("command again to retry them; every other page comes from the cache.")
    else:
        results = [
            (page, "text_layer" if page.text else "none") for page in reader.pages()
        ]

    write_pages_json(
        output, path, source_hash, model if ocr else None, results, reader.metadata()
    )

    with_text = sum(1 for page, _ in results if page.text)
    print(f"found {len(results)} pages in {path} ({with_text} with text)")
    print(f"saved to {output}")
    if not ocr and with_text < len(results):
        print("Some pages have no text layer; add --ocr to read their images.")


def _works_for(
    story: Story,
    store: StoryStore,
    source_hash: str,
    sections: list[Section],
    whole_file: bool,
    source_path: str | None = None,
) -> list[Work]:
    """One Work per section of the file, reusing works from an earlier run.

    Re-extracting replaces old results instead of duplicating them. Works
    for this file that no longer match a section (say, the contents file
    changed) are deleted, along with their mentions and relations.
    """
    old = {w.first_page: w for w in story.works if w.source_hash == source_hash}
    works = []
    for section in sections:
        key = None if whole_file else section.first_page
        work = old.pop(key, None)
        if work is None:
            work = Work(
                series=section.series, issue=section.issue, source_hash=source_hash
            )
            story.works.append(work)
        else:
            store.clear_work(work.id)
            story.mentions = [m for m in story.mentions if m.work_id != work.id]
            story.relations = [r for r in story.relations if r.work_id != work.id]
        work.series, work.issue = section.series, section.issue
        work.year = section.year if section.year is not None else work.year
        work.month = section.month if section.month is not None else work.month
        work.source_path = source_path or work.source_path
        work.first_page = None if whole_file else section.first_page
        work.last_page = None if whole_file else section.last_page
        works.append(work)

    for leftover in old.values():
        store.delete_work(leftover.id)
        story.works.remove(leftover)
        story.mentions = [m for m in story.mentions if m.work_id != leftover.id]
        story.relations = [r for r in story.relations if r.work_id != leftover.id]
        remaining = {m.id for m in story.mentions}
        story.callbacks = [
            c
            for c in story.callbacks
            if {c.from_mention_id, c.to_mention_id} <= remaining
        ]
    return works


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    page: Annotated[int, typer.Option(min=1, help="Page number, starting at 1.")] = 1,
    save_image: Annotated[
        Path | None, typer.Option(help="Save the page image to this file.")
    ] = None,
    ocr: Annotated[
        bool, typer.Option(help="Read the page image with a local vision model.")
    ] = False,
    model: Annotated[
        str, typer.Option(help="Ollama vision model for --ocr.")
    ] = DEFAULT_MODEL,
) -> None:
    """Show what a reader extracted from one page of a file."""
    reader = get_reader(path)

    # Skip ahead to the requested page without processing the ones after it.
    # Users count from 1, Python counts from 0, so page 10 is index 9.
    found = next(islice(reader.pages(), page - 1, None), None)
    if found is None:
        print(f"Page {page} doesn't exist in {path.name}.")
        raise typer.Exit(code=1)

    if ocr:
        with friendly_ollama_errors(model):
            found = make_extractor(model).extract(found)

    print(f"--- page {found.number} of {path.name} ---")
    if found.text:
        print(found.text)
    elif ocr:
        print("(no text on this page)")
    else:
        print("(no text layer; add --ocr to read the image)")

    if found.scene:
        print("\n--- scene ---")
        print(found.scene)

    if save_image is not None:
        save_image.parent.mkdir(parents=True, exist_ok=True)
        save_image.write_bytes(found.image)
        print(f"saved page image to {save_image}")


@app.command()
def extract(
    pages_json: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A file from `ingest`.")
    ],
    series: Annotated[
        str | None,
        typer.Option(help="Series name. Default: from the file, else guessed."),
    ] = None,
    issue: Annotated[
        str | None,
        typer.Option(help="Issue number. Default: from the file, else guessed."),
    ] = None,
    year: Annotated[int | None, typer.Option(help="Publication year.")] = None,
    month: Annotated[
        int | None, typer.Option(min=1, max=12, help="Publication month.")
    ] = None,
    model: Annotated[str, typer.Option(help="Ollama text model.")] = DEFAULT_TEXT_MODEL,
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
    pages_per_chunk: Annotated[
        int, typer.Option(min=1, help="Pages the model reads at once.")
    ] = 4,
    contents: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Omnibus: where each issue starts (see `contents`).",
        ),
    ] = None,
) -> None:
    """Find characters, places, devices and relationships in an ingested file."""
    document = json.loads(pages_json.read_text(encoding="utf-8"))
    source_hash = document["source_hash"]
    all_pages = document["pages"]

    if contents is None:
        # One issue. Where each detail comes from, best first: your option,
        # then the file's own ComicInfo.xml, then a guess from the filename.
        meta = document.get("metadata", {})
        guessed_series, guessed_issue = guess_series_and_issue(document["source_file"])
        sections = [
            Section(
                first_page=1,
                series=series or meta.get("series") or guessed_series,
                issue=issue or meta.get("issue") or guessed_issue,
                year=year if year is not None else meta.get("year"),
                month=month if month is not None else meta.get("month"),
                last_page=len(all_pages),
            )
        ]
        whole_file = True
    else:
        try:
            sections = issues(read_contents(contents, len(all_pages)))
        except ContentsError as error:
            print(f"Contents file problem: {error}")
            raise typer.Exit(code=1) from None
        whole_file = False
        print(f"{document['source_file']}: {len(sections)} issues from {contents.name}")

    with StoryStore(db) as store:
        story = store.load()
        works = _works_for(
            story, store, source_hash, sections, whole_file, document.get("source_path")
        )

        llm = make_text_llm(model)
        cache = ResponseCache(f"extract-{model}")
        totals = {"new_entities": 0, "mentions": 0, "relations": 0}
        problems: list[str] = []

        with friendly_ollama_errors(
            model, after="\nChunks read so far are cached; re-run to continue."
        ):
            for work, section in zip(works, sections, strict=True):
                print(f"extracting {work.series} #{work.issue} with {model}")
                pages = [
                    p
                    for p in all_pages
                    if section.first_page <= p["number"] <= section.last_page
                ]
                start = time.perf_counter()
                for numbers, report, from_cache in extract_work(
                    story, work, pages, llm, cache, pages_per_chunk
                ):
                    elapsed = time.perf_counter() - start
                    where = "cache" if from_cache else f"model, {elapsed:.0f}s"
                    print(
                        f"  pages {work.local_page(numbers[0])}-{work.local_page(numbers[-1])}: "
                        f"+{report.new_entities} entities, {report.mentions} mentions, "
                        f"{report.relations} relations ({where})"
                    )
                    for key in totals:
                        totals[key] += getattr(report, key)
                    problems.extend(report.problems)
                    start = time.perf_counter()

        broken = story.broken_references()
        if broken:
            print("Not saved: the story has broken references:")
            for problem in broken[:5]:
                print(f"  {problem}")
            raise typer.Exit(code=1)
        store.save(story)
        removed = store.remove_orphans()
        if removed:
            print(f"removed {removed} entries that never appear in the story")

    print(
        f"done: {totals['new_entities']} new entities, {totals['mentions']} mentions, "
        f"{totals['relations']} relations saved to {db}"
    )
    if problems:
        print(f"{len(problems)} items skipped, for example:")
        for problem in problems[:5]:
            print(f"  - {problem}")


@app.command()
def entities(
    kind: Annotated[
        EntityKind | None, typer.Option(help="Only show one kind, e.g. character.")
    ] = None,
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """List everything the map knows about, most-mentioned first."""
    with StoryStore(db) as store:
        story = store.load()
    counts = Counter(m.entity_id for m in story.mentions)
    found = [e for e in story.entities if kind is None or e.kind == kind]
    if not found:
        print("No entities yet. Run `litmapr extract` first.")
        raise typer.Exit(code=1)
    for entity in sorted(found, key=lambda e: (-counts[e.id], e.name.lower())):
        line = f"{counts[entity.id]:4}  {entity.name} ({entity.kind})"
        if entity.aliases:
            line += f"  aka {', '.join(entity.aliases)}"
        print(line)
    print(f"\n{len(found)} entities")


@app.command()
def show(
    name: Annotated[str, typer.Argument(help='A name or alias, e.g. "Red Hulk".')],
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Show everything the map knows about a character, place, device..."""
    with StoryStore(db) as store:
        story: Story = store.load()
        matches = store.find_entities(name)
        if not matches:
            print(f"Nothing called '{name}' yet.")
            raise typer.Exit(code=1)

        works = {w.id: w for w in story.works}
        names = {e.id: e.name for e in story.entities}
        for entity in matches:
            print(f"{entity.name} ({entity.kind})")
            if entity.aliases:
                print(f"  also called: {', '.join(entity.aliases)}")
            if entity.description:
                print(f"  {entity.description}")

            mentions = store.mentions_in_order(entity.id)
            if mentions:
                first = mentions[0]
                first_work = works[first.work_id]
                print(
                    f"  first appearance: {first_work.series} #{first_work.issue}, "
                    f"page {first_work.local_page(first.page)}"
                )
                print("  appearances:")
                for m in mentions:
                    w = works[m.work_id]
                    line = f"    {page_label(w, m.page)}: {m.summary}"
                    if m.quote:
                        line += f' "{m.quote}"'
                    print(line)

            linked = [
                r for r in story.relations if entity.id in (r.source_id, r.target_id)
            ]
            if linked:
                print("  relations:")
                for r in linked:
                    w = works[r.work_id]
                    print(
                        f"    {names[r.source_id]} -{r.kind}-> {names[r.target_id]}: "
                        f"{r.label} ({page_label(w, r.page)}, sentiment {r.sentiment:+.1f})"
                    )
            mention_ids = {m.id for m in mentions}
            by_id = {m.id: m for m in story.mentions}
            linked_callbacks = [
                c
                for c in story.callbacks
                if c.from_mention_id in mention_ids or c.to_mention_id in mention_ids
            ]
            if linked_callbacks:
                print("  callbacks:")
                for c in linked_callbacks:
                    later, earlier = by_id[c.from_mention_id], by_id[c.to_mention_id]
                    lw, ew = works[later.work_id], works[earlier.work_id]
                    print(
                        f"    {page_label(lw, later.page)} -> "
                        f"{page_label(ew, earlier.page)}: {c.reason}"
                    )
            print()


@app.command()
def merge(
    threshold: Annotated[
        float,
        typer.Option(
            min=0, max=1, help="How alike two entries must look to be checked."
        ),
    ] = resolve.DEFAULT_THRESHOLD,
    dry_run: Annotated[
        bool, typer.Option(help="Show decisions without merging.")
    ] = False,
    model: Annotated[str, typer.Option(help="Ollama text model.")] = DEFAULT_TEXT_MODEL,
    embed_model: Annotated[
        str, typer.Option(help="Ollama embedding model.")
    ] = DEFAULT_EMBED_MODEL,
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Find entities that are really the same one, and merge them."""
    with StoryStore(db) as store, friendly_ollama_errors(model):
        story = store.load()
        llm = make_text_llm(model)
        print(f"reading the cast list for codenames with {model}")
        roster = resolve.find_roster_candidates(llm, story)
        candidates = resolve.combine(
            roster,
            resolve.find_duplicate_candidates(
                story, make_embedder(embed_model), threshold
            ),
        )
        if not candidates:
            print("No likely duplicates found.")
            return
        print(f"checking {len(candidates)} possible duplicates with {model}")
        gone: set[str] = set()
        merged = 0
        for candidate in candidates:
            if candidate.a.id in gone or candidate.b.id in gone:
                continue  # one side was already merged into something else
            verdict = resolve.confirm_same(llm, story, candidate)
            pair = f"{candidate.a.name} / {candidate.b.name} ({candidate.why})"
            if not verdict.same:
                print(f"  different: {pair}: {verdict.reason}")
                continue
            keep, drop = resolve.pick_keeper(story, candidate.a, candidate.b)
            print(f"  same: {pair}: {verdict.reason}")
            if not dry_run:
                store.merge_entities(keep.id, drop.id)
                inherit_portrait(keep.id, drop.id)
                gone.add(drop.id)
                merged += 1
                print(f"    merged '{drop.name}' into '{keep.name}'")
    print(f"done: {merged} merged" + (" (dry run, nothing changed)" if dry_run else ""))


@app.command("merge-entities")
def merge_entities(
    keep: Annotated[str, typer.Argument(help="Name of the entity to keep.")],
    drop: Annotated[str, typer.Argument(help="Name of the entity to fold into it.")],
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Merge two entities by hand, e.g. when `merge` missed one."""
    with StoryStore(db) as store:
        found = []
        for name in (keep, drop):
            matches = store.find_entities(name)
            if len(matches) != 1:
                problem = "nothing" if not matches else f"{len(matches)} entities"
                print(f"'{name}' matches {problem}; use a more exact name.")
                raise typer.Exit(code=1)
            found.append(matches[0])
        if found[0].id == found[1].id:
            print("Those are already the same entity.")
            raise typer.Exit(code=1)
        store.merge_entities(found[0].id, found[1].id)
    inherit_portrait(found[0].id, found[1].id)
    print(f"merged '{found[1].name}' into '{found[0].name}'")


@app.command()
def link(
    threshold: Annotated[
        float,
        typer.Option(
            min=0, max=1, help="How related two moments must look to be checked."
        ),
    ] = callback_finder.DEFAULT_THRESHOLD,
    max_checks: Annotated[
        int, typer.Option(min=1, help="Most pairs to ask the model about in one run.")
    ] = 50,
    model: Annotated[str, typer.Option(help="Ollama text model.")] = DEFAULT_TEXT_MODEL,
    embed_model: Annotated[
        str, typer.Option(help="Ollama embedding model.")
    ] = DEFAULT_EMBED_MODEL,
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Find later moments that call back to earlier ones, across issues."""
    decisions = ResponseCache(f"callbacks-{model}")
    with StoryStore(db) as store, friendly_ollama_errors(model):
        story = store.load()
        candidates = callback_finder.find_callback_candidates(
            story, make_embedder(embed_model), threshold
        )
        unchecked = [
            c
            for c in candidates
            if decisions.get(f"{c.later.id}>{c.earlier.id}") is None
        ][:max_checks]
        if not unchecked:
            print("No new possible callbacks. (Callbacks need at least two issues.)")
            return
        print(f"checking {len(unchecked)} possible callbacks with {model}")
        llm = make_text_llm(model)
        found = []
        for candidate in unchecked:
            verdict = callback_finder.confirm_callback(llm, story, candidate)
            # Remember every decision, yes or no, so the next run skips this pair.
            decisions.put(
                f"{candidate.later.id}>{candidate.earlier.id}",
                verdict.model_dump_json(),
            )
            if verdict.is_callback:
                found.append(
                    Callback(
                        from_mention_id=candidate.later.id,
                        to_mention_id=candidate.earlier.id,
                        reason=verdict.reason,
                    )
                )
                print(f"  callback: {verdict.reason}")
        store.add_callbacks(found)
    left = len(candidates) - len(unchecked)
    print(f"done: {len(found)} callbacks saved")
    if left > 0:
        print(f"{left} pairs not checked yet; run `litmapr link` again to continue.")


@app.command()
def add(
    path: Annotated[
        Path, typer.Argument(exists=True, help="A comic or book, or a folder of them.")
    ],
    series: Annotated[
        str | None,
        typer.Option(help="Series name. Default: from the file, else guessed."),
    ] = None,
    year: Annotated[int | None, typer.Option(help="Publication year.")] = None,
    month: Annotated[
        int | None, typer.Option(min=1, max=12, help="Publication month.")
    ] = None,
    vision_model: Annotated[
        str, typer.Option(help="Ollama model that reads pages.")
    ] = (DEFAULT_MODEL),
    model: Annotated[str, typer.Option(help="Ollama model that finds entities.")] = (
        DEFAULT_TEXT_MODEL
    ),
    contents: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Omnibus: where each issue starts. Default: <name>.contents.txt beside it.",
        ),
    ] = None,
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """One step: read a file (or a whole folder) and add it to the map."""
    if path.is_dir() and contents is not None:
        print("--contents describes one file. In a folder, put each omnibus's")
        print("contents file beside it as <name>.contents.txt.")
        raise typer.Exit(code=1)
    if path.is_dir() and (year is not None or month is not None):
        print("--year and --month describe one issue. For a folder, litmapr uses each")
        print(
            "file's own date (ComicInfo.xml); fix any that are missing with `extract`."
        )
        raise typer.Exit(code=1)
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in READERS)
        if not files:
            print(f"No {', '.join(READERS)} files in {path}.")
            raise typer.Exit(code=1)
    else:
        files = [path]

    for number, file in enumerate(files, start=1):
        print(f"\n[{number}/{len(files)}] {file.name}")
        # Commands are plain functions, so `add` can simply call them in order.
        beside = contents_path_for(file)
        file_contents = contents or (beside if beside.exists() else None)
        if file_contents is not None:
            print(f"omnibus: using {file_contents.name}")
        ingest(
            path=file, ocr=True, model=vision_model, out=None, contents=file_contents
        )
        # Free the GPU before the (much bigger) text model loads: Ollama would
        # otherwise keep the vision model in memory for five more minutes.
        free_model_memory(vision_model)
        extract(
            pages_json=default_output_path(file),
            series=series,
            issue=None,
            year=year,
            month=month,
            model=model,
            db=db,
            pages_per_chunk=4,
            contents=file_contents,
        )
    print("\nNext: `litmapr merge` to tidy duplicates, `litmapr link` for callbacks,")
    print("then `litmapr serve` to explore.")


def make_locator(model: str):
    """Build the character finder `portraits` uses. Tests swap this out for a fake."""
    from litmapr.portraits import OllamaLocator

    return OllamaLocator(model=model)


@app.command()
def portraits(
    name: Annotated[
        str | None, typer.Option(help="Only this entry (name or alias).")
    ] = None,
    kind: Annotated[
        list[str] | None,
        typer.Option(
            help="Which kinds get pictures (repeat it): character, device, "
            "location, faction, event. Default: all of them."
        ),
    ] = None,
    top: Annotated[
        int, typer.Option(min=1, help="How many of the most-seen entries per kind.")
    ] = 30,
    image: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="With --name: use this picture instead."
        ),
    ] = None,
    page: Annotated[
        int | None,
        typer.Option(
            min=1, help="With --name: look on this page (as the issue numbers it)."
        ),
    ] = None,
    issue: Annotated[
        str | None,
        typer.Option(help="With --page: which issue, when there's more than one."),
    ] = None,
    remove: Annotated[
        bool, typer.Option(help="With --name: delete a wrong picture.")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Redo pictures that already exist.")
    ] = False,
    model: Annotated[str, typer.Option(help="Ollama vision model.")] = DEFAULT_MODEL,
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Cut pictures of characters, objects, places and moments out of the comics."""
    from litmapr import portraits as portrait_maker

    with StoryStore(db) as store:
        story = store.load()

    def only_one(what: str) -> Entity:
        if name is None:
            print(f"{what} needs --name, so litmapr knows who it's for.")
            raise typer.Exit(code=1)
        matches = story.find_entities(name)
        if len(matches) != 1:
            print(f"'{name}' matches {len(matches)} entities; use a more exact name.")
            raise typer.Exit(code=1)
        return matches[0]

    if remove:
        entity = only_one("--remove")
        if portrait_maker.remove_portrait(entity.id):
            print(f"removed the picture of {entity.name}")
        else:
            print(f"{entity.name} has no picture.")
        return

    if image is not None:
        entity = only_one("--image")
        saved = portrait_maker.portrait_from_file(entity, image)
        print(f"saved {saved}")
        return

    if page is not None:
        entity = only_one("--page")
        works = [w for w in story.works if issue is None or w.issue == issue]
        if len(works) != 1:
            numbers = ", ".join(f"#{w.issue}" for w in story.works)
            print(f"Which issue? Add --issue (one of: {numbers}).")
            raise typer.Exit(code=1)
        with friendly_ollama_errors(model):
            saved, note = portrait_maker.portrait_from_page(
                entity, works[0], page, make_locator(model)
            )
        print(f"  {'saved' if saved else 'skipped'}: {entity.name}: {note}")
        if saved is None:
            print(
                "Try a page where they're drawn big, or supply a picture with --image."
            )
            raise typer.Exit(code=1)
        return

    kinds = (
        [EntityKind(k) for k in kind] if kind else list(portrait_maker.PICTURE_KINDS)
    )
    counts = Counter(m.entity_id for m in story.mentions)
    if name is not None:
        chosen = story.find_entities(name)
        if not chosen:
            print(f"Nothing called '{name}'. Try `litmapr entities`.")
            raise typer.Exit(code=1)
    else:
        chosen = []
        for wanted in kinds:
            of_kind = [e for e in story.entities if e.kind == wanted]
            chosen += sorted(of_kind, key=lambda e: -counts[e.id])[:top]
    todo = [
        e for e in chosen if force or not portrait_maker.portrait_path(e.id).exists()
    ]
    if not todo:
        print("Every chosen entry already has a picture. Use --force to redo them.")
        return

    locator = make_locator(model)
    made = 0
    with friendly_ollama_errors(model):
        for entity in todo:
            saved, note = portrait_maker.make_portrait(story, entity, locator)
            mark = "saved" if saved else "skipped"
            print(f"  {mark}: {entity.name} ({entity.kind}): {note}")
            made += saved is not None
    print(
        f"done: {made} of {len(todo)} pictures in {portrait_maker.DEFAULT_PORTRAIT_DIR}"
    )
    if made < len(todo):
        print(
            "Missing ones: point at a page where they're drawn big with\n"
            '  litmapr portraits --name "Galactus" --issue 574 --page 12\n'
            "or use your own picture, here or with the dossier's Replace button."
        )


@app.command("contents")
def contents_command(
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="An omnibus.")
    ],
    force: Annotated[
        bool, typer.Option(help="Replace an existing contents file.")
    ] = False,
) -> None:
    """Draft a contents file for an omnibus from its bookmarks, to check and edit."""
    target = contents_path_for(path)
    if target.exists() and not force:
        print(f"{target} already exists. Edit it, or use --force to start over.")
        raise typer.Exit(code=1)
    draft = suggest_contents(path)
    target.write_text(draft, encoding="utf-8")
    issues = [line for line in draft.splitlines() if line and not line.startswith("#")]
    print(f"wrote {target} with {len(issues)} issues found in the bookmarks")
    print("Check it against the book and add years and months, then run:")
    print(f'  litmapr add "{path}"')


@app.command("export")
def export_story(
    out: Annotated[Path, typer.Argument(help="JSON file to write.")],
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Save the whole map as one JSON file, to share or back up. No comic pages are included."""
    with StoryStore(db) as store:
        story = store.load()
    for work in story.works:
        work.source_path = None  # your folders are nobody else's business
    story.save(out)
    print(
        f"exported {len(story.works)} works and {len(story.entities)} entities to {out}"
    )


@app.command("import")
def import_story(
    source: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="From `export`.")
    ],
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Load a map someone exported into your database."""
    story = Story.load(source)
    problems = story.broken_references()
    if problems:
        print(f"Not imported: {source} has {len(problems)} broken references.")
        raise typer.Exit(code=1)
    with StoryStore(db) as store:
        store.save(story)
    print(
        f"imported {len(story.works)} works and {len(story.entities)} entities into {db}"
    )


def resolve_upto(story: Story, up_to: str | None) -> int | None:
    """Turn --up-to into a reading position. Accepts an issue label
    ("Avengers #5") or a number counted from 1 in reading order."""
    if up_to is None:
        return None
    ordered = story.reading_order()
    if up_to.isdigit():
        position = int(up_to) - 1
        if 0 <= position < len(ordered):
            return position
    else:
        wanted = up_to.lower().replace(" ", "")
        for position, work in enumerate(ordered):
            if f"{work.series} #{work.issue}".lower().replace(" ", "") == wanted:
                return position
    print(f"'{up_to}' isn't in the reading order. Try `litmapr works`.")
    raise typer.Exit(code=1)


@app.command()
def works(
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """List every ingested issue in reading order."""
    with StoryStore(db) as store:
        ordered = store.load().reading_order()
    if not ordered:
        print("No works yet. Run `litmapr add` first.")
        raise typer.Exit(code=1)
    for position, work in enumerate(ordered, start=1):
        when = (
            f"{work.year}-{work.month:02d}"
            if work.year and work.month
            else "date unknown"
        )
        print(f"{position:3}. {work.series} #{work.issue}  ({when})")


@app.command()
def narrate(
    name: Annotated[str, typer.Argument(help="A name or alias.")],
    up_to: Annotated[
        str | None,
        typer.Option(
            help='Spoiler limit: "Avengers #5", or a number from `litmapr works`.'
        ),
    ] = None,
    style: Annotated[
        str, typer.Option(help="How the narrator sounds: storyteller or briefing.")
    ] = narrator.DEFAULT_STYLE,
    model: Annotated[
        str, typer.Option(help="Ollama text model for the narrator.")
    ] = DEFAULT_NARRATOR_MODEL,
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Have the narrator explain a character, place or idea, without spoilers."""
    if style not in narrator.STYLES:
        print(f"Unknown style '{style}'. Pick one of: {', '.join(narrator.STYLES)}.")
        raise typer.Exit(code=1)
    with StoryStore(db) as store:
        story = store.load()
        matches = store.find_entities(name)
    if len(matches) != 1:
        problem = "Nothing" if not matches else f"{len(matches)} entities"
        print(f"{problem} called '{name}'. Try `litmapr entities`.")
        raise typer.Exit(code=1)
    upto = resolve_upto(story, up_to)
    with friendly_ollama_errors(model):
        for piece in narrator.narrate_stream(
            make_text_llm(model),
            story,
            matches[0].id,
            upto,
            ResponseCache("narration"),
            style,
        ):
            print(piece, end="", flush=True)
    print()


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="127.0.0.1 = only this computer.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port for the web page.")] = 8000,
    narrator_on: Annotated[
        bool,
        typer.Option("--narrator/--no-narrator", help="Allow the narrator button."),
    ] = True,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the page in your browser.")
    ] = True,
    portraits_on: Annotated[
        bool,
        typer.Option("--portraits/--no-portraits", help="Show character portraits."),
    ] = True,
    model: Annotated[str, typer.Option(help="Ollama text model for the narrator.")] = (
        DEFAULT_NARRATOR_MODEL
    ),
    db: Annotated[Path, typer.Option(help="The story database.")] = DEFAULT_DB,
) -> None:
    """Start the 3D explorer in your browser."""
    import threading
    import webbrowser

    import uvicorn

    from litmapr.web.api import create_app

    if not db.exists():
        print(f"No map at {db} yet. Run `litmapr add yourcomic.cbz` first.")
        raise typer.Exit(code=1)
    llm_factory = (lambda: make_text_llm(model)) if narrator_on else None
    url = f"http://{host}:{port}"
    print(f"litmapr explorer running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    from litmapr.portraits import DEFAULT_PORTRAIT_DIR

    app_ = create_app(db, llm_factory, DEFAULT_PORTRAIT_DIR if portraits_on else None)
    uvicorn.run(app_, host=host, port=port, log_level="warning")
