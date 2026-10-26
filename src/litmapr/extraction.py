"""Phase 3: turn page text into entities, mentions and relations.

The LLM reads a few pages at a time, along with the characters already
known, and answers in JSON. Each item is checked by Pydantic on its own,
so one bad relation doesn't throw away a whole chunk of good data.
"""

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from litmapr.model import EntityKind, Mention, Relation, RelationKind, Story, Work

# Bump when the prompt changes, so cached answers from the old prompt aren't reused.
PROMPT_VERSION = 1

MAX_KNOWN_ENTITIES = 80  # keeps the prompt from growing forever
RETRIES = 1  # extra tries when the model's answer isn't valid JSON


# --- What the LLM answers with ---------------------------------------------------
# These use NAMES, not ids: the model can't know our random ids.
# merge_into_story() turns names into real entities afterwards.


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1)
    kind: EntityKind
    aliases: list[str] = []
    description: str = ""


class ExtractedMention(BaseModel):
    entity: str = Field(min_length=1)
    page: int = Field(ge=1)
    summary: str
    quote: str | None = None


class ExtractedRelation(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: RelationKind
    label: str
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    page: int = Field(ge=1)


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity] = []
    mentions: list[ExtractedMention] = []
    relations: list[ExtractedRelation] = []


# --- Building the prompt --------------------------------------------------------


def chunk_pages(pages: list[dict], size: int) -> list[list[dict]]:
    """Split pages into groups of `size`, keeping only pages with words."""
    readable = [page for page in pages if page.get("text") or page.get("scene")]
    return [readable[i : i + size] for i in range(0, len(readable), size)]


def build_prompt(work: Work, pages: list[dict], story: Story) -> str:
    known = story.entities[-MAX_KNOWN_ENTITIES:]
    if known:
        roster = "\n".join(
            f"- {e.name} ({e.kind})"
            + (f", also called: {', '.join(e.aliases)}" if e.aliases else "")
            for e in known
        )
    else:
        roster = "(none yet)"

    page_blocks = []
    for page in pages:
        block = f"--- PAGE {page['number']} ---"
        if page.get("text"):
            block += f"\nTEXT:\n{page['text']}"
        if page.get("scene"):
            block += f"\nSCENE:\n{page['scene']}"
        page_blocks.append(block)

    kinds = ", ".join(kind.value for kind in EntityKind)
    relation_kinds = ", ".join(kind.value for kind in RelationKind)
    return f"""You are building a map of a story's plot. Read these pages from
{work.series} #{work.issue} and list what matters for the plot.

KNOWN ENTITIES (use these exact names when they appear, even under another name):
{roster}

RULES:
- entities: every person, place, object, group, idea or event in these pages that is
  NOT in the known list, even famous ones. Every name you use in mentions or
  relations must be either in the known list or in your entities. kind is one of: {kinds}.
  Put other names the same entity goes by in "aliases".
- mentions: every time an entity appears or is referred to. "page" must be one of
  the page numbers below. "summary" is one short sentence about what happens to or
  with it. "quote" is an exact line from TEXT if there is one, otherwise null.
- relations: how two entities connect on that page. kind is one of: {relation_kinds}.
  "label" says the specific connection in a few words. "sentiment" is -1.0 (hostile)
  to 1.0 (warm).
- Only use what the pages say. Do not invent events, names or quotes.

PAGES:
{chr(10).join(page_blocks)}
"""


# --- Reading the answer ---------------------------------------------------------


def parse_extraction(raw: str) -> tuple[ChunkExtraction, list[str]]:
    """Validate the LLM's JSON item by item. Returns the good items and a
    list of problems with the bad ones."""
    problems: list[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        return ChunkExtraction(), [f"answer was not valid JSON: {error}"]
    if not isinstance(data, dict):
        return ChunkExtraction(), ["answer was not a JSON object"]

    result = ChunkExtraction()
    for key, model, target in (
        ("entities", ExtractedEntity, result.entities),
        ("mentions", ExtractedMention, result.mentions),
        ("relations", ExtractedRelation, result.relations),
    ):
        for item in data.get(key) or []:
            try:
                target.append(model.model_validate(item))
            except ValidationError as error:
                first = error.errors()[0]
                where = ".".join(str(part) for part in first["loc"])
                problems.append(f"skipped one of {key}: {where}: {first['msg']}")
    return result, problems


# --- Putting it into the story --------------------------------------------------


@dataclass
class MergeReport:
    new_entities: int = 0
    mentions: int = 0
    relations: int = 0
    problems: list[str] = field(default_factory=list)


def merge_into_story(
    story: Story, work: Work, extraction: ChunkExtraction, pages: set[int]
) -> MergeReport:
    """Turn names into real entities and add everything to the story.

    Matching is deliberately cautious. Entries are matched by NAME only, never
    by alias: aliases like "Ben" are shared (Ben Grimm, Bentley Wittman), and
    matching on them glued different characters together. Leftover duplicates
    ("Mr. Fantastic" and "Reed Richards") are for `litmapr merge`, which asks
    the model to confirm each pair.
    """
    report = MergeReport()
    taken_names = {e.name.lower() for e in story.entities}
    for found in extraction.entities:
        found.name = clean_name(found.name)
        found.aliases = [clean_name(a) for a in found.aliases]
    for found in extraction.mentions:
        found.entity = clean_name(found.entity)
    for found in extraction.relations:
        found.source, found.target = clean_name(found.source), clean_name(found.target)

    for found in extraction.entities:
        existing = _named(story, found.name)
        # Never give an entity another entity's name as an alias.
        aliases = [
            a.strip()
            for a in found.aliases
            if a.strip() and a.strip().lower() not in taken_names - {found.name.lower()}
        ]
        if existing is None:
            story.add_entity(
                found.name, found.kind, aliases=aliases, description=found.description
            )
            taken_names.add(found.name.lower())
            report.new_entities += 1
        else:
            known_names = existing.all_names()
            for alias in aliases:
                if alias.lower() not in known_names:
                    existing.aliases.append(alias)
                    known_names.add(alias.lower())
            if not existing.description and found.description:
                existing.description = found.description

    def entity_for(name: str):
        """Find who a mention or relation means. A name nobody has yet becomes
        a new character (the model sometimes skips describing famous ones)."""
        entity, ambiguous = _lookup(story, name)
        if entity is None and not ambiguous and looks_like_a_name(name):
            entity = story.add_entity(name.strip(), EntityKind.CHARACTER)
            report.new_entities += 1
            report.problems.append(
                f"'{name.strip()}' wasn't described; added as a character"
            )
        return entity

    for found in extraction.mentions:
        if found.page not in pages:
            report.problems.append(
                f"mention on page {found.page}, which wasn't in this chunk"
            )
            continue
        entity = entity_for(found.entity)
        if entity is None:
            report.problems.append(
                f"skipped a mention of '{found.entity}': unclear who that is"
            )
            continue
        story.mentions.append(
            Mention(
                entity_id=entity.id,
                work_id=work.id,
                page=found.page,
                summary=found.summary,
                quote=found.quote,
            )
        )
        report.mentions += 1

    for found in extraction.relations:
        if found.page not in pages:
            report.problems.append(
                f"relation on page {found.page}, which wasn't in this chunk"
            )
            continue
        source, target = entity_for(found.source), entity_for(found.target)
        if source is None or target is None:
            unclear = found.source if source is None else found.target
            report.problems.append(
                f"skipped a relation with '{unclear}': unclear who that is"
            )
        elif source.id == target.id:
            report.problems.append(f"relation of '{found.source}' with itself")
        else:
            story.relations.append(
                Relation(
                    source_id=source.id,
                    target_id=target.id,
                    kind=found.kind,
                    label=found.label,
                    sentiment=found.sentiment,
                    work_id=work.id,
                    page=found.page,
                )
            )
            report.relations += 1

    return report


_KIND_SUFFIX = re.compile(
    r"\s*\((?:{})\)\s*$".format("|".join(k.value for k in EntityKind)), re.IGNORECASE
)
_SMALL_WORDS = {"of", "the", "von", "van", "de", "da", "del", "la", "le", "and", "in"}


def clean_name(name: str) -> str:
    """Undo a habit of the model: copying the prompt's 'Name (kind)' format.
    'Spider-Man (character)' -> 'Spider-Man'."""
    return _KIND_SUFFIX.sub("", name).strip()


def looks_like_a_name(name: str) -> bool:
    """'Mr. Fantastic' yes; 'character in a black suit' or 'barrier' no.
    Every word must start with a capital letter or a digit, apart from small
    joining words like 'of' or 'von'."""
    words = name.split()
    if not words or not (words[0][0].isupper() or words[0][0].isdigit()):
        return False
    return all(
        w[0].isupper() or w[0].isdigit() or w.lower() in _SMALL_WORDS for w in words
    )


def _named(story: Story, name: str):
    """The one entity whose main name is exactly this (ignoring case), else None."""
    wanted = name.strip().lower()
    matches = [e for e in story.entities if e.name.lower() == wanted]
    return matches[0] if len(matches) == 1 else None


def _lookup(story: Story, name: str) -> tuple:
    """Find the entity a mention or relation names. Returns (entity, ambiguous).

    1. Exactly one entity with that main name ("Reed Richards").
    2. Otherwise exactly one entity with that alias ("Red Hulk").
    3. Otherwise a short form matching exactly one entity as whole words
       ("Ross" -> "Thaddeus Ross").
    If a step finds several, the name is ambiguous: give up rather than guess.
    """
    wanted = name.strip().lower()
    if not wanted:
        return None, False
    named = [e for e in story.entities if e.name.lower() == wanted]
    if len(named) == 1:
        return named[0], False
    if len(named) > 1:
        return None, True
    aliased = [e for e in story.entities if wanted in {a.lower() for a in e.aliases}]
    if len(aliased) == 1:
        return aliased[0], False
    if len(aliased) > 1:
        return None, True
    words = wanted.split()
    partial = [
        e
        for e in story.entities
        if any(_contains_words(known.split(), words) for known in e.all_names())
    ]
    if len(partial) == 1:
        return partial[0], False
    return None, len(partial) > 1


def _resolve(story: Story, name: str, aliases: list[str] | None = None):
    """Kept for callers that want just the entity (or None)."""
    return _lookup(story, name)[0]


def _contains_words(words: list[str], wanted: list[str]) -> bool:
    """True if `wanted` appears as consecutive whole words inside `words`."""
    size = len(wanted)
    return any(words[i : i + size] == wanted for i in range(len(words) - size + 1))


# --- Naming a work from its file ------------------------------------------------


def guess_series_and_issue(filename: str) -> tuple[str, str]:
    """'Avengers_Armageddon005.cbz' -> ('Avengers Armageddon', '5').

    Just a starting guess; the extract command lets you set both yourself.
    """
    stem = re.sub(r"\.[^.]+$", "", filename)
    match = re.search(r"[\s_#-]*(\d+(?:\.\d+)?)\D*$", stem)
    if not match:
        return stem.replace("_", " ").strip(), "1"
    series = stem[: match.start()].replace("_", " ").strip(" -#")
    issue = match.group(1).lstrip("0") or "0"
    if issue.startswith("."):
        issue = "0" + issue
    return series or stem, issue


# --- Running a whole work -------------------------------------------------------


def extract_work(
    story: Story,
    work: Work,
    pages: list[dict],
    llm,
    cache,
    pages_per_chunk: int = 4,
):
    """Read a work chunk by chunk. Yields (page_numbers, report, from_cache)
    after each chunk so the caller can show progress."""
    schema = ChunkExtraction.model_json_schema()
    for chunk in chunk_pages(pages, pages_per_chunk):
        numbers = [page["number"] for page in chunk]
        key = f"v{PROMPT_VERSION}:{work.source_hash}:{','.join(map(str, numbers))}"
        raw = cache.get(key)
        if raw is not None and not _is_json(raw):
            raw = None  # a broken answer cached by an older version: ask again
        from_cache = raw is not None
        if raw is None:
            prompt = build_prompt(work, chunk, story)
            for _attempt in range(1 + RETRIES):
                raw = llm.complete_json(prompt, schema)
                if _is_json(raw):
                    cache.put(key, raw)  # only good answers are worth keeping
                    break
        extraction, problems = parse_extraction(raw)
        report = merge_into_story(story, work, extraction, set(numbers))
        report.problems[:0] = problems
        yield numbers, report, from_cache


def _is_json(raw: str) -> bool:
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True
