"""The shape of a story: what litmapr knows, and how it connects.

Phase 2 defines these containers. Phase 3 (LLM extraction) fills them.
Pydantic checks every value on the way in, so a malformed LLM answer
fails loudly here instead of quietly corrupting the map.
"""

import math
import re
import uuid
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


def new_id() -> str:
    """A random ID that never changes, even if names clash later."""
    return uuid.uuid4().hex[:12]


def slugify(text: str) -> str:
    """'Carol Danvers' -> 'carol-danvers'. For humans reading data, not for lookups."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class EntityKind(StrEnum):
    CHARACTER = "character"
    DEVICE = "device"
    CONCEPT = "concept"
    LOCATION = "location"
    FACTION = "faction"
    EVENT = "event"


class RelationKind(StrEnum):
    """Fixed categories the map can color and filter by. The details go in `label`."""

    ALLY = "ally"
    RIVAL = "rival"
    LOVES = "loves"
    FAMILY = "family"
    CREATED = "created"
    FEARS = "fears"
    MEMBER_OF = "member_of"
    OPPOSES = "opposes"
    OTHER = "other"


class Work(BaseModel):
    """One issue or book."""

    id: str = Field(default_factory=new_id)
    series: str
    issue: str  # text, because issues can be "5", "0.1" or "Annual 1"
    title: str | None = None
    year: int | None = None
    month: int | None = Field(default=None, ge=1, le=12)
    source_hash: str | None = None  # links back to the file it came from
    source_path: str | None = None  # where that file was, to revisit pages (portraits)
    # For an omnibus (many issues in one file): the pages of the file this
    # issue covers. None means the work is the whole file.
    first_page: int | None = Field(default=None, ge=1)
    last_page: int | None = Field(default=None, ge=1)
    # Same scale as the automatic order (year * 100 + month), so
    # 202605.5 means "read this in the middle of May 2026's issues".
    order_override: float | None = None

    def local_page(self, page: int) -> int:
        """Turn a page of the file into a page of this issue.

        Omnibus page 612, in an issue that starts on page 601, is page 12.
        """
        return page - self.first_page + 1 if self.first_page else page

    @property
    def issue_number(self) -> float:
        """The first number in the issue text: '5' -> 5.0, 'Annual 1' -> 1.0."""
        match = re.search(r"\d+(\.\d+)?", self.issue)
        return float(match.group()) if match else 0.0

    @property
    def reading_key(self) -> tuple:
        """Sort key for reading order: date first, then series, then issue.

        Works with no date sort last, because we can't place them.
        """
        if self.order_override is not None:
            when = self.order_override
        elif self.year is not None:
            when = self.year * 100 + (self.month or 0)
        else:
            when = math.inf
        return (when, self.series.lower(), self.issue_number)


class Entity(BaseModel):
    """Anything the story tracks: a character, a device, a place, an idea..."""

    id: str = Field(default_factory=new_id)
    slug: str = ""
    name: str = Field(min_length=1)
    kind: EntityKind
    aliases: list[str] = []
    description: str = ""

    @field_validator("aliases")
    @classmethod
    def drop_duplicate_aliases(cls, aliases: list[str]) -> list[str]:
        seen = set()
        unique = []
        for alias in aliases:
            key = alias.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(alias.strip())
        return unique

    def all_names(self) -> set[str]:
        """Every name this entity goes by, lowercased for matching."""
        return {self.name.lower(), *(alias.lower() for alias in self.aliases)}


class Mention(BaseModel):
    """An entity showing up in a work, on a page."""

    id: str = Field(default_factory=new_id)
    entity_id: str
    work_id: str
    page: int = Field(ge=1)
    summary: str
    # The exact words from the page, when there are any. Optional because
    # many appearances are visual only (Carol walking out has no dialogue).
    quote: str | None = None


class Relation(BaseModel):
    """How two entities connect at one point in the story.

    Stored per appearance: Carol and Reed can be allies in one issue
    and rivals three issues later, and both are true.
    """

    id: str = Field(default_factory=new_id)
    source_id: str
    target_id: str
    kind: RelationKind
    label: str  # free text from the LLM: the nuance the fixed `kind` can't hold
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    work_id: str
    page: int = Field(ge=1)

    @model_validator(mode="after")
    def not_with_itself(self) -> "Relation":
        if self.source_id == self.target_id:
            raise ValueError("an entity can't have a relation with itself")
        return self


class Callback(BaseModel):
    """A later mention that points back to an earlier one."""

    id: str = Field(default_factory=new_id)
    from_mention_id: str  # the later mention
    to_mention_id: str  # the earlier one it refers to
    reason: str


class Story(BaseModel):
    """Everything litmapr knows about one reading list (a run, a series...)."""

    works: list[Work] = []
    entities: list[Entity] = []
    mentions: list[Mention] = []
    relations: list[Relation] = []
    callbacks: list[Callback] = []

    def add_entity(self, name: str, kind: EntityKind, **fields) -> Entity:
        """Create an entity with a unique readable slug: a second
        'Captain America' becomes 'captain-america-2'."""
        base = slugify(name) or "entity"
        taken = {entity.slug for entity in self.entities}
        slug, n = base, 2
        while slug in taken:
            slug, n = f"{base}-{n}", n + 1
        entity = Entity(name=name, kind=kind, slug=slug, **fields)
        self.entities.append(entity)
        return entity

    def find_entities(self, name: str) -> list[Entity]:
        """Every entity that goes by this name or alias. Can be more than one!"""
        wanted = name.lower()
        return [entity for entity in self.entities if wanted in entity.all_names()]

    def reading_order(self) -> list[Work]:
        return sorted(self.works, key=lambda work: work.reading_key)

    def mentions_in_order(self, entity_id: str) -> list[Mention]:
        """Every mention of an entity, in reading order (work first, then page)."""
        works = {work.id: work for work in self.works}
        found = [
            m for m in self.mentions if m.entity_id == entity_id and m.work_id in works
        ]
        return sorted(found, key=lambda m: (works[m.work_id].reading_key, m.page))

    def first_appearance(self, entity_id: str) -> Mention | None:
        """The earliest mention of an entity in what has been ingested so far.

        Worked out, not stored: ingest an earlier series later, and the
        first appearance moves back automatically.
        """
        ordered = self.mentions_in_order(entity_id)
        return ordered[0] if ordered else None

    def broken_references(self) -> list[str]:
        """List every ID that points at something that doesn't exist."""
        works = {work.id for work in self.works}
        entities = {entity.id for entity in self.entities}
        mentions = {mention.id for mention in self.mentions}
        problems = []
        for m in self.mentions:
            if m.entity_id not in entities:
                problems.append(f"mention {m.id}: unknown entity {m.entity_id}")
            if m.work_id not in works:
                problems.append(f"mention {m.id}: unknown work {m.work_id}")
        for r in self.relations:
            for end in (r.source_id, r.target_id):
                if end not in entities:
                    problems.append(f"relation {r.id}: unknown entity {end}")
            if r.work_id not in works:
                problems.append(f"relation {r.id}: unknown work {r.work_id}")
        for c in self.callbacks:
            for end in (c.from_mention_id, c.to_mention_id):
                if end not in mentions:
                    problems.append(f"callback {c.id}: unknown mention {end}")
        return problems

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Story":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
