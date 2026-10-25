"""Save and load a Story in a SQLite database.

A database answers questions like "every mention of Red Hulk, in reading
order" without loading the whole story into memory. SQLite is a database
in a single file and ships with Python, so there's nothing to install.
"""

import sqlite3
from pathlib import Path
from typing import Self

from litmapr.model import Callback, Entity, Mention, Relation, Story, Work

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS works (
    id             TEXT PRIMARY KEY,
    series         TEXT NOT NULL,
    issue          TEXT NOT NULL,
    issue_number   REAL NOT NULL,     -- '5' -> 5.0, kept so SQL can sort
    title          TEXT,
    year           INTEGER,
    month          INTEGER,
    source_hash    TEXT,
    order_override REAL
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT ''
);

-- One row per alias, so "who goes by Captain America?" is a simple lookup.
CREATE TABLE IF NOT EXISTS aliases (
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL,
    position  INTEGER NOT NULL,       -- keeps aliases in their original order
    PRIMARY KEY (entity_id, alias)
);

CREATE TABLE IF NOT EXISTS mentions (
    id        TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    work_id   TEXT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    page      INTEGER NOT NULL CHECK (page >= 1),
    summary   TEXT NOT NULL,
    quote     TEXT
);

CREATE TABLE IF NOT EXISTS relations (
    id        TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    kind      TEXT NOT NULL,
    label     TEXT NOT NULL,
    sentiment REAL NOT NULL CHECK (sentiment BETWEEN -1 AND 1),
    work_id   TEXT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    page      INTEGER NOT NULL CHECK (page >= 1)
);

CREATE TABLE IF NOT EXISTS callbacks (
    id              TEXT PRIMARY KEY,
    from_mention_id TEXT NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
    to_mention_id   TEXT NOT NULL REFERENCES mentions(id) ON DELETE CASCADE,
    reason          TEXT NOT NULL
);

-- Indexes: like a book's index, they let SQLite jump straight to rows
-- instead of reading the whole table.
CREATE INDEX IF NOT EXISTS mentions_by_entity ON mentions(entity_id);
CREATE INDEX IF NOT EXISTS relations_by_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS relations_by_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS aliases_by_name ON aliases(alias COLLATE NOCASE);
"""

# Same rule as Work.reading_key, written in SQL: date, then series, then issue.
READING_ORDER = """
    COALESCE(w.order_override, w.year * 100 + COALESCE(w.month, 0), 1e18),
    w.series COLLATE NOCASE,
    w.issue_number
"""


class StoryStore:
    """A story kept in a SQLite file. Use it with `with`:

    with StoryStore("data/litmapr.db") as store:
        store.save(story)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row  # rows act like dicts: row["name"]
        # SQLite only enforces REFERENCES when this is switched on.
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    # --- writing ---------------------------------------------------------

    def save(self, story: Story) -> None:
        """Insert or update everything in the story, all-or-nothing."""
        # `with self.db` is a transaction: if anything fails, nothing is saved.
        with self.db:
            for work in story.works:
                self._upsert(
                    "works",
                    {**work.model_dump(), "issue_number": work.issue_number},
                )
            for entity in story.entities:
                data = entity.model_dump(exclude={"aliases"})
                self._upsert("entities", data)
                self.db.execute("DELETE FROM aliases WHERE entity_id = ?", (entity.id,))
                self.db.executemany(
                    "INSERT INTO aliases (entity_id, alias, position) VALUES (?, ?, ?)",
                    [(entity.id, alias, i) for i, alias in enumerate(entity.aliases)],
                )
            for mention in story.mentions:
                self._upsert("mentions", mention.model_dump())
            for relation in story.relations:
                self._upsert("relations", relation.model_dump())
            for callback in story.callbacks:
                self._upsert("callbacks", callback.model_dump())

    def _upsert(self, table: str, row: dict) -> None:
        """Insert a row, or update it if that id already exists.

        Not INSERT OR REPLACE: that deletes the old row first, and the
        ON DELETE CASCADE would silently wipe its mentions and relations.
        """
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        updates = ", ".join(f"{name} = excluded.{name}" for name in row if name != "id")
        self.db.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            row,
        )

    # --- reading ---------------------------------------------------------

    def load(self) -> Story:
        """Load the whole story back into Pydantic models (validated again)."""
        aliases: dict[str, list[str]] = {}
        for row in self.db.execute(
            "SELECT entity_id, alias FROM aliases ORDER BY entity_id, position"
        ):
            aliases.setdefault(row["entity_id"], []).append(row["alias"])

        return Story(
            works=[
                Work.model_validate(self._work_fields(row))
                for row in self._all("works")
            ],
            entities=[
                Entity.model_validate(
                    {**dict(row), "aliases": aliases.get(row["id"], [])}
                )
                for row in self._all("entities")
            ],
            mentions=[
                Mention.model_validate(dict(row)) for row in self._all("mentions")
            ],
            relations=[
                Relation.model_validate(dict(row)) for row in self._all("relations")
            ],
            callbacks=[
                Callback.model_validate(dict(row)) for row in self._all("callbacks")
            ],
        )

    def find_entities(self, name: str) -> list[Entity]:
        """Every entity whose name or alias matches, ignoring upper/lower case."""
        rows = self.db.execute(
            """
            SELECT DISTINCT e.id FROM entities e
            LEFT JOIN aliases a ON a.entity_id = e.id
            WHERE e.name = :name COLLATE NOCASE OR a.alias = :name COLLATE NOCASE
            """,
            {"name": name},
        ).fetchall()
        return [self.get_entity(row["id"]) for row in rows]

    def get_entity(self, entity_id: str) -> Entity:
        row = self.db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            raise KeyError(entity_id)
        names = [
            r["alias"]
            for r in self.db.execute(
                "SELECT alias FROM aliases WHERE entity_id = ? ORDER BY position",
                (entity_id,),
            )
        ]
        return Entity.model_validate({**dict(row), "aliases": names})

    def mentions_in_order(self, entity_id: str) -> list[Mention]:
        """Every mention of an entity in reading order, sorted by the database."""
        rows = self.db.execute(
            f"""
            SELECT m.* FROM mentions m
            JOIN works w ON w.id = m.work_id
            WHERE m.entity_id = ?
            ORDER BY {READING_ORDER}, m.page
            """,
            (entity_id,),
        )
        return [Mention.model_validate(dict(row)) for row in rows]

    def first_appearance(self, entity_id: str) -> Mention | None:
        ordered = self.mentions_in_order(entity_id)
        return ordered[0] if ordered else None

    def _all(self, table: str) -> list[sqlite3.Row]:
        return self.db.execute(f"SELECT * FROM {table}").fetchall()

    @staticmethod
    def _work_fields(row: sqlite3.Row) -> dict:
        fields = dict(row)
        fields.pop("issue_number")  # worked out from `issue`, not a real field
        return fields
