"""Phase 4a: find entities that are really the same one.

The extraction LLM sometimes misses that "Mr. Fantastic" is "Reed Richards".
Candidates come from two places:
1. embeddings: similar descriptions end up close together, so pairs above
   a similarity threshold become candidates. Good at "Reed" / "Reed Richards".
2. the roster: the LLM reads the whole cast list of one kind at once and
   groups names for the same one. Embeddings can't know that "Invisible Woman"
   is Sue Storm; a model that knows comics can.
Then a careful check: the LLM looks at each candidate pair and decides.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from litmapr.llm.base import Embedder, TextLLM, cosine
from litmapr.model import Entity, Story

DEFAULT_THRESHOLD = 0.85


class SameEntityVerdict(BaseModel):
    same: bool
    reason: str


@dataclass
class Candidate:
    a: Entity
    b: Entity
    score: float
    why: str = ""

    def __post_init__(self):
        if not self.why:
            self.why = f"{self.score:.2f}"


class SameGroup(BaseModel):
    names: list[str]
    reason: str


class Roster(BaseModel):
    groups: list[SameGroup]


ROSTER_SIZE = 150  # entries per prompt; bigger casts are read in slices


def entity_text(entity: Entity) -> str:
    """The text that gets embedded for an entity."""
    text = f"{entity.name} ({entity.kind})"
    if entity.aliases:
        text += f". Also called: {', '.join(entity.aliases)}"
    if entity.description:
        text += f". {entity.description}"
    return text


def find_duplicate_candidates(
    story: Story, embedder: Embedder, threshold: float = DEFAULT_THRESHOLD
) -> list[Candidate]:
    """Pairs of same-kind entities that look alike, most similar first."""
    if len(story.entities) < 2:
        return []
    vectors = embedder.embed([entity_text(e) for e in story.entities])
    candidates = []
    for i, a in enumerate(story.entities):
        for j in range(i + 1, len(story.entities)):
            b = story.entities[j]
            if a.kind != b.kind:
                continue  # a place is never the same thing as a person
            if a.all_names() & b.all_names():
                score = 1.0  # they share a name outright
            else:
                score = cosine(vectors[i], vectors[j])
            if score >= threshold:
                candidates.append(Candidate(a, b, score))
    return sorted(candidates, key=lambda c: -c.score)


def describe_for_check(story: Story, entity: Entity, limit: int = 3) -> str:
    lines = [entity_text(entity)]
    for mention in story.mentions_in_order(entity.id)[:limit]:
        lines.append(f"  - page {mention.page}: {mention.summary}")
    return "\n".join(lines)


def confirm_same(llm: TextLLM, story: Story, candidate: Candidate) -> SameEntityVerdict:
    prompt = f"""Two entries were extracted from the same comic series. Decide whether
they are the SAME {candidate.a.kind} under different names, or two different ones.
Different people who held the same title (two different Captain Americas) are NOT the same.

ENTRY A:
{describe_for_check(story, candidate.a)}

ENTRY B:
{describe_for_check(story, candidate.b)}

Answer with "same" (true or false) and a one-sentence "reason"."""
    raw = llm.complete_json(prompt, SameEntityVerdict.model_json_schema())
    return SameEntityVerdict.model_validate_json(raw)


def pick_keeper(story: Story, a: Entity, b: Entity) -> tuple[Entity, Entity]:
    """Keep the entity with more mentions (ties: the longer name). Returns (keep, drop)."""
    counts = {e.id: 0 for e in (a, b)}
    for mention in story.mentions:
        if mention.entity_id in counts:
            counts[mention.entity_id] += 1
    if (counts[b.id], len(b.name)) > (counts[a.id], len(a.name)):
        return b, a
    return a, b


def roster_line(story: Story, entity: Entity) -> str:
    line = f"- {entity.name}"
    if entity.aliases:
        line += f" (also called {', '.join(entity.aliases)})"
    if entity.description:
        line += f": {entity.description}"
    first = story.mentions_in_order(entity.id)[:1]
    if first:
        line += f" | first seen: {first[0].summary}"
    return line


def find_roster_candidates(llm: TextLLM, story: Story) -> list[Candidate]:
    """Ask the LLM to group same-kind names that are one entity (codename and
    real name, nickname, title). Each group becomes pairs for the careful check."""
    candidates = []
    for kind in dict.fromkeys(e.kind for e in story.entities):
        cast = [e for e in story.entities if e.kind == kind]
        for start in range(0, len(cast), ROSTER_SIZE):
            part = cast[start : start + ROSTER_SIZE]
            if len(part) < 2:
                continue
            candidates += _roster_part(llm, story, kind, part)
    return candidates


def _roster_part(llm, story, kind, part) -> list[Candidate]:
    listing = "\n".join(roster_line(story, e) for e in part)
    prompt = f"""Below is every {kind} found in a comic. Some are the SAME {kind} listed
twice under different names: a codename and a real name (Invisible Woman is Sue
Storm), a first name or nickname, a married name, a title.

Return "groups": each group lists the exact names from the list that are one
{kind}, with a short "reason". Only include groups of two or more, and only when
you are confident. A team is not one of its members. Different people who held
the same title are different. If nothing should be grouped, return no groups.

{listing}"""
    raw = llm.complete_json(prompt, Roster.model_json_schema())
    try:
        roster = Roster.model_validate_json(raw)
    except ValueError:
        return []  # a garbled answer just means no roster suggestions
    by_name = {e.name.lower(): e for e in part}
    pairs = []
    for group in roster.groups:
        members = list(
            {
                id(e): e for n in group.names if (e := by_name.get(n.strip().lower()))
            }.values()
        )
        for other in members[1:]:
            pairs.append(Candidate(members[0], other, 1.0, f"roster: {group.reason}"))
    return pairs


def combine(*lists: list[Candidate]) -> list[Candidate]:
    """All candidates, each pair once, in the order given."""
    seen: set[frozenset[str]] = set()
    out = []
    for candidates in lists:
        for c in candidates:
            key = frozenset((c.a.id, c.b.id))
            if c.a.id != c.b.id and key not in seen:
                seen.add(key)
                out.append(c)
    return out
