"""Phase 6: the narrator explains a plot point to the reader.

Retrieval-augmented generation (RAG): instead of asking the model what it
"knows" about Red Hulk (which may be wrong, or spoil later issues), we
RETRIEVE exactly what the map knows up to where the reader is, put it in
the prompt, and tell the model to use nothing else.
"""

import hashlib
from collections.abc import Iterator

from litmapr.cache import ResponseCache
from litmapr.llm.base import TextLLM
from litmapr.model import Story, Work


def work_label(work: Work) -> str:
    return f"{work.series} #{work.issue}"


def page_label(work: Work, page: int) -> str:
    """'FF #9 p14': the page within the issue, even inside an omnibus."""
    return f"{work_label(work)} p{work.local_page(page)}"


def visible_work_ids(story: Story, upto: int | None) -> set[str]:
    """Works the reader has reached. `upto` is a reading position (0 = first
    work in reading order); None means everything."""
    ordered = story.reading_order()
    if upto is not None:
        ordered = ordered[: upto + 1]
    return {work.id for work in ordered}


def build_context(story: Story, entity_id: str, upto: int | None) -> str:
    """Everything the map knows about one entity, limited to what the reader has read."""
    visible = visible_work_ids(story, upto)
    works = {w.id: w for w in story.works}
    names = {e.id: e.name for e in story.entities}
    entity = next(e for e in story.entities if e.id == entity_id)

    lines = [f"ENTITY: {entity.name} ({entity.kind})"]
    if entity.aliases:
        lines.append(f"ALSO CALLED: {', '.join(entity.aliases)}")
    if entity.description:
        lines.append(f"DESCRIPTION: {entity.description}")

    mentions = [m for m in story.mentions_in_order(entity_id) if m.work_id in visible]
    if mentions:
        lines.append("APPEARANCES, IN READING ORDER:")
        for m in mentions:
            line = f"- {page_label(works[m.work_id], m.page)}: {m.summary}"
            if m.quote:
                line += f' Quote: "{m.quote}"'
            lines.append(line)

    relations = [
        r
        for r in story.relations
        if entity_id in (r.source_id, r.target_id) and r.work_id in visible
    ]
    if relations:
        lines.append("RELATIONSHIPS:")
        for r in relations:
            lines.append(
                f"- {names[r.source_id]} -> {names[r.target_id]} ({r.kind}): {r.label} "
                f"[{page_label(works[r.work_id], r.page)}]"
            )

    mention_ids = {m.id for m in mentions}
    by_id = {m.id: m for m in story.mentions}
    callbacks = [
        c
        for c in story.callbacks
        if (c.from_mention_id in mention_ids or c.to_mention_id in mention_ids)
        and by_id[c.from_mention_id].work_id in visible
        and by_id[c.to_mention_id].work_id in visible
    ]
    if callbacks:
        lines.append("CALLBACKS (later moments referring back to earlier ones):")
        for c in callbacks:
            later, earlier = by_id[c.from_mention_id], by_id[c.to_mention_id]
            lines.append(
                f"- {page_label(works[later.work_id], later.page)} refers back to "
                f"{page_label(works[earlier.work_id], earlier.page)}: {c.reason}"
            )
    return "\n".join(lines)


# How the narrator sounds. Both are written to be read ALOUD as well as on screen.
STYLES = {
    "storyteller": """You are the narrator of this comic: the voice in the caption boxes,
telling the story to a reader by the fire. Speak in present tense, straight to the
reader ("you"), with a storyteller's rhythm: vivid, a little dramatic, never
flowery. Build to the most intriguing open question and end on it.""",
    "briefing": """You are an intelligence officer briefing an agent on this file.
Clipped, confident, second person ("You need to know three things."). Facts first,
then what to watch for. End with one line on the open question.""",
}
DEFAULT_STYLE = "storyteller"


def build_prompt(
    story: Story, entity_id: str, upto: int | None, style: str = DEFAULT_STYLE
) -> str:
    ordered = story.reading_order()
    if upto is not None and ordered:
        progress = f"The reader has read up to {work_label(ordered[min(upto, len(ordered) - 1)])}."
    else:
        progress = "The reader has read everything in the map."
    voice = STYLES.get(style, STYLES[DEFAULT_STYLE])
    return f"""{voice}

The reader just opened the entry below. In about 150 words, tell them who or what
it is, how it has developed so far, and how it ties into the rest of the story.
Point out callbacks to earlier moments.

This will be spoken aloud, so write for the ear: short sentences, no headings,
lists, brackets or abbreviations. Say issues the way a person would ("back in
issue 570"), never "FF #570 p5". Quote a line of dialogue if one is striking.

{progress} Use ONLY the facts below. Do not add anything you know from elsewhere,
and never hint at events that are not listed: that would spoil the story.

{build_context(story, entity_id, upto)}
"""


def _key(llm: TextLLM, prompt: str) -> str:
    # The key covers the whole prompt: if the map learns something new,
    # the facts change, and the narration is written again.
    return hashlib.sha256(f"{llm.model}\n{prompt}".encode()).hexdigest()[:24]


def narrate(
    llm: TextLLM,
    story: Story,
    entity_id: str,
    upto: int | None,
    cache: ResponseCache | None = None,
    style: str = DEFAULT_STYLE,
) -> str:
    return "".join(narrate_stream(llm, story, entity_id, upto, cache, style)).strip()


def narrate_stream(
    llm: TextLLM,
    story: Story,
    entity_id: str,
    upto: int | None,
    cache: ResponseCache | None = None,
    style: str = DEFAULT_STYLE,
) -> Iterator[str]:
    """Yield the narration piece by piece as the model writes it.

    A cached narration comes back in one piece. A new one is cached once it
    is complete, so a narration cut short is never saved.
    """
    prompt = build_prompt(story, entity_id, upto, style)
    key = _key(llm, prompt)
    if cache is not None and (cached := cache.get(key)) is not None:
        yield after_thinking(cached).strip()  # saved before thinking was filtered
        return
    pieces = []
    for piece in without_thinking(llm.stream_text(prompt)):
        pieces.append(piece)
        yield piece
    text = after_thinking("".join(pieces)).strip()
    if cache is not None and text:
        cache.put(key, text)


THINK_END = "</think>"


def after_thinking(text: str) -> str:
    """Drop a thinking model's reasoning: everything up to its last </think>."""
    return text.rsplit(THINK_END, 1)[-1]


def without_thinking(pieces: Iterator[str]) -> Iterator[str]:
    """Hold back a reasoning block that starts with <think>. (A model that
    reasons without the opening tag can't be caught here; the browser drops
    everything before a </think> when one shows up.)"""
    held = ""
    for piece in pieces:
        if held is None:
            yield piece
            continue
        held += piece
        start = held.lstrip()
        if start.startswith("<think>") or "<think>".startswith(start):
            if THINK_END in held:
                rest = after_thinking(held).lstrip()
                held = None
                if rest:
                    yield rest
            continue  # still inside the reasoning
        yield held
        held = None
    if held:
        yield after_thinking(held)
