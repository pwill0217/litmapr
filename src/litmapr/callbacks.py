"""Phase 4b: find later moments that refer back to earlier ones.

Same two-step idea as resolve.py: embeddings shortlist earlier moments that
sound related, then the LLM decides whether the later one really calls back.
Only moments from EARLIER works are considered, so callbacks always point
back in reading order.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from litmapr.llm.base import Embedder, TextLLM, cosine
from litmapr.model import Mention, Story

DEFAULT_THRESHOLD = 0.75
PER_MENTION = 2  # how many earlier moments to check for each later one


class CallbackVerdict(BaseModel):
    is_callback: bool
    reason: str


@dataclass
class CallbackCandidate:
    later: Mention
    earlier: Mention
    score: float


def mention_text(story: Story, mention: Mention) -> str:
    entity = next(e for e in story.entities if e.id == mention.entity_id)
    text = f"{entity.name}: {mention.summary}"
    if mention.quote:
        text += f' "{mention.quote}"'
    return text


def ordered_mentions(story: Story) -> list[tuple[int, Mention]]:
    """Every mention with the reading position of its work."""
    position = {work.id: i for i, work in enumerate(story.reading_order())}
    found = [(position[m.work_id], m) for m in story.mentions if m.work_id in position]
    return sorted(found, key=lambda pair: (pair[0], pair[1].page))


def find_callback_candidates(
    story: Story,
    embedder: Embedder,
    threshold: float = DEFAULT_THRESHOLD,
    per_mention: int = PER_MENTION,
) -> list[CallbackCandidate]:
    ordered = ordered_mentions(story)
    if len(ordered) < 2:
        return []
    vectors = embedder.embed([mention_text(story, m) for _, m in ordered])
    already = {(c.from_mention_id, c.to_mention_id) for c in story.callbacks}

    candidates = []
    for i, (later_pos, later) in enumerate(ordered):
        scored = []
        for j in range(i):
            earlier_pos, earlier = ordered[j]
            if earlier_pos >= later_pos or (later.id, earlier.id) in already:
                continue  # only earlier works, and only new pairs
            score = cosine(vectors[i], vectors[j])
            if score >= threshold:
                scored.append(CallbackCandidate(later, earlier, score))
        scored.sort(key=lambda c: -c.score)
        candidates.extend(scored[:per_mention])
    return sorted(candidates, key=lambda c: -c.score)


def confirm_callback(
    llm: TextLLM, story: Story, candidate: CallbackCandidate
) -> CallbackVerdict:
    works = {w.id: w for w in story.works}
    earlier_work = works[candidate.earlier.work_id]
    later_work = works[candidate.later.work_id]
    prompt = f"""In a comic series, does the LATER moment deliberately refer back to,
pay off, or continue the EARLIER moment? Sharing a character is not enough:
it must build on that specific earlier event, object, idea or relationship.

EARLIER ({earlier_work.series} #{earlier_work.issue}, page {candidate.earlier.page}):
{mention_text(story, candidate.earlier)}

LATER ({later_work.series} #{later_work.issue}, page {candidate.later.page}):
{mention_text(story, candidate.later)}

Answer with "is_callback" (true or false) and a one-sentence "reason"
that a reader would find useful."""
    raw = llm.complete_json(prompt, CallbackVerdict.model_json_schema())
    return CallbackVerdict.model_validate_json(raw)
