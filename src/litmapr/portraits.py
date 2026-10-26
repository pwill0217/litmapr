"""Pictures for the dossier: cut a character, object, place or moment out of a page.

1. Pick a page where the entity appears (pages where it does the most first).
2. Ask the vision model where it is on that page (a bounding box).
3. Crop that box. For characters and objects, check the crop really shows
   them (small vision models sometimes point at the wrong person), then
   remove the background so only the figure is left.

Pictures are made from YOUR comics and stay in data/portraits on your
computer. `export` never includes them.
"""

import io
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from itertools import islice
from pathlib import Path

import ollama
from PIL import Image
from pydantic import BaseModel, Field

from litmapr.extract.ollama_vision import DEFAULT_MODEL
from litmapr.ingest.cbz_reader import CbzReader
from litmapr.ingest.pdf_reader import PdfReader
from litmapr.model import Entity, EntityKind, Mention, Story, Work

DEFAULT_PORTRAIT_DIR = Path("data") / "portraits"
LOCATE_SIZE = 1024  # longest side of the page sent to the model
PORTRAIT_SIZE = 512  # longest side of the saved picture
PADDING = 0.08  # extra room around the box, as a share of its size
MAX_FIGURE_SHARE = 0.45  # a "character" box bigger than this is a group shot
READERS = {".cbz": CbzReader, ".pdf": PdfReader}

# What to look for, per kind. Concepts are ideas: there is nothing to cut out.
WHAT_TO_FIND = {
    EntityKind.CHARACTER: (
        "Return the box around this ONE character's head and upper body. "
        "Leave out everyone else."
    ),
    EntityKind.DEVICE: "Return a tight box around this object only.",
    EntityKind.LOCATION: "Return the box around the panel that shows this place best.",
    EntityKind.FACTION: "Return the box around the panel that shows this group best.",
    EntityKind.EVENT: "Return the box around the panel that shows this moment.",
}
PICTURE_KINDS = tuple(WHAT_TO_FIND)
CUT_OUT_KINDS = {EntityKind.CHARACTER, EntityKind.DEVICE}  # background removed
FIGURE_KINDS = CUT_OUT_KINDS  # one figure: checked, and group shots refused


class Location(BaseModel):
    """Where the model found it: [left, top, right, bottom] in pixels."""

    found: bool
    box: list[int] = Field(default_factory=list)


class Check(BaseModel):
    """The model's answer to "is this really them?"."""

    who: str
    matches: bool


class CharacterLocator(ABC):
    @abstractmethod
    def locate(
        self, image: bytes, description: str, kind: EntityKind = EntityKind.CHARACTER
    ) -> list[int] | None:
        """Return [left, top, right, bottom] in the image's own pixels, or None."""

    def confirm(self, picture: bytes, description: str, kind: EntityKind) -> bool:
        """Does this crop really show what was asked for? Default: trust the box."""
        return True


def _jpeg(picture: Image.Image) -> bytes:
    buffer = io.BytesIO()
    picture.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class OllamaLocator(CharacterLocator):
    """Asks a local vision model (Qwen-VL can point at things) for a bounding box."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def _ask(self, prompt: str, image: bytes, schema: type[BaseModel]) -> BaseModel:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image]}],
            format=schema.model_json_schema(),
            options={"temperature": 0},
        )
        return schema.model_validate_json(response.message.content)

    def locate(
        self, image: bytes, description: str, kind: EntityKind = EntityKind.CHARACTER
    ) -> list[int] | None:
        picture = Image.open(io.BytesIO(image)).convert("RGB")
        # Send a smaller copy, and ask for coordinates in THAT copy's pixels;
        # then scale the answer back up to the full page.
        scale = min(1.0, LOCATE_SIZE / max(picture.size))
        small = picture.resize(
            (round(picture.width * scale), round(picture.height * scale))
        )
        prompt = (
            f"This comic page is {small.width} pixels wide and {small.height} pixels tall. "
            f"Find this {kind}: {description}\n"
            f"{WHAT_TO_FIND[kind]} Give it as [left, top, right, bottom] in pixels "
            "of this image. Pick the clearest, largest view. "
            "If it is not shown on this page, set found to false."
        )
        answer = self._ask(prompt, _jpeg(small), Location)
        if not answer.found or len(answer.box) != 4:
            return None
        return [round(value / scale) for value in answer.box]

    def confirm(self, picture: bytes, description: str, kind: EntityKind) -> bool:
        if kind not in FIGURE_KINDS:
            return True  # a panel of a place or a moment: nothing to check
        thing = "person" if kind == EntityKind.CHARACTER else "object"
        prompt = (
            f"This is a crop from a comic page. We are looking for this {kind}: "
            f"{description}\n"
            f'First say in "who" what the main {thing} in the crop is, in a few words. '
            f'Then set "matches" to true only if the crop clearly shows the {kind} '
            f"described, as the main {thing}. Set it to false if it shows someone or "
            f"something else, several {thing}s of equal size, or nothing clear."
        )
        return self._ask(prompt, picture, Check).matches


def describe(entity: Entity, mention: Mention | None = None) -> str:
    text = entity.name
    if entity.aliases:
        text += f" (also called {', '.join(entity.aliases)})"
    if entity.description:
        text += f". {entity.description}"
    if mention is not None:
        text += f". On this page: {mention.summary}"
    return text


def crop(image: bytes, box: list[int], max_share: float = 1.0) -> Image.Image | None:
    """Cut the box (plus a little padding) out of the page.

    None if the box is nonsense, or covers more than `max_share` of the page.
    """
    page = Image.open(io.BytesIO(image)).convert("RGB")
    left, top, right, bottom = box
    if right - left < 16 or bottom - top < 16:
        return None
    if (right - left) * (bottom - top) > max_share * page.width * page.height:
        return None  # too big to be one figure: probably a crowd
    pad_x, pad_y = (right - left) * PADDING, (bottom - top) * PADDING
    area = (
        max(0, round(left - pad_x)),
        max(0, round(top - pad_y)),
        min(page.width, round(right + pad_x)),
        min(page.height, round(bottom + pad_y)),
    )
    if area[2] - area[0] < 16 or area[3] - area[1] < 16:
        return None  # the box was mostly off the page
    piece = page.crop(area)
    piece.thumbnail((PORTRAIT_SIZE, PORTRAIT_SIZE))
    return piece


def remove_background(picture: Image.Image) -> tuple[Image.Image, bool]:
    """Cut out the background with rembg, if it's installed (`uv sync --extra portraits`).

    Returns the picture and whether the background was removed.
    """
    try:
        from rembg import remove
    except ImportError:
        return picture, False
    return remove(picture), True


def page_image(path: Path, page: int) -> bytes | None:
    """Read one page of a comic file again (pages aren't stored anywhere)."""
    reader_class = READERS.get(path.suffix.lower())
    if reader_class is None or not path.exists():
        return None
    found = next(islice(reader_class(path).pages(), page - 1, None), None)
    return found.image if found else None


def portrait_path(entity_id: str, folder: Path = DEFAULT_PORTRAIT_DIR) -> Path:
    return folder / f"{entity_id}.png"


def best_pages(story: Story, entity: Entity) -> list[Mention]:
    """One mention per page, busiest pages first (more going on = more likely
    drawn big), earliest first among equals."""
    mentions = story.mentions_in_order(entity.id)
    per_page = Counter((m.work_id, m.page) for m in mentions)
    seen: set[tuple[str, int]] = set()
    unique = []
    for mention in mentions:
        key = (mention.work_id, mention.page)
        if key not in seen:
            seen.add(key)
            unique.append(mention)
    # sorted() is stable, so equal counts keep reading order.
    return sorted(unique, key=lambda m: -per_page[(m.work_id, m.page)])


def make_portrait(
    story: Story,
    entity: Entity,
    locator: CharacterLocator,
    folder: Path = DEFAULT_PORTRAIT_DIR,
    tries: int = 4,
    background_remover=remove_background,
) -> tuple[Path | None, str]:
    """Try the entity's best pages until one gives a picture that checks out.

    Returns (where the picture was saved or None, a short note on what happened).
    """
    if entity.kind not in PICTURE_KINDS:
        return None, f"no picture for a {entity.kind}"
    works = {w.id: w for w in story.works}
    attempts = rejected = 0
    for mention in best_pages(story, entity):
        work = works[mention.work_id]
        if not work.source_path:
            continue
        image = page_image(Path(work.source_path), mention.page)
        if image is None:
            continue
        attempts += 1
        result = _cut_out(
            image,
            entity,
            describe(entity, mention),
            locator,
            folder,
            background_remover,
        )
        if result == "wrong":
            rejected += 1
        elif result is not None:
            return result[0], _note(work, mention.page, entity, result[1])
        if attempts >= tries:
            break
    if attempts == 0:
        return None, "no comic file found for any appearance (moved or deleted?)"
    if rejected:
        return None, (
            f"not found on {attempts} page(s) "
            f"({rejected} crop(s) showed someone or something else)"
        )
    return None, f"not found on {attempts} page(s)"


def portrait_from_page(
    entity: Entity,
    work: Work,
    page: int,
    locator: CharacterLocator,
    folder: Path = DEFAULT_PORTRAIT_DIR,
    background_remover=remove_background,
) -> tuple[Path | None, str]:
    """Look on one page you picked (a page of the issue, not of the file)."""
    file_page = work.first_page + page - 1 if work.first_page else page
    if work.last_page and file_page > work.last_page:
        return None, f"{work.series} #{work.issue} has no page {page}"
    image = page_image(Path(work.source_path), file_page) if work.source_path else None
    if image is None:
        return None, "couldn't read that page (comic file moved or deleted?)"
    result = _cut_out(
        image, entity, describe(entity), locator, folder, background_remover
    )
    where = f"{work.series} #{work.issue} p{page}"
    if result is None:
        return None, f"not found on {where}"
    if result == "wrong":
        return None, f"the crop on {where} showed someone or something else"
    return result[0], _note(work, file_page, entity, result[1])


def _cut_out(image, entity, description, locator, folder, background_remover):
    """Find, crop, check, cut out and save.

    Returns (path, background removed?), "wrong" if the check failed, or None.
    """
    kind = EntityKind(entity.kind)
    box = locator.locate(image, description, kind)
    figure = kind in FIGURE_KINDS
    picture = crop(image, box, MAX_FIGURE_SHARE if figure else 1.0) if box else None
    if picture is None:
        return None
    if not locator.confirm(_jpeg(picture), description, kind):
        return "wrong"
    cut_out = False
    if kind in CUT_OUT_KINDS:
        picture, cut_out = background_remover(picture)
    target = portrait_path(entity.id, folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    picture.save(target, format="PNG")
    return target, cut_out


def _note(work: Work, file_page: int, entity: Entity, cut_out: bool) -> str:
    where = f"{work.series} #{work.issue} p{work.local_page(file_page)}"
    if cut_out or entity.kind not in CUT_OUT_KINDS:
        return where
    return f"{where} (background kept: rembg not installed)"


def portrait_from_bytes(
    entity: Entity,
    data: bytes,
    folder: Path = DEFAULT_PORTRAIT_DIR,
    cut_out: bool | None = None,
) -> Path:
    """Use your own picture. Raises ValueError if it isn't a picture.

    The background is removed for characters and objects unless cut_out says otherwise.
    """
    try:
        picture = Image.open(io.BytesIO(data))
        picture.load()
    except Exception as error:
        raise ValueError("That file isn't a picture litmapr can read.") from error
    picture = picture.convert("RGBA")
    picture.thumbnail((PORTRAIT_SIZE, PORTRAIT_SIZE))
    if cut_out is None:
        cut_out = entity.kind in CUT_OUT_KINDS
    if cut_out:
        picture, _ = remove_background(picture)
    target = portrait_path(entity.id, folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    picture.save(target, format="PNG")
    return target


def portrait_from_file(
    entity: Entity,
    image_path: Path,
    folder: Path = DEFAULT_PORTRAIT_DIR,
    cut_out: bool | None = None,
) -> Path:
    """Use your own picture file as an entity's portrait."""
    return portrait_from_bytes(entity, Path(image_path).read_bytes(), folder, cut_out)


def remove_portrait(entity_id: str, folder: Path = DEFAULT_PORTRAIT_DIR) -> bool:
    """Delete a wrong picture. Returns whether there was one."""
    path = portrait_path(entity_id, folder)
    if not path.exists():
        return False
    path.unlink()
    return True


def inherit_portrait(
    keep_id: str, drop_id: str, folder: Path = DEFAULT_PORTRAIT_DIR
) -> None:
    """After a merge: the kept entity takes the dropped one's picture if it has
    none of its own. The dropped picture is removed either way."""
    keep, drop = portrait_path(keep_id, folder), portrait_path(drop_id, folder)
    if not drop.exists():
        return
    if keep.exists():
        drop.unlink()
    else:
        shutil.move(drop, keep)
