import time
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Annotated

import ollama
import typer

from litmapr.cache import PageCache, file_hash
from litmapr.extract.base import TextExtractor
from litmapr.extract.ollama_vision import DEFAULT_MODEL, OllamaVisionExtractor
from litmapr.ingest.base import Reader
from litmapr.ingest.cbz_reader import CbzReader
from litmapr.ingest.pdf_reader import PdfReader
from litmapr.output import default_output_path, write_pages_json
from litmapr.pipeline import read_pages

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
        print(f"Ollama error with model '{model}': {error.error}{after}")
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
) -> None:
    """Read every page of a book or comic and save the words to a JSON file."""
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
            for page, source in read_pages(reader.pages(), extractor, cache):
                results.append((page, source))
                elapsed = time.perf_counter() - start
                print(f"  page {page.number}: {source} ({elapsed:.1f}s)")
                start = time.perf_counter()
    else:
        results = [
            (page, "text_layer" if page.text else "none") for page in reader.pages()
        ]

    write_pages_json(output, path, source_hash, model if ocr else None, results)

    with_text = sum(1 for page, _ in results if page.text)
    print(f"found {len(results)} pages in {path} ({with_text} with text)")
    print(f"saved to {output}")
    if not ocr and with_text < len(results):
        print("Some pages have no text layer; add --ocr to read their images.")


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
def serve():
    """Start the 3D explorer."""
    print("Not implemented yet")
