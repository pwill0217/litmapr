from itertools import islice
from pathlib import Path
from typing import Annotated

import ollama
import typer

from litmapr.extract.base import TextExtractor
from litmapr.extract.ollama_vision import DEFAULT_MODEL, OllamaVisionExtractor
from litmapr.ingest.base import Reader
from litmapr.ingest.cbz_reader import CbzReader
from litmapr.ingest.pdf_reader import PdfReader

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


@app.command()
def ingest(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]):
    """Import a book or comic file."""
    reader = get_reader(path)
    pages = list(reader.pages())  # the reader does all the real work here
    with_text = sum(1 for page in pages if page.text)
    print(f"found {len(pages)} pages in {path} ({with_text} with text)")


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
        try:
            found = make_extractor(model).extract(found)
        except ConnectionError:
            print(
                "Can't reach Ollama. Is it running? Start the Ollama app and try again."
            )
            raise typer.Exit(code=1) from None
        except ollama.ResponseError as error:
            print(f"Ollama error with model '{model}': {error.error}")
            raise typer.Exit(code=1) from None

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
